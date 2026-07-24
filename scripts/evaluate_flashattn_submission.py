#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MACA_PATH", "/opt/maca")
os.environ["PATH"] = f"/opt/maca/bin:/opt/maca/mxgpu_llvm/bin:{os.environ.get('PATH', '')}"
os.environ["LD_LIBRARY_PATH"] = f"/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

torch = None
flash_attn_with_kvcache = None
load = None


BINDING_CPP = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <stdint.h>

extern "C" void run_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache_paged,
    const __nv_bfloat16* v_cache_paged,
    __nv_bfloat16* output,
    const int32_t* cache_seqlens,
    const int32_t* block_table,
    int64_t batch_size,
    int64_t seqlen_k,
    int64_t seqlen_q,
    int64_t num_heads,
    int64_t num_heads_k,
    int64_t headdim,
    int64_t page_block_size,
    int64_t num_blocks,
    int64_t causal);

void run_kernel_torch(
    torch::Tensor q,
    torch::Tensor k_cache_paged,
    torch::Tensor v_cache_paged,
    torch::Tensor output,
    torch::Tensor cache_seqlens,
    torch::Tensor block_table,
    int64_t batch_size,
    int64_t seqlen_k,
    int64_t seqlen_q,
    int64_t num_heads,
    int64_t num_heads_k,
    int64_t headdim,
    int64_t page_block_size,
    int64_t num_blocks,
    int64_t causal) {
    run_kernel(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_cache_paged.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v_cache_paged.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        cache_seqlens.data_ptr<int32_t>(),
        block_table.data_ptr<int32_t>(),
        batch_size,
        seqlen_k,
        seqlen_q,
        num_heads,
        num_heads_k,
        headdim,
        page_block_size,
        num_blocks,
        causal);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_kernel", &run_kernel_torch, "FlashAttention KVCACHE run_kernel wrapper");
}
"""


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def compile_candidate(src: Path, build_dir: Path) -> Any:
    binding = build_dir / "binding.cpp"
    binding.write_text(BINDING_CPP, encoding="utf-8")
    name = f"flashattn_candidate_{int(time.time() * 1000)}"
    return load(
        name=name,
        sources=[str(binding), str(src)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        build_directory=str(build_dir),
        verbose=False,
    )


def make_case(
    *,
    batch_size: int,
    seqlen_kv: int,
    seqlen_q: int,
    num_heads: int,
    num_heads_k: int,
    headdim: int,
    page_block_size: int,
    dtype: torch.dtype,
    device: str,
    task_profile: str = "legacy",
) -> dict[str, Any]:
    blocks_per_batch = math.ceil(seqlen_kv / page_block_size)
    if task_profile == "contest":
        num_blocks = batch_size * blocks_per_batch
    else:
        num_blocks = blocks_per_batch * batch_size * 3
        num_blocks = max(1024, num_blocks)
        if num_blocks % batch_size:
            num_blocks += batch_size - (num_blocks % batch_size)
    torch.manual_seed(0)
    q = torch.randn(batch_size, seqlen_q, num_heads, headdim, device=device, dtype=dtype)
    k_cache_paged = torch.randn(num_blocks, page_block_size, num_heads_k, headdim, device=device, dtype=dtype)
    v_cache_paged = torch.randn(num_blocks, page_block_size, num_heads_k, headdim, device=device, dtype=dtype)
    block_table = torch.randperm(num_blocks, dtype=torch.int32, device=device).reshape(batch_size, -1)
    if task_profile == "contest":
        vals = []
        for b in range(batch_size):
            if b == 0:
                vals.append(seqlen_kv)
            elif b == 1:
                vals.append(1)
            else:
                span = max(1, seqlen_kv - 1)
                vals.append(1 + ((b * 9973 + seqlen_kv // 3) % span))
        cache_seqlens = torch.tensor(vals, dtype=torch.int32, device=device)
    else:
        cache_seqlens = torch.full((batch_size,), seqlen_kv, dtype=torch.int32, device=device)
    return {
        "q": q,
        "k_cache_paged": k_cache_paged,
        "v_cache_paged": v_cache_paged,
        "block_table": block_table,
        "cache_seqlens": cache_seqlens,
        "num_blocks": num_blocks,
    }


def time_cuda(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def profile_cuda(fn, warmup: int, repeat: int) -> list[dict[str, Any]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            for _ in range(repeat):
                fn()
        torch.cuda.synchronize()
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]
    events = []
    for evt in prof.key_averages():
        device_time = getattr(evt, "device_time", None)
        if device_time is None:
            device_time = getattr(evt, "cuda_time_total", None)
        events.append(
            {
                "name": evt.key,
                "device_time_us": float(device_time or 0.0),
                "count": int(getattr(evt, "count", 0) or 0),
            }
        )
    events.sort(key=lambda item: item.get("device_time_us", 0.0), reverse=True)
    return events[:16]


def evaluate_case(
    module: Any,
    shape: dict[str, int],
    warmup: int,
    repeat: int,
    atol: float,
    rtol: float,
    task_profile: str,
) -> dict[str, Any]:
    device = "cuda"
    dtype = torch.bfloat16
    page_block_size = shape["page_block_size"]
    data = make_case(
        batch_size=shape["batch_size"],
        seqlen_kv=shape["seqlen_kv"],
        seqlen_q=shape["seqlen_q"],
        num_heads=shape["num_heads"],
        num_heads_k=shape["num_heads_k"],
        headdim=shape["headdim"],
        page_block_size=page_block_size,
        dtype=dtype,
        device=device,
        task_profile=task_profile,
    )
    q = data["q"]
    k_cache_paged = data["k_cache_paged"]
    v_cache_paged = data["v_cache_paged"]
    cache_seqlens = data["cache_seqlens"]
    block_table = data["block_table"]
    num_blocks = data["num_blocks"]
    causal = False

    def run_ref():
        return flash_attn_with_kvcache(
            q,
            k_cache_paged,
            v_cache_paged,
            None,
            None,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=None,
            block_table=block_table,
            causal=causal,
            window_size=(-1, -1),
            rotary_interleaved=False,
            alibi_slopes=None,
            num_splits=0 if task_profile == "contest" else 1,
        )

    output = torch.empty_like(q)

    def run_candidate():
        output.zero_()
        module.run_kernel(
            q,
            k_cache_paged,
            v_cache_paged,
            output,
            cache_seqlens,
            block_table,
            shape["batch_size"],
            shape["seqlen_kv"],
            shape["seqlen_q"],
            shape["num_heads"],
            shape["num_heads_k"],
            shape["headdim"],
            page_block_size,
            num_blocks,
            int(causal),
        )
        return output

    ref = run_ref()
    cand = run_candidate()
    torch.cuda.synchronize()
    max_abs_err = float((ref.float() - cand.float()).abs().max().item())
    ok = bool(torch.allclose(ref.float(), cand.float(), atol=atol, rtol=rtol))

    ref_ms = time_cuda(run_ref, warmup=warmup, repeat=repeat)
    cand_ms = time_cuda(run_candidate, warmup=warmup, repeat=repeat) if ok else None
    speedup = (ref_ms / cand_ms) if (ok and cand_ms and cand_ms > 0) else None
    return {
        **shape,
        "ok": ok,
        "max_abs_err": max_abs_err,
        "ref_ms": ref_ms,
        "candidate_ms": cand_ms,
        "speedup": speedup,
    }


def evaluate_case_profile(module: Any, shape: dict[str, int], warmup: int, repeat: int, task_profile: str) -> dict[str, Any]:
    device = "cuda"
    dtype = torch.bfloat16
    page_block_size = shape["page_block_size"]
    data = make_case(
        batch_size=shape["batch_size"],
        seqlen_kv=shape["seqlen_kv"],
        seqlen_q=shape["seqlen_q"],
        num_heads=shape["num_heads"],
        num_heads_k=shape["num_heads_k"],
        headdim=shape["headdim"],
        page_block_size=page_block_size,
        dtype=dtype,
        device=device,
        task_profile=task_profile,
    )
    q = data["q"]
    k_cache_paged = data["k_cache_paged"]
    v_cache_paged = data["v_cache_paged"]
    cache_seqlens = data["cache_seqlens"]
    block_table = data["block_table"]
    output = torch.empty_like(q)
    num_blocks = data["num_blocks"]

    def run_candidate():
        output.zero_()
        module.run_kernel(
            q,
            k_cache_paged,
            v_cache_paged,
            output,
            cache_seqlens,
            block_table,
            shape["batch_size"],
            shape["seqlen_kv"],
            shape["seqlen_q"],
            shape["num_heads"],
            shape["num_heads_k"],
            shape["headdim"],
            page_block_size,
            num_blocks,
            0,
        )
        return output

    return {**shape, "candidate_events": profile_cuda(run_candidate, warmup=warmup, repeat=repeat)}


def default_shapes(smoke: bool) -> list[dict[str, int]]:
    batch_sizes = [1, 2, 4, 8] if smoke else [1, 2, 4, 8, 16, 32, 64, 128]
    seqs = [512, 2048, 8192] if smoke else [512, 1024, 2048, 4096, 8192, 16384]
    shapes = []
    for seqlen_kv in seqs:
        for batch_size in batch_sizes:
            shapes.append(
                {
                    "batch_size": batch_size,
                    "seqlen_kv": seqlen_kv,
                    "seqlen_q": 1,
                    "num_heads": 8,
                    "num_heads_k": 8,
                    "headdim": 256,
                    "page_block_size": 16,
                }
            )
    return shapes


def contest_shapes(smoke: bool) -> list[dict[str, int]]:
    raw_cases = [
        (4, 1, 8),
        (4, 2, 8),
        (16, 17, 4),
        (64, 8, 4),
        (16, 141, 4),
        (32, 362, 8),
        (64, 2048, 4),
        (16, 4096, 4),
        (32, 4096, 8),
        (1, 8192, 4),
        (16, 12251, 4),
        (8, 32768, 8),
        (1, 58966, 8),
        (1, 61519, 4),
    ]
    if smoke:
        raw_cases = raw_cases[:5]
    return [
        {
            "batch_size": batch,
            "seqlen_kv": seqlen,
            "seqlen_q": 1,
            "num_heads": 32,
            "num_heads_k": kv_heads,
            "headdim": 128,
            "page_block_size": 16,
        }
        for batch, seqlen, kv_heads in raw_cases
    ]


def load_shapes(path: Path | None, smoke: bool, task_profile: str) -> list[dict[str, int]]:
    if path is None:
        if task_profile == "contest":
            return contest_shapes(smoke)
        return default_shapes(smoke)
    obj = json.loads(path.read_text(encoding="utf-8"))
    cases = obj.get("cases") if isinstance(obj, dict) else obj
    if not isinstance(cases, list):
        raise ValueError(f"shapes file must contain a list or a cases list: {path}")
    shapes: list[dict[str, int]] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        shapes.append(
            {
                "batch_size": int(case["batch_size"]),
                "seqlen_kv": int(case["seqlen_kv"]),
                "seqlen_q": int(case.get("seqlen_q", 1)),
                "num_heads": int(case.get("num_heads", 32 if task_profile == "contest" else 8)),
                "num_heads_k": int(case.get("num_heads_k", 4 if task_profile == "contest" else 8)),
                "headdim": int(case.get("headdim", 128 if task_profile == "contest" else 256)),
                "page_block_size": int(case.get("page_block_size", 16)),
            }
        )
    if not shapes:
        raise ValueError(f"no valid shapes in {path}")
    return shapes


def add_region_metrics(result: dict[str, Any]) -> None:
    records = result.get("records", [])
    measured = [r for r in records if r.get("ok") and r.get("speedup") is not None]
    if not measured:
        result["region_scores"] = {}
        result["worst_cases"] = []
        result["best_cases"] = []
        return

    def avg_speedup(predicate) -> float | None:
        vals = [float(r["speedup"]) for r in measured if predicate(r)]
        if not vals:
            return None
        return sum(vals) / len(vals)

    result["region_scores"] = {
        "short_kv_le_2048": avg_speedup(lambda r: int(r["seqlen_kv"]) <= 2048),
        "mid_kv_4096": avg_speedup(lambda r: int(r["seqlen_kv"]) == 4096),
        "long_kv_ge_8192": avg_speedup(lambda r: int(r["seqlen_kv"]) >= 8192),
        "small_batch_le_4": avg_speedup(lambda r: int(r["batch_size"]) <= 4),
        "large_batch_ge_8": avg_speedup(lambda r: int(r["batch_size"]) >= 8),
        "target_long_small": avg_speedup(lambda r: int(r["batch_size"]) <= 4 and int(r["seqlen_kv"]) >= 4096),
    }
    result["worst_cases"] = [
        compact_case(r) for r in sorted(measured, key=lambda x: float(x["speedup"]))[:8]
    ]
    result["best_cases"] = [
        compact_case(r) for r in sorted(measured, key=lambda x: float(x["speedup"]), reverse=True)[:8]
    ]


def compact_case(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch_size": record.get("batch_size"),
        "seqlen_kv": record.get("seqlen_kv"),
        "speedup": record.get("speedup"),
        "candidate_ms": record.get("candidate_ms"),
        "ref_ms": record.get("ref_ms"),
        "max_abs_err": record.get("max_abs_err"),
    }


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate a FlashAttention KVCACHE CUDAMaca submission")
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--profile", action="store_true", help="Write lightweight torch-profiler feedback for weakest cases.")
    parser.add_argument("--profile-limit", type=int, default=4)
    parser.add_argument("--shapes-file", default=None, help="Optional JSON list of benchmark shapes.")
    parser.add_argument("--benchmark-kind", default="official", choices=["official", "auxiliary"])
    parser.add_argument("--task-profile", default="legacy", choices=["legacy", "contest"])
    args = parser.parse_args()

    global torch, flash_attn_with_kvcache, load
    import torch as torch_mod
    from flash_attn.flash_attn_interface import flash_attn_with_kvcache as flash_attn_with_kvcache_mod
    from torch.utils.cpp_extension import load as load_mod

    torch = torch_mod
    flash_attn_with_kvcache = flash_attn_with_kvcache_mod
    load = load_mod

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(args.src).resolve()
    records: list[dict[str, Any]] = []
    shapes_file = Path(args.shapes_file) if args.shapes_file else None
    shapes = load_shapes(shapes_file, args.smoke, args.task_profile)
    result: dict[str, Any] = {
        "src": str(src),
        "passed": False,
        "benchmark_kind": args.benchmark_kind,
        "task_profile": args.task_profile,
        "shapes_file": str(shapes_file) if shapes_file else None,
        "records": records,
    }

    try:
        with tempfile.TemporaryDirectory(prefix="flashattn_candidate_build_") as tmp:
            module = compile_candidate(src, Path(tmp))
            for shape in shapes:
                try:
                    record = evaluate_case(module, shape, args.warmup, args.repeat, args.atol, args.rtol, args.task_profile)
                except Exception as exc:
                    record = {**shape, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
                records.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
            if args.profile:
                weak_shapes = [
                    {
                        "batch_size": int(r["batch_size"]),
                        "seqlen_kv": int(r["seqlen_kv"]),
                        "seqlen_q": int(r["seqlen_q"]),
                        "num_heads": int(r["num_heads"]),
                        "num_heads_k": int(r["num_heads_k"]),
                        "headdim": int(r["headdim"]),
                        "page_block_size": int(r["page_block_size"]),
                    }
                    for r in sorted(
                        [r for r in records if r.get("ok") and r.get("speedup") is not None],
                        key=lambda item: float(item["speedup"]),
                    )[: max(0, args.profile_limit)]
                ]
                profiler_feedback = {
                    "source": "torch_profiler",
                    "available": True,
                    "profiled_cases": [
                        evaluate_case_profile(module, shape, max(1, args.warmup), max(1, min(args.repeat, 5)), args.task_profile)
                        for shape in weak_shapes
                    ],
                    "notes": [
                        "This is a lightweight torch.profiler trace over weakest benchmark cases.",
                        "Use it as profiler feedback, not as a full MACA hardware-counter replacement.",
                    ],
                }
                write_json(out_dir / "profiler_feedback.json", profiler_feedback)
    except Exception as exc:
        result["compile_error"] = f"{type(exc).__name__}: {exc}"

    passed_records = [r for r in records if r.get("ok")]
    result["passed"] = bool(records) and len(passed_records) == len(records)
    if passed_records:
        result["score"] = sum(float(r["speedup"]) for r in passed_records if r.get("speedup") is not None) / len(passed_records)
        result["avg_candidate_ms"] = sum(float(r["candidate_ms"]) for r in passed_records) / len(passed_records)
        result["avg_ref_ms"] = sum(float(r["ref_ms"]) for r in passed_records) / len(passed_records)
    else:
        result["score"] = None
    add_region_metrics(result)

    write_json(out_dir / "metrics.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, ensure_ascii=False), flush=True)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
