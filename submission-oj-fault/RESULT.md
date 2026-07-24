# OJ Contest Submission (FAULTY - ARCHIVED)

This is the ARCHIVED faulty OJ track. It was incorrectly based on legacy iter_017 instead of the original baseline.

The correct fresh OJ track is at `submission-oj/`.

## Candidate

```text
flashattn_kvcache_decode_optimized.cu
flashattn_kvcache_decode_optimized.zip
```

## Current Status

The current files are copied from the latest legacy benchmark best and validated under the OJ profile as a baseline.

```text
source: submission/ legacy best from iter_017_full_upgrade_proxy
contest-profile correctness: passed
contest-profile score: 0.5620690623642882
```

Latest OJ agent run:

```text
run: experiments-oj/operator_loops/iter_001_oj_agent_retry
baseline score in that run: 0.5609105101614815
round 000: rejected, compile failed
round 001: correctness passed, score 0.4658190080346749, rejected by acceptance gate
current OJ best: unchanged
```

## Contest Profile

```text
num_heads = 32
num_heads_k = 4 or 8
headdim = 128
seqlen_q = 1
page_block_size = 16
num_blocks = batch_size * ceil(seqlen_k / page_block_size)
cache_seqlens varies per batch row
reference uses num_splits=0
```

## Decision

This candidate is format-compatible and correctness-compatible, but it is not yet a good final leaderboard candidate because long-KV and large-batch OJ cases are slow.

Use this directory as the OJ track output location. Future OJ-targeted agent runs should update only `submission-oj/`, not `submission/`.

Next OJ optimization should use the `iter_001_oj_agent_retry/next_notes.txt` guidance and avoid repeating the compile-failing generated code from round 000.

## Evidence

```text
docs/contest_validation.md
experiments/contest_validation/current_submission_full/metrics.json
```
