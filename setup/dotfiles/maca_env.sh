# MACA C500 Environment Variables
# Source this file in your ~/.bashrc or ~/.profile:
#   source /path/to/setup/dotfiles/maca_env.sh

# --- MACA GPU Toolkit ---
export MACA_PATH=/opt/maca
export MACA_CLANG_PATH=/opt/maca/mxgpu_llvm/bin

# --- Library Paths ---
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/opt/mxdriver/lib:$LIBRARY_PATH

# --- PATH ---
export PATH=/opt/conda/bin:/opt/conda/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/maca/mxgpu_llvm/bin:/opt/maca/ompi/bin:/opt/maca/ucx/bin:/opt/mxdriver/bin:$PATH

# --- PyTorch ---
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# --- HuggingFace Mirror (for China) ---
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/huggingface_home
export HF_HUB_CACHE=$HF_HOME/hub

# Ensure HF cache directory exists
mkdir -p /data/huggingface_home/hub

# --- Proxy (uncomment when needed) ---
# export http_proxy=http://127.0.0.1:7897
# export https_proxy=http://127.0.0.1:7897
