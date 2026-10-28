#!/usr/bin/env bash
# One-command speed test on a Google Colab T4 (Runtime -> Change runtime type -> T4 GPU).
# Unzip mmp_t4.zip into /content, then:  !bash /content/mmp-moderator-poc/scripts/colab_t4.sh
# Installs the packages (keeps Colab's CUDA PyTorch), downloads the models, then runs the full
# pipeline on /content/bench with GPU video decoding. Prints the per-file lines and the summary.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"

if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
    echo "NO GPU: this notebook runs on a CPU. Runtime -> Change runtime type -> T4 GPU -> Save,"
    echo "then upload the zip again (changing the runtime wipes /content) and rerun this cell."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv export --no-hashes --no-dev --format requirements-txt | grep -v -E "^(torch|torchvision|torchaudio)" > /tmp/req.txt
uv pip install --system -q -r /tmp/req.txt

python -m scripts.download_models --models siglip2_base owlv2 bioclip2 nsfw_vit ai_smaug ast_audioset whisper_base
python -m scripts.download_models --ocr

ML_DEVICE=cuda ML_DTYPE=float16 FFMPEG_HWACCEL=cuda python -m scripts.bench_pipeline ../bench --runs 2
