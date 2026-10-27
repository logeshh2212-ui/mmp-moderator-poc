"""Download model files into HF_HOME (D:\\hf-cache). The server itself never downloads.

    uv run python -m scripts.download_models          # every model in settings.ml.models
    uv run python -m scripts.download_models --smoke  # tiny check: one small file
    python -m scripts.download_models --models siglip2_base owlv2 bioclip2   # only these (e.g. Colab)
"""
import argparse
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "0"  # this script is the only place allowed to go online

from app.core.config import MODEL_IGNORE_PATTERNS, settings  # noqa: E402  loads .env (HF_HOME) first
from huggingface_hub import HfApi, constants, hf_hub_download, snapshot_download  # noqa: E402

SMOKE_REPO = "google/siglip2-base-patch16-384"


def check_cache_on_d() -> None:
    cache = constants.HF_HUB_CACHE
    print(f"model cache: {cache}")
    # Rule 9 (keep C: free) is about the Windows laptop; Linux machines such as Colab are fine.
    if os.name == "nt" and not cache.upper().startswith("D:"):
        sys.exit("HF_HOME is not on D:. Check .env (HF_HOME=D:\\hf-cache).")


def smoke() -> None:
    path = hf_hub_download(SMOKE_REPO, "config.json")
    sha = HfApi().model_info(SMOKE_REPO).sha
    print(f"downloaded {path}")
    print(f"current revision of {SMOKE_REPO}: {sha}")


def download_all(only: list[str] | None = None) -> None:
    if not settings.ml.models:
        print("settings.ml.models is empty; nothing to download yet (Step 16 adds SigLIP 2).")
        return
    unknown = set(only or []) - set(settings.ml.models)
    if unknown:
        sys.exit(f"not in settings.ml.models: {sorted(unknown)}")
    for name, model in settings.ml.models.items():
        if only and name not in only:
            continue
        print(f"{name}: {model.repo} @ {model.revision} ({model.licence})")
        # Only the .safetensors weights (see MODEL_IGNORE_PATTERNS)
        folder = snapshot_download(model.repo, revision=model.revision, ignore_patterns=MODEL_IGNORE_PATTERNS)
        size = sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(folder) for f in files)
        print(f"  -> {folder} ({size / 1e9:.2f} GB)")


def download_ocr() -> None:
    """EasyOCR detection + recognition weights for gate2.text.ocr.languages into its model_dir."""
    import easyocr

    cfg = settings.gate2.text.ocr
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    easyocr.Reader(cfg.languages, gpu=False, model_storage_directory=str(cfg.model_dir),
                   user_network_directory=str(cfg.model_dir / "user_network"), download_enabled=True,
                   verbose=False)  # its progress bar crashes the Windows console (cp1252)
    size = sum(f.stat().st_size for f in cfg.model_dir.rglob("*") if f.is_file())
    print(f"ocr ({'+'.join(cfg.languages)}) -> {cfg.model_dir} ({size / 1e6:.0f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="download one small file to test the path")
    parser.add_argument("--models", nargs="+", help="only these names from settings.ml.models")
    parser.add_argument("--ocr", action="store_true", help="only the EasyOCR weights")
    args = parser.parse_args()

    check_cache_on_d()
    if args.smoke:
        smoke()
    elif args.ocr:
        download_ocr()
    else:
        download_all(args.models)


if __name__ == "__main__":
    main()
