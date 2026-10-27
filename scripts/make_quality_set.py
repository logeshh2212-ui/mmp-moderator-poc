"""Make the D5 quality tuning set: known-bad copies of good photos (spec 4.5 table).

    uv run python -m scripts.make_quality_set D:/samples/gate1 D:/samples/quality

Every image under the source folder with a short side of at least --min-side is resized to
a long side of 1280 and saved once per kind into <out>/<kind>/. Never point this at the
hard-case test set.
"""
import argparse
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
BASE_LONG_SIDE = 1280


def letterbox(img: Image.Image) -> Image.Image:
    """A landscape video shown on a portrait phone screen, then screenshotted."""
    screen = Image.new("RGB", (720, 1280), "black")
    shown = ImageOps.contain(img, (720, 1280))
    screen.paste(shown, ((720 - shown.width) // 2, (1280 - shown.height) // 2))
    return screen


KINDS = {
    "good": lambda im: (im, 92),
    "slight_blur": lambda im: (im.filter(ImageFilter.GaussianBlur(2)), 92),
    "heavy_blur": lambda im: (im.filter(ImageFilter.GaussianBlur(8)), 92),
    "dark": lambda im: (ImageEnhance.Brightness(im).enhance(0.2), 92),
    "overexposed": lambda im: (ImageEnhance.Brightness(im).enhance(3.0), 92),
    "compressed": lambda im: (im, 5),
    "lowres": lambda im: (ImageOps.contain(im, (360, 360)), 92),
    "screenshot": lambda im: (letterbox(im), 92),
}
BLANKS = {"black": (0, 0, 0), "white": (255, 255, 255), "grey": (128, 128, 128)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--min-side", type=int, default=480)
    args = parser.parse_args()

    sources = [
        p for p in sorted(args.source.rglob("*"))
        if p.suffix.lower() in IMAGE_EXT and min(Image.open(p).size) >= args.min_side
    ]
    for kind in [*KINDS, "blank"]:
        (args.out / kind).mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(sources):
        base = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        base = ImageOps.contain(base, (BASE_LONG_SIDE, BASE_LONG_SIDE))
        for kind, make in KINDS.items():
            picture, quality = make(base)
            picture.save(args.out / kind / f"{i:03d}_{path.stem[:40]}.jpg", quality=quality)
    for name, colour in BLANKS.items():
        Image.new("RGB", (1280, 960), colour).save(args.out / "blank" / f"{name}.jpg", quality=92)

    print(f"{len(sources)} sources x {len(KINDS)} kinds + {len(BLANKS)} blanks -> {args.out}")


if __name__ == "__main__":
    main()
