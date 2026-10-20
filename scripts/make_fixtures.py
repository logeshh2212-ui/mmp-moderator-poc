"""Generate every test file for the Step 11 matrix into tests/fixtures/.

    uv run python -m scripts.make_fixtures          # build missing files
    uv run python -m scripts.make_fixtures --force  # rebuild everything

All files are generated with Pillow and ffmpeg; nothing is downloaded (rule 10).
The only hand-made file is tests/fixtures/iphone.heic: copy one real iPhone photo there.
"""
import argparse
import shutil
import struct
import subprocess
import sys
import zlib
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def scene(size=(320, 240)) -> Image.Image:
    """A detailed picture, so a cut-off copy really is missing pixel data."""
    img = Image.effect_noise(size, 40).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.ellipse((20, 20, size[0] // 2, size[1] // 2), fill="orange")
    draw.rectangle((size[0] // 2, size[1] // 3, size[0] - 20, size[1] - 20), fill="navy")
    return img


def save(fmt, **kwargs):
    return lambda path: scene().save(path, fmt, **kwargs)


def animated(fmt):
    def make(path):
        frames = [Image.new("RGB", (64, 64), color) for color in ("red", "green", "blue")]
        frames[0].save(path, fmt, save_all=True, append_images=frames[1:], duration=200, loop=0)
    return make


def exif_rotated(path):
    exif = Image.Exif()
    exif[0x0112] = 6
    scene().save(path, "JPEG", exif=exif)


def tiny(path):
    Image.new("RGB", (16, 16), "orange").save(path, "PNG")


def truncated(path):
    buffer = BytesIO()
    scene((512, 512)).save(buffer, "JPEG")
    data = buffer.getvalue()
    path.write_bytes(data[: len(data) // 2])


def garbage(path):
    path.write_bytes(b"\xff\xd8\xff" + b"not really a jpeg" * 50)


def bomb(path):
    """A tiny PNG that claims to be 50,000 x 50,000 pixels."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    ihdr = struct.pack(">IIBBBBB", 50_000, 50_000, 8, 0, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b""))


def fake_exe(path):
    """A real Windows program (this Python), renamed to .jpg."""
    shutil.copyfile(sys.executable, path)


def empty(path):
    path.write_bytes(b"")


def video_src(seconds):
    return ["-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=160x120:rate=10"]


def audio_src(seconds):
    return ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]


H264 = ["-pix_fmt", "yuv420p", "-c:v", "libx264"]
AAC = ["-c:a", "aac", "-shortest"]


def ffmpeg(*args):
    def make(path):
        subprocess.run(
            [shutil.which("ffmpeg"), "-y", "-v", "error", *args, str(path)],
            check=True,
            timeout=120,
        )
    return make


def corrupt_video(path):
    ffmpeg(*video_src(3), *H264)(path)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])  # the index (moov) is at the end, so it is lost


GENERATORS = {
    # images that must pass
    "ok.jpg": save("JPEG", quality=90),
    "ok.png": save("PNG"),
    "ok.webp": save("WEBP"),
    "animated.webp": animated("WEBP"),
    "animated.png": animated("PNG"),
    "exif_rotated.jpg": exif_rotated,
    # images that must be rejected
    "tiny.png": tiny,
    "truncated.jpg": truncated,
    "garbage.jpg": garbage,
    "bomb.png": bomb,
    "fake_exe.jpg": fake_exe,
    "empty.jpg": empty,
    # videos that must pass
    "ok.mp4": ffmpeg(*video_src(3), *audio_src(3), *H264, *AAC),
    "ok.mov": ffmpeg(*video_src(3), *audio_src(3), *H264, *AAC),
    "ok.avi": ffmpeg(*video_src(3), "-c:v", "mpeg4"),
    "no_audio.mp4": ffmpeg(*video_src(3), *H264),
    "exactly_60s.mp4": ffmpeg(*video_src(60), *H264),
    # videos that must be rejected
    "long_61s.mp4": ffmpeg(*video_src(61), *H264),
    "audio_only.mp4": ffmpeg(*audio_src(3), "-c:a", "aac"),
    "corrupt.mp4": corrupt_video,
}


def build(folder: Path = FIXTURES, force: bool = False) -> dict[str, Path]:
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        raise RuntimeError("ffmpeg/ffprobe must be on PATH")
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, make in GENERATORS.items():
        path = folder / name
        if force or not path.exists():
            make(path)
        paths[name] = path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="rebuild every file")
    args = parser.parse_args()

    for name, path in build(force=args.force).items():
        print(f"{path.stat().st_size:>10,} bytes  {name}")
    heic = FIXTURES / "iphone.heic"
    print(("found" if heic.exists() else "MISSING (copy one by hand)") + f": {heic}")


if __name__ == "__main__":
    main()
