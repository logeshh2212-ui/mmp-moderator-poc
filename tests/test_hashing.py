from PIL import Image, ImageDraw

from app.validation.hashing import hamming, phash_image, phash_png_bytes
from app.validation.video_check import check_video


def scene(kind: str = "shapes", size=(400, 300)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    if kind == "shapes":
        draw.ellipse((40, 40, 200, 200), fill="black")
        draw.rectangle((250, 100, 380, 280), fill="gray")
    else:
        for x in range(0, size[0], 40):
            draw.rectangle((x, 0, x + 20, size[1]), fill="black")
    return img


def save(tmp_path, img, name, **kwargs):
    path = tmp_path / name
    img.save(path, "JPEG", **kwargs)
    return path


def test_hash_is_16_hex_characters(tmp_path):
    h = phash_image(save(tmp_path, scene(), "a.jpg"))

    assert len(h) == 16
    int(h, 16)  # raises if not hex


def test_resaved_and_resized_copy_is_close(tmp_path):
    original = phash_image(save(tmp_path, scene(), "a.jpg", quality=95))
    copy = phash_image(save(tmp_path, scene().resize((200, 150)), "b.jpg", quality=60))

    assert hamming(original, copy) <= 8


def test_different_pictures_are_far_apart(tmp_path):
    a = phash_image(save(tmp_path, scene("shapes"), "a.jpg"))
    b = phash_image(save(tmp_path, scene("stripes"), "b.jpg"))

    assert hamming(a, b) > 10


def test_exif_rotation_is_undone_before_hashing(tmp_path):
    exif = Image.Exif()
    exif[0x0112] = 6  # stored sideways, shown rotated 90 degrees clockwise
    sideways = save(tmp_path, scene(), "sideways.jpg", exif=exif)
    upright = save(tmp_path, scene().rotate(-90, expand=True), "upright.jpg")

    assert hamming(phash_image(sideways, orientation=6), phash_image(upright)) <= 4


def test_identical_hashes_have_distance_zero():
    assert hamming("c3a1f0e8d2b4a596", "c3a1f0e8d2b4a596") == 0


def test_video_frame_hash_is_stable(videos):
    a = phash_png_bytes(check_video(videos["mp4"], "mp4").first_frame_png)
    b = phash_png_bytes(check_video(videos["mp4"], "mp4").first_frame_png)

    assert a is not None
    assert a == b


def test_unreadable_input_returns_none(tmp_path):
    path = tmp_path / "junk.jpg"
    path.write_bytes(b"junk")

    assert phash_image(path) is None
    assert phash_png_bytes(b"junk") is None
