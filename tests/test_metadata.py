from PIL import Image, PngImagePlugin

from app.validation.metadata import MediaMetadata, read_image_metadata, read_video_metadata
from app.validation.video_check import check_video


def save_jpeg(tmp_path, exif=None, xmp=None, name="img.jpg"):
    path = tmp_path / name
    kwargs = {}
    if exif is not None:
        kwargs["exif"] = exif
    if xmp is not None:
        kwargs["xmp"] = xmp
    Image.new("RGB", (64, 64), "orange").save(path, "JPEG", **kwargs)
    return path


def test_reads_camera_fields(tmp_path):
    exif = Image.Exif()
    exif[0x010F] = "Apple"
    exif[0x0110] = "iPhone 15 Pro"
    exif[0x0131] = "17.4"
    exif[0x0132] = "2026:10:01 09:30:00"

    meta = read_image_metadata(save_jpeg(tmp_path, exif=exif))

    assert meta.camera_make == "Apple"
    assert meta.camera_model == "iPhone 15 Pro"
    assert meta.software == "17.4"
    assert meta.taken_at == "2026:10:01 09:30:00"
    assert meta.ai_hints == []


def test_gps_is_detected_but_not_stored(tmp_path):
    exif = Image.Exif()
    exif[0x8825] = {1: "N", 2: (12.0, 58.0, 7.0)}

    meta = read_image_metadata(save_jpeg(tmp_path, exif=exif))

    assert meta.has_gps is True
    assert "58.0" not in repr(meta)


def test_generator_software_gives_hint(tmp_path):
    exif = Image.Exif()
    exif[0x0131] = "Midjourney v6"

    meta = read_image_metadata(save_jpeg(tmp_path, exif=exif))

    assert "generator:midjourney" in meta.ai_hints


def test_xmp_trained_algorithmic_media_gives_hint(tmp_path):
    xmp = (
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
        b'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description '
        b'xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/" '
        b'Iptc4xmpExt:DigitalSourceType='
        b'"http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"/>'
        b"</rdf:RDF></x:xmpmeta>"
    )

    meta = read_image_metadata(save_jpeg(tmp_path, xmp=xmp))

    assert meta.ai_hints == ["xmp:trained_algorithmic_media"]


def test_png_parameters_chunk_gives_hint(tmp_path):
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "a cat on a sofa, Steps: 20, Sampler: Euler")
    path = tmp_path / "sd.png"
    Image.new("RGB", (64, 64)).save(path, "PNG", pnginfo=info)

    meta = read_image_metadata(path)

    assert meta.text_keys == ["parameters"]
    assert "png_text:parameters" in meta.ai_hints


def test_plain_image_has_empty_metadata(tmp_path):
    assert read_image_metadata(save_jpeg(tmp_path)) == MediaMetadata()


def test_unreadable_file_never_raises(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"not an image at all")

    assert read_image_metadata(path) == MediaMetadata()


def test_iphone_video_tags():
    meta = read_video_metadata({
        "com.apple.quicktime.make": "Apple",
        "com.apple.quicktime.model": "iPhone 15",
        "com.apple.quicktime.software": "17.4",
        "com.apple.quicktime.creationdate": "2026-10-01T09:30:00+0530",
        "com.apple.quicktime.location.ISO6709": "+12.9716+077.5946+000.000/",
    })

    assert meta.camera_make == "Apple"
    assert meta.camera_model == "iPhone 15"
    assert meta.software == "17.4"
    assert meta.taken_at.startswith("2026-10-01")
    assert meta.has_gps is True
    assert "12.9716" not in repr(meta)


def test_generated_video_has_encoder_tag(videos):
    meta = read_video_metadata(check_video(videos["mp4"], "mp4").tags)

    assert meta.software.startswith("Lavf")
    assert meta.ai_hints == []
