"""Per-file analysis: content hash, EXIF, perceptual hashes, quality, thumbnail.

Runs in worker threads; must not touch the catalog.
"""
import hashlib
import os
import re
from datetime import datetime
from pathlib import Path

import imagehash
from PIL import Image, ImageFilter, ImageOps, ImageStat
from PIL.ExifTags import GPSTAGS, IFD, TAGS

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # HEIC support is optional
    pass

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tif", ".tiff", ".bmp", ".gif"}
THUMB_SIZE = 320
SCREENSHOT_NAME = re.compile(r"screen ?shot|screen_?recording|^scr_|simulator screen", re.I)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _exif_dict(img: Image.Image) -> dict:
    exif = img.getexif()
    out = {TAGS.get(k, k): v for k, v in exif.items()}
    try:
        out.update({TAGS.get(k, k): v for k, v in exif.get_ifd(IFD.Exif).items()})
    except KeyError:
        pass
    try:
        gps = exif.get_ifd(IFD.GPSInfo)
        out["GPSInfo"] = {GPSTAGS.get(k, k): v for k, v in gps.items()}
    except KeyError:
        pass
    return out


def _parse_exif_date(value) -> str | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value.strip()[:19], "%Y:%m:%d %H:%M:%S").isoformat()
    except ValueError:
        return None


# e.g. IMG_20250101_120000, PXL_20240101_123456789, 2026-06-28_194557, 2025-12-20 10.00.00
_FN_DATETIME = re.compile(
    r"(?<!\d)((?:19|20)\d{2})[-_.]?(\d{2})[-_.]?(\d{2})[ _T-]?(\d{2})[.:\-_]?(\d{2})[.:\-_]?(\d{2})(?!\d{1,2}\b)"
)
# e.g. "Screenshot 2026-06-12 at 5.50.49 PM"
_FN_AT = re.compile(r"((?:19|20)\d{2})-(\d{2})-(\d{2}) at (\d{1,2})\.(\d{2})\.(\d{2})\s*([AP]M)?", re.I)
_FN_DATE = re.compile(r"(?<!\d)((?:19|20)\d{2})[-_]?(\d{2})[-_]?(\d{2})(?!\d)")


def date_from_filename(name: str) -> str | None:
    """Best-effort capture time from common camera/phone/messaging file names."""
    m = _FN_AT.search(name)
    if m:
        y, mo, d, hh, mm, ss, ampm = m.groups()
        hh = int(hh) % 12 + (12 if ampm and ampm.upper() == "PM" else 0) if ampm else int(hh)
        parts = (y, mo, d, hh, mm, ss)
    else:
        m = _FN_DATETIME.search(name) or _FN_DATE.search(name)
        if not m:
            return None
        parts = (*m.groups(), 0, 0, 0)[:6]
    try:
        dt = datetime(*(int(x) for x in parts))
    except ValueError:
        return None
    return dt.isoformat() if 1990 <= dt.year <= datetime.now().year + 1 else None


def _gps_to_deg(coord, ref) -> float | None:
    try:
        d, m, s = (float(x) for x in coord)
    except (TypeError, ValueError):
        return None
    deg = d + m / 60 + s / 3600
    return -deg if ref in ("S", "W") else deg


def _quality(gray_small: Image.Image) -> float:
    """Sharpness proxy: variance of the edge map (higher = sharper)."""
    edges = gray_small.filter(ImageFilter.FIND_EDGES)
    return round(ImageStat.Stat(edges).var[0], 2)


def analyze_file(path: Path, thumbs_dir: Path) -> dict:
    """Return everything the catalog needs about one file. Raises on unreadable images."""
    st = path.stat()
    sha = sha256_file(path)
    info: dict = {
        "path": str(path),
        "sha256": sha,
        "size": st.st_size,
        "mtime": st.st_mtime,
        "ext": path.suffix.lower().lstrip(".") or "bin",
    }

    with Image.open(path) as img:
        exif = _exif_dict(img)
        img = ImageOps.exif_transpose(img)  # respect orientation for thumbs/hashes
        info["width"], info["height"] = img.size

        taken = _parse_exif_date(exif.get("DateTimeOriginal")) or _parse_exif_date(exif.get("DateTime"))
        from_name = None if taken else date_from_filename(path.name)
        info["taken_at"] = taken or from_name or datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
        info["taken_src"] = "exif" if taken else "filename" if from_name else "mtime"

        gps = exif.get("GPSInfo") or {}
        lat = _gps_to_deg(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"))
        lon = _gps_to_deg(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))
        info["lat"], info["lon"] = (lat, lon) if lat is not None and lon is not None else (None, None)

        make = exif.get("Make")
        model = exif.get("Model")
        info["camera_make"] = make.strip("\x00 ").strip() if isinstance(make, str) else None
        info["camera_model"] = model.strip("\x00 ").strip() if isinstance(model, str) else None

        rgb = img.convert("RGB")
        info["phash"] = str(imagehash.phash(rgb))
        info["dhash"] = str(imagehash.dhash(rgb))

        small = rgb.copy()
        small.thumbnail((512, 512))
        info["quality"] = _quality(small.convert("L"))

        thumb_path = thumbs_dir / f"{sha}.jpg"
        if not thumb_path.exists():
            t = rgb.copy()
            t.thumbnail((THUMB_SIZE, THUMB_SIZE))
            tmp = thumb_path.with_suffix(f".{os.getpid()}.{id(t)}.tmp")
            t.save(tmp, "JPEG", quality=82)
            os.replace(tmp, thumb_path)

    info["is_screenshot"] = int(
        bool(SCREENSHOT_NAME.search(path.name))
        or (info["ext"] == "png" and not info["camera_model"] and info["taken_src"] != "exif")
    )
    return info
