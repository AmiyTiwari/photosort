#!/usr/bin/env python3
"""Build the test photo set used for manual testing and the KT demo.

Downloads 70 freely available images from the web (not committed to the repo) and arranges them
as a phone backup: EXIF capture dates and GPS for four trips, two camera models, WhatsApp-style
copies, an exact duplicate, a burst pair, screenshots and an undated download. Result: 74 images
in Camera/, WhatsApp/, Screenshots/ and Downloads/ under --out.

    .venv/bin/python scripts/make_testdata.py                # -> testdata/web_photos
    .venv/bin/python scripts/make_testdata.py --cache ~/dl   # keep the raw downloads around

Sources: ageitgey/face_recognition examples (fr_*), OpenCV samples/data 4.10.0 (cv_*),
Lorem Picsum / Unsplash (px_*). Then: python -m photosort --library library_web serve --port 8766
and import the folder.
"""
import argparse
import shutil
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps
from PIL.ExifTags import IFD

FR = "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples"
CV = "https://raw.githubusercontent.com/opencv/opencv/4.10.0/samples/data"
PICSUM = [237, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 139, 140, 141, 142,
          143, 144, 145, 146, 147, 149, 151, 152, 153, 154]
URLS = {
    **{f"fr_{f}": f"{FR}/{f}" for f in [
        "obama.jpg", "obama2.jpg", "obama_small.jpg", "obama-240p.jpg", "obama-480p.jpg", "obama-720p.jpg",
        "obama-1080p.jpg", "biden.jpg", "alex-lacamoire.png", "lin-manuel-miranda.png", "two_people.jpg"]},
    "fr_biden2.jpg": f"{FR}/knn_examples/train/biden/biden2.jpg",
    "fr_alex_img1.jpg": f"{FR}/knn_examples/train/alex_lacamoire/img1.jpg",
    **{f"fr_{f}": f"{FR}/knn_examples/test/{f}" for f in [
        "alex_lacamoire1.jpg", "johnsnow_test1.jpg", "kit_with_rose.jpg", "obama1.jpg", "obama_and_biden.jpg"]},
    **{f"cv_{f}": f"{CV}/{f}" for f in [
        "fruits.jpg", "building.jpg", "butterfly.jpg", "starry_night.jpg", "sudoku.png", "text_defocus.jpg",
        "notes.png", "smarties.png", "HappyFish.jpg", "home.jpg", "aero1.jpg", "apple.jpg", "orange.jpg",
        "baboon.jpg", "squirrel_cls.jpg", "licenseplate_motion.jpg", "chicky_512.png", "messi5.jpg", "board.jpg",
        "stuff.jpg", "cards.png", "pic1.png"]},
    **{f"px_{i}.jpg": f"https://picsum.photos/id/{i}/1200/800" for i in PICSUM},
}

DC, NYC, GOA, MANALI = (38.8977, -77.0365), (40.7590, -73.9845), (15.5494, 73.7535), (32.2432, 77.1892)
IPHONE, PIXEL = ("Apple", "iPhone 15"), ("Google", "Pixel 8")


def download(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    todo = {n: u for n, u in URLS.items() if not (cache / n).exists()}
    for i, (name, url) in enumerate(todo.items(), 1):
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "PhotoSort-testdata/1.0"})
                with urllib.request.urlopen(req, timeout=120) as r, open(cache / f"{name}.part", "wb") as f:
                    shutil.copyfileobj(r, f)
                with Image.open(cache / f"{name}.part") as im:
                    im.verify()
                (cache / f"{name}.part").rename(cache / name)
                print(f"[{i}/{len(todo)}] {name}", flush=True)
                break
            except Exception as e:  # noqa: BLE001 - retry anything, then give up on this file
                (cache / f"{name}.part").unlink(missing_ok=True)
                if attempt == 2:
                    print(f"FAILED {name}: {e}", file=sys.stderr)
                time.sleep(2)
    missing = [n for n in URLS if not (cache / n).exists()]
    if missing:
        sys.exit(f"{len(missing)} downloads failed: {missing}")


def dms(x: float):
    x = abs(x)
    d = int(x)
    m = int((x - d) * 60)
    return float(d), float(m), round(((x - d) * 60 - m) * 60, 2)


class Builder:
    def __init__(self, cache: Path, out: Path):
        self.cache, self.out = cache, out
        self.cam, self.wa = out / "Camera", out / "WhatsApp"
        for d in (self.cam, self.wa, out / "Screenshots", out / "Downloads"):
            d.mkdir(parents=True)

    def raw(self, name: str) -> Image.Image:
        return ImageOps.exif_transpose(Image.open(self.cache / name)).convert("RGB")

    def shoot(self, src, when, gps=None, camera=IPHONE, img=None) -> Path:
        img = img or self.raw(src)
        exif = Image.Exif()
        exif[0x0132] = when.strftime("%Y:%m:%d %H:%M:%S")
        exif[0x010F], exif[0x0110] = camera
        exif.get_ifd(IFD.Exif)[0x9003] = when.strftime("%Y:%m:%d %H:%M:%S")
        if gps:
            lat, lon = gps
            exif[IFD.GPSInfo] = {1: "N" if lat >= 0 else "S", 2: dms(lat), 3: "E" if lon >= 0 else "W", 4: dms(lon)}
        dest = self.cam / f"IMG_{when:%Y%m%d_%H%M%S}.jpg"
        img.save(dest, "JPEG", quality=92, exif=exif)
        return dest

    def series(self, files, start, gps, camera=IPHONE, step=timedelta(minutes=17)) -> list[Path]:
        return [self.shoot(f, start + i * step, gps, camera) for i, f in enumerate(files)]

    def build(self) -> int:
        # Trip 1: Washington, 15 Mar 2025. The four-resolution portrait + a different crop = near-dup test.
        t = datetime(2025, 3, 15, 10, 0)
        self.series(["fr_obama1.jpg", "fr_obama2.jpg", "fr_biden.jpg", "fr_biden2.jpg", "fr_obama_and_biden.jpg",
                     "fr_two_people.jpg", "fr_obama_small.jpg"], t, DC)
        for i, f in enumerate(["fr_obama-1080p.jpg", "fr_obama-720p.jpg", "fr_obama-480p.jpg", "fr_obama-240p.jpg"]):
            self.shoot(f, t + timedelta(hours=3, seconds=2 * i), DC)
        self.shoot("fr_obama.jpg", t + timedelta(hours=3, seconds=9), DC)
        # Trip 2: New York, 5 Jun 2025.
        self.series(["fr_alex-lacamoire.png", "fr_alex_img1.jpg", "fr_lin-manuel-miranda.png", "fr_kit_with_rose.jpg",
                     "fr_johnsnow_test1.jpg", "fr_alex_lacamoire1.jpg"], datetime(2025, 6, 5, 19, 0), NYC)
        # Trip 3: Goa, 20-21 Dec 2025, with a burst pair and a re-crop.
        goa_a = self.series(["px_124.jpg", "px_147.jpg", "px_131.jpg", "px_129.jpg", "px_123.jpg"], datetime(2025, 12, 20, 9, 30), GOA)
        goa_b = self.series(["px_154.jpg", "px_135.jpg", "px_144.jpg", "px_132.jpg", "px_149.jpg"], datetime(2025, 12, 21, 16, 0), GOA)
        sea = self.raw("px_147.jpg")
        self.shoot(None, datetime(2025, 12, 20, 9, 47, 3), GOA, img=ImageEnhance.Brightness(sea.crop((30, 20, 1180, 790))).enhance(1.08))
        self.shoot(None, datetime(2025, 12, 20, 9, 30, 25), GOA, img=self.raw("px_124.jpg").crop((200, 120, 1100, 760)))
        # Trip 4: Manali, 10-11 Jan 2026.
        self.series(["px_128.jpg", "px_127.jpg", "px_136.jpg", "px_140.jpg", "px_141.jpg", "px_125.jpg"], datetime(2026, 1, 10, 8, 0), MANALI)
        self.series(["px_152.jpg", "px_126.jpg", "px_139.jpg", "px_143.jpg", "px_142.jpg"], datetime(2026, 1, 11, 11, 0), MANALI)
        # Everyday Bengaluru, Feb 2026: another phone, no GPS, 2-3 photos a day (too few for events).
        days = [["cv_apple.jpg", "cv_orange.jpg", "cv_fruits.jpg"], ["cv_chicky_512.png", "px_237.jpg"],
                ["cv_squirrel_cls.jpg", "cv_baboon.jpg", "cv_butterfly.jpg"],
                ["cv_sudoku.png", "cv_text_defocus.jpg", "cv_licenseplate_motion.jpg"],
                ["cv_building.jpg", "cv_home.jpg", "px_134.jpg"], ["px_133.jpg", "px_146.jpg", "px_153.jpg"],
                ["cv_starry_night.jpg", "cv_messi5.jpg"], ["cv_aero1.jpg", "cv_board.jpg", "cv_stuff.jpg"],
                ["cv_smarties.png", "px_145.jpg", "cv_HappyFish.jpg"], ["cv_cards.png", "px_151.jpg"]]
        for d, files in enumerate(days):
            self.series(files, datetime(2026, 2, 1 + d, 12, 0), None, PIXEL, step=timedelta(hours=2))
        # WhatsApp: no EXIF, date only in the name; one exact copy, one resized copy, one unrelated.
        shutil.copy2(goa_a[0], self.wa / "IMG-20251220-WA0001.jpg")
        Image.open(goa_b[0]).convert("RGB").resize((600, 400)).save(self.wa / "IMG-20251221-WA0002.jpg", quality=60)
        self.raw("px_137.jpg").save(self.wa / "IMG-20260214-WA0003.jpg", quality=80)
        # Screenshots (PNG, date in the name) and a download with no date anywhere.
        self.raw("cv_pic1.png").save(self.out / "Screenshots" / "Screenshot 2026-02-10 at 9.14.02 AM.png")
        self.raw("cv_notes.png").save(self.out / "Screenshots" / "Screenshot 2026-02-12 at 6.40.11 PM.png")
        shutil.copy2(self.cache / "px_130.jpg", self.out / "Downloads" / "wallpaper.jpg")
        (self.out / "SOURCES.md").write_text(
            "Test images from the web: ageitgey/face_recognition examples (fr_*), OpenCV samples/data (cv_*), "
            "Lorem Picsum / Unsplash (px_*). Dates, GPS and camera tags were added to simulate a phone backup. "
            "Generated by scripts/make_testdata.py; not committed to the repository.\n")
        return sum(1 for p in self.out.rglob("*") if p.is_file())


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(root / "testdata" / "web_photos"))
    ap.add_argument("--cache", help="directory for the raw downloads (default: a temporary one)")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; remove it first")
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(args.cache) if args.cache else Path(tmp)
        download(cache)
        n = Builder(cache, out).build()
    print(f"{n} files in {out}")


if __name__ == "__main__":
    main()
