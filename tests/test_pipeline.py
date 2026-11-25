import os
import random
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from PIL.ExifTags import IFD

from photosort.api import create_app
from photosort.library import Library


def scene(seed: int, size=(800, 600), shift=0, bright=0) -> Image.Image:
    rnd = random.Random(seed)
    img = Image.new("RGB", size, (rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)))
    d = ImageDraw.Draw(img)
    for _ in range(25):
        x, y = rnd.randrange(size[0]) + shift, rnd.randrange(size[1])
        r = rnd.randrange(20, 140)
        c = tuple(min(255, rnd.randrange(256) + bright) for _ in range(3))
        (d.ellipse if rnd.random() < 0.5 else d.rectangle)([x, y, x + r, y + r], fill=c)
    return img


def save(img, path: Path, date=None, gps=None, camera=None, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    if date:
        exif[0x0132] = date  # DateTime
    if camera:
        exif[0x010F], exif[0x0110] = camera
    if gps:
        exif[IFD.GPSInfo] = gps
    img.save(path, exif=exif, **kw)


GOA = {1: "N", 2: (15.0, 33.0, 0.0), 3: "E", 4: (73.0, 45.0, 0.0)}


@pytest.fixture
def photos(tmp_path):
    src = tmp_path / "phone"
    a = scene(1)
    save(a, src / "IMG_001.jpg", "2025:12:20 10:00:00", GOA, ("Apple", "iPhone 15"), quality=95)
    (src / "backup").mkdir()
    shutil.copy2(src / "IMG_001.jpg", src / "backup" / "IMG_001 copy.jpg")
    save(a.resize((400, 300)), src / "whatsapp" / "IMG-WA0001.jpg", "2025:12:20 10:00:00", quality=60)
    save(scene(2), src / "burst" / "B1.jpg", "2025:12:21 18:30:00", quality=92)
    save(scene(2, shift=22, bright=15), src / "burst" / "B2.jpg", "2025:12:21 18:30:03", quality=92)
    save(scene(3), src / "Screenshot 2026-01-02 at 9.00.00.png")
    (src / "broken.jpg").write_bytes(b"not an image")
    return src


@pytest.fixture
def lib(tmp_path):
    return Library(tmp_path / "library", workers=4)


def groups(lib):
    c = lib.catalog.connect()
    out = {}
    for g in c.execute("SELECT * FROM similar_groups"):
        names = sorted(
            Path(r["original_path"]).name
            for r in c.execute(
                "SELECT s.original_path FROM photo_similar ps JOIN sources s ON s.sha256 = ps.sha256 "
                "WHERE ps.group_id=?", (g["id"],))
        )
        out[g["kind"]] = (names, g["best_sha256"])
    return out


def sha_of(lib, name):
    return lib.catalog.connect().execute(
        "SELECT sha256 FROM sources WHERE original_path LIKE ?", (f"%/{name}",)).fetchone()[0]


def test_import_dedups_and_groups(lib, photos):
    b = lib.import_sync(str(photos))
    assert b["status"] == "done"
    assert (b["total"], b["new_photos"], b["exact_dupes"], b["errors"]) == (7, 5, 1, 1)

    c = lib.catalog.connect()
    assert c.execute("SELECT COUNT(*) FROM photos").fetchone()[0] == 5
    assert len(list((lib.root / "store").rglob("*.*"))) == 5  # one stored copy per unique photo
    assert c.execute("SELECT COUNT(*) FROM sources WHERE sha256=?", (sha_of(lib, "IMG_001.jpg"),)).fetchone()[0] == 2

    g = groups(lib)
    assert g["variant"][0] == ["IMG-WA0001.jpg", "IMG_001 copy.jpg", "IMG_001.jpg"]
    assert g["variant"][1] == sha_of(lib, "IMG_001.jpg")  # full-res original wins
    assert g["burst"][0] == ["B1.jpg", "B2.jpg"]


def test_factors(lib, photos):
    lib.import_sync(str(photos))
    c = lib.catalog.connect()
    labels = {r[0] for r in c.execute(
        "SELECT v.label FROM photo_factor pf JOIN factor_values v ON v.id = pf.value_id WHERE pf.sha256=?",
        (sha_of(lib, "IMG_001.jpg"),))}
    assert {"2025", "December 2025", "15.6°N, 73.8°E", "Apple iPhone 15", "Photos"} <= labels
    shot = sha_of(lib, "Screenshot 2026-01-02 at 9.00.00.png")
    assert c.execute("SELECT is_screenshot FROM photos WHERE sha256=?", (shot,)).fetchone()[0] == 1


def test_reimport_is_incremental(lib, photos, tmp_path):
    lib.import_sync(str(photos))
    again = lib.import_sync(str(photos))
    assert (again["skipped"], again["new_photos"], again["exact_dupes"]) == (6, 0, 0)  # broken.jpg is retried

    # New batch from another device: a different re-encode of an existing photo joins the existing group.
    other = tmp_path / "laptop"
    save(scene(1).resize((640, 480)), other / "edited.jpg", "2025:12:20 10:00:00", quality=75)
    save(scene(9), other / "new.jpg", "2026:02:01 12:00:00")
    b = lib.import_sync(str(other))
    assert b["new_photos"] == 2
    assert "edited.jpg" in groups(lib)["variant"][0]
    assert lib.catalog.connect().execute("SELECT COUNT(*) FROM similar_groups").fetchone()[0] == 2


def test_user_choices_and_policy(lib, photos):
    lib.import_sync(str(photos))
    c = lib.catalog.connect()
    gid = c.execute("SELECT id FROM similar_groups WHERE kind='variant'").fetchone()[0]
    small = sha_of(lib, "IMG-WA0001.jpg")

    lib.set_dup_policy("hide_non_best")
    assert c.execute("SELECT COUNT(*) FROM photos WHERE hidden=1").fetchone()[0] == 2  # one per group

    lib.set_group_best(gid, small)
    assert c.execute("SELECT hidden FROM photos WHERE sha256=?", (small,)).fetchone()[0] == 0
    assert c.execute("SELECT best_sha256, user_picked FROM similar_groups WHERE id=?", (gid,)).fetchone()[:] == (small, 1)

    lib.remove_from_group(gid, small)  # group of 2 -> dissolves
    assert c.execute("SELECT COUNT(*) FROM similar_groups WHERE id=?", (gid,)).fetchone()[0] == 0
    assert c.execute("SELECT COUNT(*) FROM photos WHERE hidden=1 AND sha256 != ?", (small,)).fetchone()[0] == 1


def test_export_uses_hardlinks(lib, photos, tmp_path):
    lib.import_sync(str(photos))
    r = lib.export_view("date", str(tmp_path / "views"))
    assert r["hardlinks"] == 5
    files = list(Path(r["path"]).rglob("*.*"))
    assert {f.parent.name for f in files} >= {"December 2025"}
    stored = {os.stat(p).st_ino for p in (lib.root / "store").rglob("*.*")}
    assert {f.stat().st_ino for f in files} <= stored


def test_hardlink_import_mode(lib, photos):
    lib.import_sync(str(photos), mode="hardlink")
    sha = sha_of(lib, "IMG_001.jpg")
    stored = next((lib.root / "store").rglob(f"{sha}.*"))
    assert stored.stat().st_ino == (photos / "IMG_001.jpg").stat().st_ino


def test_api(lib, photos):
    client = TestClient(create_app(lib))
    assert client.post("/api/imports", json={"path": str(lib.root)}).status_code == 400
    bid = client.post("/api/imports", json={"path": str(photos)}).json()["id"]
    with lib._import_lock:
        pass
    assert client.get("/api/imports").json()[0]["id"] == bid
    s = client.get("/api/stats").json()
    assert (s["photos"], s["sources"], s["similar_groups"]) == (5, 6, 2)

    factors = {f["key"]: f for f in client.get("/api/factors").json()}
    month = next(v for v in factors["date"]["values"] if v["label"] == "December 2025")
    page = client.get(f"/api/photos?value_id={month['id']}").json()
    assert page["total"] == 4

    sha = page["items"][0]["sha256"]
    assert client.get(f"/thumbs/{sha}.jpg").status_code == 200
    assert client.get(f"/api/photos/{sha}/file").status_code == 200
    assert client.get(f"/api/photos/{sha}").json()["factors"]
    sim = client.get("/api/similar").json()
    assert sim["total"] == 2 and client.get(f"/api/similar/{sim['groups'][0]['id']}").json()["members"]
    assert client.get("/").status_code == 200
