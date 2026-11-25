"""Offline reverse geocoding: GPS -> City / Region / Country using GeoNames (CC-BY 4.0).

`download_places()` fetches cities with population >= 15,000 plus region and country names and
stores a compact index (~1.5 MB) in models/places/. Lookups are a nearest-neighbour search on
unit vectors, all in numpy.
"""
import csv
import gzip
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .faces import MODELS_DIR

PLACES_DIR = MODELS_DIR / "places"
GEONAMES = "https://download.geonames.org/export/dump/"
CITY_KM = 30        # within this distance the city name is used
REGION_KM = 250     # beyond this only the country is trusted ... beyond 600 km nothing


def _fetch(name: str) -> bytes:
    with urllib.request.urlopen(GEONAMES + name, timeout=60) as r:
        return r.read()


def download_places(dest: Path = PLACES_DIR) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(_fetch("cities15000.zip"))) as z:
        cities_txt = z.read("cities15000.txt").decode("utf-8")
    admin1 = {}
    for line in _fetch("admin1CodesASCII.txt").decode("utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            admin1[parts[0]] = parts[1]
    countries = {}
    for line in _fetch("countryInfo.txt").decode("utf-8").splitlines():
        if line and not line.startswith("#"):
            parts = line.split("\t")
            countries[parts[0]] = parts[4]

    rows = []
    for line in cities_txt.splitlines():
        p = line.split("\t")
        if len(p) < 15:
            continue
        rows.append((p[0], p[1], p[4], p[5], p[8], p[10], p[14]))
    if len(rows) < 10000:
        raise RuntimeError(f"GeoNames download looks wrong ({len(rows)} cities)")
    with gzip.open(dest / "cities.tsv.gz", "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerows(rows)
    (dest / "names.json").write_text(json.dumps({"admin1": admin1, "countries": countries}))


def _unit(lat, lon) -> np.ndarray:
    lat, lon = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


class PlaceIndex:
    def __init__(self, dir_: Path = PLACES_DIR):
        self.dir = dir_
        self._loaded = False

    def available(self) -> bool:
        return (self.dir / "cities.tsv.gz").exists() and (self.dir / "names.json").exists()

    def _load(self):
        if self._loaded:
            return
        ids, names, lats, lons, ccs, adm, pops = [], [], [], [], [], [], []
        with gzip.open(self.dir / "cities.tsv.gz", "rt", encoding="utf-8") as f:
            for gid, name, lat, lon, cc, a1, pop in csv.reader(f, delimiter="\t"):
                pops.append(float(pop or 0))
                ids.append(gid)
                names.append(name)
                lats.append(float(lat))
                lons.append(float(lon))
                ccs.append(cc)
                adm.append(a1)
        meta = json.loads((self.dir / "names.json").read_text())
        self.ids, self.names, self.ccs, self.adm = ids, names, ccs, adm
        self.admin1, self.countries = meta["admin1"], meta["countries"]
        self.vecs = _unit(np.array(lats), np.array(lons))
        self.pops = np.array(pops)
        self._loaded = True

    def lookup(self, lat: float, lon: float) -> dict | None:
        """{"country": (key, label), "region": (key, label)|None, "city": (key, label)|None}."""
        self._load()
        with np.errstate(all="ignore"):  # numpy+Accelerate on macOS emits spurious matmul warnings
            kms = 6371 * np.arccos(np.clip(self.vecs @ _unit(lat, lon), -1.0, 1.0))
        i = int(np.argmin(kms))
        km = float(kms[i])
        if km > 600:
            return None
        # Prefer the well-known city over a suburb/district that happens to be a bit closer.
        near = np.nonzero(kms <= CITY_KM)[0]
        if len(near):
            i = int(near[np.argmax(self.pops[near] / (1 + (kms[near] / 5) ** 2))])
            km = float(kms[i])
        cc, a1 = self.ccs[i], self.adm[i]
        out = {"country": (cc, self.countries.get(cc, cc)), "region": None, "city": None}
        if km <= REGION_KM and a1 and f"{cc}.{a1}" in self.admin1:
            out["region"] = (f"{cc}.{a1}", self.admin1[f"{cc}.{a1}"])
            if km <= CITY_KM:
                out["city"] = (f"{cc}.{a1}.{self.ids[i]}", self.names[i])
        return out
