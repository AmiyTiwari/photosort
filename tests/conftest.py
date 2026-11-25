"""Tests never depend on downloaded models: content/places engines are off unless a test injects fakes."""
import pytest

from photosort import library


class Off:
    def available(self):
        return False


@pytest.fixture(autouse=True)
def no_heavy_models(monkeypatch):
    monkeypatch.setattr(library, "ClipEngine", Off)
    monkeypatch.setattr(library, "PlaceIndex", Off)
