"""CLIP image/text embeddings (open_clip ViT-B/32), run locally.

Used for: content groups ("Beach", "Documents"), plain-language search, learning from corrections,
and catching near-duplicates that perceptual hashes miss. Optional: without torch/open_clip or the
weights, `ClipEngine.available()` is False and the rest of the app works as before.
"""
import logging
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from .faces import MODELS_DIR

log = logging.getLogger("photosort.clip")

MODEL = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
CLIP_DIR = MODELS_DIR / "clip"
DIM = 512


def download_clip(cache_dir: Path = CLIP_DIR) -> None:
    import open_clip

    open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED, cache_dir=str(cache_dir))


class ClipEngine:
    def __init__(self, cache_dir: Path = CLIP_DIR):
        self.cache_dir = cache_dir
        self._lock = threading.Lock()
        self._model = None

    def available(self) -> bool:
        try:
            import open_clip  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            return False
        return any(self.cache_dir.rglob("open_clip_*")) if self.cache_dir.exists() else False

    def warm(self) -> None:
        """Load the model now (e.g. in a background thread) so the first query is fast."""
        with self._lock:
            self._load()

    def _load(self):
        if self._model is None:
            import open_clip
            import torch

            self.torch = torch
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            model, _, preprocess = open_clip.create_model_and_transforms(
                MODEL, pretrained=PRETRAINED, cache_dir=str(self.cache_dir))
            self._model = model.eval().to(self.device)
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(MODEL)
            log.info("CLIP loaded on %s", self.device)
        return self._model

    def embed_images(self, paths: list[Path], batch: int = 64) -> np.ndarray:
        """L2-normalised float32 [n, 512]. Unreadable images get a zero vector."""
        out = np.zeros((len(paths), DIM), np.float32)
        with self._lock:
            model = self._load()
            for start in range(0, len(paths), batch):
                tensors, idx = [], []
                for i, p in enumerate(paths[start:start + batch], start):
                    try:
                        with Image.open(p) as img:
                            tensors.append(self._preprocess(img.convert("RGB")))
                        idx.append(i)
                    except Exception as e:
                        log.warning("clip: cannot read %s: %s", p, e)
                if not tensors:
                    continue
                with self.torch.no_grad():
                    feats = model.encode_image(self.torch.stack(tensors).to(self.device)).float().cpu().numpy()
                out[idx] = feats / np.linalg.norm(feats, axis=1, keepdims=True)
        return out

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            model = self._load()
            with self.torch.no_grad():
                feats = model.encode_text(self._tokenizer(texts).to(self.device)).float().cpu().numpy()
        return feats / np.linalg.norm(feats, axis=1, keepdims=True)
