"""Face detection (YuNet) + face embeddings (SFace), both small ONNX models run by OpenCV.

Fully local. If OpenCV or the model files are missing, `FaceEngine.available()` is False and
the rest of the app keeps working without people sorting.
"""
import hashlib
import threading
import urllib.request
from pathlib import Path

from PIL import Image, ImageOps

try:
    import cv2
    import numpy as np
except ImportError:  # optional dependency
    cv2 = None
    np = None

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS = {
    "detector": (
        "face_detection_yunet_2023mar.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "recognizer": (
        "face_recognition_sface_2021dec.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}

MAX_SIDE = 1280        # detect on a downscaled copy
MIN_FACE_PX = 40       # ignore tiny background faces
SCORE_THRESHOLD = 0.85


def download_models(models_dir: Path = MODELS_DIR) -> None:
    """Fetch the two ONNX models and verify their pinned SHA-256."""
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, url, sha in MODELS.values():
        dest = models_dir / name
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == sha:
            continue
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        got = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if got != sha:
            tmp.unlink()
            raise RuntimeError(f"checksum mismatch for {name}: {got}")
        tmp.replace(dest)


class FaceEngine:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.det_path = models_dir / MODELS["detector"][0]
        self.rec_path = models_dir / MODELS["recognizer"][0]
        self._local = threading.local()  # OpenCV DNN nets are not shared across threads

    def available(self) -> bool:
        return cv2 is not None and self.det_path.exists() and self.rec_path.exists()

    def _nets(self):
        if not hasattr(self._local, "det"):
            self._local.det = cv2.FaceDetectorYN.create(str(self.det_path), "", (320, 320), SCORE_THRESHOLD, 0.3, 5000)
            self._local.rec = cv2.FaceRecognizerSF.create(str(self.rec_path), "")
        return self._local.det, self._local.rec

    def detect(self, path: Path) -> list[dict]:
        """Return [{x, y, w, h (relative), score, embedding (float32[128], unit norm), crop (jpeg bytes)}]."""
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((MAX_SIDE, MAX_SIDE))
            bgr = np.ascontiguousarray(np.asarray(img)[:, :, ::-1])
        h_img, w_img = bgr.shape[:2]
        det, rec = self._nets()
        det.setInputSize((w_img, h_img))
        _, found = det.detect(bgr)
        out = []
        for f in found if found is not None else []:
            x, y, w, h = (float(v) for v in f[:4])
            if min(w, h) < MIN_FACE_PX:
                continue
            aligned = rec.alignCrop(bgr, f)
            emb = rec.feature(aligned).flatten().astype(np.float32)
            emb /= np.linalg.norm(emb) or 1.0
            ok, jpg = cv2.imencode(".jpg", aligned, [cv2.IMWRITE_JPEG_QUALITY, 85])
            out.append({
                "x": max(0.0, x / w_img), "y": max(0.0, y / h_img), "w": w / w_img, "h": h / h_img,
                "score": float(f[-1]), "embedding": emb, "crop": jpg.tobytes() if ok else None,
            })
        return out
