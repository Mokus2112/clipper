"""Identity lock: tell the streamer's face apart from other people (SFace embeddings).

The reference is a unit-norm centroid of the streamer's face embeddings, built once from the
streamer's own footage (`clipper identity build <name> <video>`) and stored in models/ref_<name>.npy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SFACE = ROOT / "models" / "face_recognition_sface_2021dec.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
YUNET = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
MATCH_COS = 0.363   # SFace's published cosine threshold for "same person"


def ref_path(name: str) -> Path:
    return ROOT / "models" / f"ref_{name}.npy"


def _recognizer():
    import cv2
    if not SFACE.exists():
        import urllib.request
        urllib.request.urlretrieve(SFACE_URL, SFACE)
    return cv2.FaceRecognizerSF.create(str(SFACE), "")


def _detector(w: int, h: int):
    import cv2
    d = cv2.FaceDetectorYN.create(str(YUNET), "", (w, h), 0.8, 0.3, 50)
    return d


def embed_faces(frame, min_px: int = 60) -> list[tuple[np.ndarray, tuple[float, float, float, float]]]:
    """All faces in a BGR frame -> [(unit embedding, (x, y, w, h) fractions)]."""
    import cv2
    rec = _recognizer()
    scale = 960 / frame.shape[1] if frame.shape[1] > 960 else 1.0
    small = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
    H, W = small.shape[:2]
    det = _detector(W, H)
    _, faces = det.detect(small)
    out = []
    for f in (faces if faces is not None else []):
        if f[2] < min_px * scale or f[3] < min_px * scale:
            continue
        feat = rec.feature(rec.alignCrop(small, f)).flatten()
        feat = feat / (np.linalg.norm(feat) + 1e-9)
        out.append((feat, (float(max(f[0], 0) / W), float(max(f[1], 0) / H), float(f[2] / W), float(f[3] / H))))
    return out


def build_reference(video: Path, name: str, step: float = 20.0, max_samples: int = 300,
                    sheet: Path | None = None) -> tuple[Path, int, int]:
    """Cluster every face found in the footage; the biggest cluster is the streamer.

    Returns (ref path, faces in the winning cluster, faces total). Writes a contact sheet of the
    cluster's members to `sheet` so a human/agent can confirm it is the right person.
    """
    import cv2
    cap = cv2.VideoCapture(str(video))
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    step = max(step, dur / max_samples)
    embs, crops = [], []
    t = 0.0
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if ok:
            for e, (x, y, w, h) in embed_faces(fr):
                embs.append(e)
                H, W = fr.shape[:2]
                crops.append(cv2.resize(fr[int(y * H):int((y + h) * H), int(x * W):int((x + w) * W)], (96, 96)))
        t += step
    cap.release()
    if not embs:
        raise RuntimeError("no faces found")
    E = np.array(embs)
    # Greedy clustering: each face joins the first cluster whose centroid matches, else starts one.
    cents, members = [], []
    for i, e in enumerate(E):
        for k, c in enumerate(cents):
            if float(e @ c) >= MATCH_COS:
                members[k].append(i)
                c2 = E[members[k]].mean(0)
                cents[k] = c2 / np.linalg.norm(c2)
                break
        else:
            cents.append(e)
            members.append([i])
    k = max(range(len(cents)), key=lambda j: len(members[j]))
    ref_path(name).parent.mkdir(exist_ok=True)
    np.save(ref_path(name), cents[k])
    if sheet is not None:
        pick = members[k][:: max(1, len(members[k]) // 24)][:24]
        rows = [np.hstack([crops[i] for i in pick[r:r + 8]]) for r in range(0, len(pick) - len(pick) % 8 or len(pick), 8) if len(pick[r:r + 8]) == 8]
        if rows:
            cv2.imwrite(str(sheet), np.vstack(rows))
    return ref_path(name), len(members[k]), len(embs)


def load_reference(name: str) -> np.ndarray | None:
    p = ref_path(name)
    return np.load(p) if p.exists() else None


def match_scores(frame, ref: np.ndarray) -> list[tuple[float, tuple[float, float, float, float]]]:
    """Every face in the frame with its cosine similarity to the reference: [(score, box)]."""
    return [(float(e @ ref), box) for e, box in embed_faces(frame)]
