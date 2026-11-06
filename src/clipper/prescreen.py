"""Cheap pre-render screen of ranked candidates (Codee #232.2).

Samples each candidate's time range in the source at ~1 fps and drops candidates that would fail
internal QA anyway: wrong/missing person (identity), several faces, too many camera cuts, burned-in
text in the top strip, digital silence. Reasons go to <job>/prescreen.json; thresholds in config.yaml
(`prescreen:`). The render-time identity drop stays as a safety net.
"""

import re
import subprocess

import cv2
import numpy as np

from .common import find_source, job_dir, log, read_json, source_video, tool, write_json

DEFAULTS = {
    "enabled": True,
    "fps": 1.0,
    "min_identity_ratio": 0.6,   # share of samples with the locked person (only if source.identity)
    "max_multi_face_ratio": 0.4, # share of samples with 2+ faces (no identity sources)
    "max_cuts_per_40s": 10,
    "max_top_text_ratio": 0.15,  # share of samples with OCR words in the top strip
    "max_top_edge_ratio": 0.15,  # share of samples with a text-free graphic banner (edges) in the top strip
    "max_silence_ratio": 0.25,   # share of 50 ms windows below silence_db
    "silence_db": -70.0,
    "animated_max_cuts_per_40s": 30,  # source `prescreen: {animated: true}` only (Codee #1799)
}


def top_band_has_edges(frame, strip_px: int = 130, out_h: int = 1920, thr: float = 0.003) -> bool:
    """Same test as qa.dense_top_hits, on a source frame scaled to the output height: edges in the top
    strip mean a broadcast banner or title graphic (flat footage has none). Fox/CNN banners carry no OCR text."""
    h = frame.shape[0]
    sc = out_h / h
    band = cv2.cvtColor(frame[:max(int(h * strip_px / out_h), 8)], cv2.COLOR_BGR2GRAY)
    band = cv2.resize(band, None, fx=sc, fy=sc)
    return cv2.Canny(band, 60, 150).mean() / 255 >= thr


def decide_full(m: dict, thr: dict, has_identity: bool, tracks_faces: bool = True,
                animated: bool = False) -> tuple[list[str], list[str]]:
    """(drop reasons, warnings) from a candidate's measurements. Without `animated` the warnings are always
    empty and the reasons are exactly the historical ones. Animated source (Codee #1799): the cuts limit is
    animated_max_cuts_per_40s and the top-strip banner/graphic detector only warns (the reviewer sees it)."""
    why, warn = [], []
    if has_identity and m["identity_ratio"] < thr["min_identity_ratio"]:
        why.append(f"identity ratio {m['identity_ratio']:.2f} < {thr['min_identity_ratio']}")
    if tracks_faces and not has_identity and m["multi_face_ratio"] > thr["max_multi_face_ratio"]:
        why.append(f"2+ faces in {m['multi_face_ratio']:.0%} of samples")
    max_cuts = thr["animated_max_cuts_per_40s"] if animated else thr["max_cuts_per_40s"]
    if m["cuts_per_40s"] > max_cuts:
        why.append(f"{m['cuts_per_40s']:.1f} cuts per 40 s")
    if m["top_text_ratio"] > thr["max_top_text_ratio"]:
        why.append(f"top-strip text in {m['top_text_ratio']:.0%} of samples")
    if m.get("top_edge_ratio", 0.0) > thr["max_top_edge_ratio"]:
        msg = f"burned-in banner/graphic in the top strip in {m['top_edge_ratio']:.0%} of samples"
        (warn if animated else why).append(msg)
    if m["silence_ratio"] > thr["max_silence_ratio"]:
        why.append(f"digital silence {m['silence_ratio']:.0%}")
    return why, warn


def decide(m: dict, thr: dict, has_identity: bool, tracks_faces: bool = True) -> list[str]:
    """Reasons to drop a candidate from its measurements (empty list = keep)."""
    return decide_full(m, thr, has_identity, tracks_faces)[0]


def silence_ratio(video, start: float, dur: float, db: float) -> float:
    raw = subprocess.run([tool("ffmpeg"), "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(video),
                          "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    n = len(a) // 800
    if n == 0:
        return 1.0
    rms = np.sqrt((a[:n * 800].reshape(n, 800) ** 2).mean(axis=1) + 1e-12)
    return float((20 * np.log10(rms) < db).mean())


def measure(video, start: float, end: float, identity: str | None, fps: float, silence_db: float) -> dict:
    import cv2
    from .render import scene_cuts, all_faces
    from .common import ROOT
    dur = end - start
    ref = None
    if identity:
        from . import identity as idm
        ref = idm.load_reference(identity)
        if ref is None:
            raise RuntimeError(f"identity '{identity}' has no reference")
    cap = cv2.VideoCapture(str(video))
    detector = None
    n = locked = multi = top = edge = 0
    t = start
    while t < end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        t += 1.0 / fps
        if not ok:
            continue
        n += 1
        if ref is not None:
            from . import identity as idm
            if any(s >= idm.LOCK_COS for s, _ in idm.match_scores(fr, ref)):
                locked += 1
        else:
            if detector is None:
                detector = cv2.FaceDetectorYN.create(
                    str(ROOT / "models" / "face_detection_yunet_2023mar.onnx"), "", (640, 360), 0.7, 0.3, 50)
            multi += len(all_faces(cap, detector, t - 1.0 / fps)) >= 2
        edge += top_band_has_edges(fr)
        band = cv2.cvtColor(fr[:max(int(fr.shape[0] * 0.08), 24)], cv2.COLOR_BGR2GRAY)
        if cv2.Canny(band, 100, 200).mean() / 255 >= 0.004:
            g = cv2.resize(band, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            res = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "6", "tsv"],
                                 input=cv2.imencode(".png", g)[1].tobytes(), capture_output=True)
            for row in res.stdout.decode(errors="ignore").splitlines()[1:]:
                c = row.split("\t")
                if len(c) == 12 and float(c[10]) >= 60 and re.search(r"[A-Za-z]{3,}", c[11]):
                    top += 1
                    break
    cap.release()
    n = max(n, 1)
    cuts = scene_cuts(video, start, dur)
    return {
        "samples": n,
        "identity_ratio": locked / n if ref is not None else 1.0,
        "multi_face_ratio": multi / n,
        "cuts_per_40s": len(cuts) * 40.0 / max(dur, 1.0),
        "top_text_ratio": top / n,
        "top_edge_ratio": float(edge) / n,
        "silence_ratio": silence_ratio(video, start, dur, silence_db),
    }


def prescreen(cfg: dict, job_id: str) -> list[dict]:
    """Filter <job>/clips.json in place; write prescreen.json with every candidate's measurements."""
    thr = {**DEFAULTS, **cfg.get("prescreen", {})}
    jd = job_dir(cfg, job_id)
    clips = read_json(jd / "clips.json", [])
    if not thr["enabled"] or not clips:
        return clips
    source = find_source(cfg, read_json(jd / "job.json")["source"])
    video = source_video(jd)
    keep, report = [], []
    for c in clips:
        try:
            m = measure(video, c["start"], c["end"], source.get("identity"), thr["fps"], thr["silence_db"])
            why, warn = decide_full(m, thr, bool(source.get("identity")), source.get("layout") != "center",
                                    animated=bool((source.get("prescreen") or {}).get("animated")))
            if warn:
                c["prescreen_warnings"] = warn
        except Exception as exc:  # a failed measurement must never block the pipeline
            log(f"prescreen: {c['id']} not measured ({exc}), keeping")
            m, why, warn = {"error": str(exc)}, [], []
        report.append({"id": c["id"], "start": c["start"], "end": c["end"], "metrics": m,
                       "dropped": bool(why), "reasons": why, "warnings": warn})
        if why:
            log(f"prescreen DROP {c['id']}: {'; '.join(why)}")
        else:
            keep.append(c)
    write_json(jd / "prescreen.json", report)
    write_json(jd / "clips.json", keep)
    log(f"prescreen: {len(keep)}/{len(clips)} candidates kept")
    return keep
