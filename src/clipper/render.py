"""Step 5: render each selected clip as a 1080x1920 Short with layout, word-by-word captions and a hook."""

import re
import subprocess
from pathlib import Path

from .common import ROOT, find_source, job_dir, log, path_of, probe, read_json, run, source_video, tool, write_json
from . import review_queue


def even(x: float) -> int:
    return int(x) // 2 * 2


def face_boxes(video: Path, start: float, end: float, timed: bool = False
               ) -> tuple[list, int]:
    """Largest face per frame as (x, y, w, h) fractions of the frame, sampled twice per second.
    If timed=True, returns ([(t, cx_frac), ...], samples) for dynamic tracking."""
    import cv2

    # YuNet (OpenCV zoo, 230 KB) finds small and angled facecam faces far more reliably than Haar cascades.
    detector = cv2.FaceDetectorYN.create(str(ROOT / "models" / "face_detection_yunet_2023mar.onnx"), "",
                                         (640, 360), 0.6, 0.3, 50)
    cap = cv2.VideoCapture(str(video))
    boxes: list = []
    timed_pts: list[tuple[float, float]] = []
    samples = 0
    t = start
    while t < end:
        samples += 1
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            t += 0.5
            continue
        # 640 px keeps small facecam faces detectable while staying fast.
        scale = 640 / frame.shape[1]
        small = cv2.resize(frame, None, fx=scale, fy=scale)
        H, W = small.shape[:2]
        detector.setInputSize((W, H))
        _, faces = detector.detect(small)
        if faces is not None and len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])[:4]
            bx = (float(max(x, 0) / W), float(max(y, 0) / H), float(w / W), float(h / H))
            boxes.append(bx)
            timed_pts.append((t, bx[0] + bx[2] / 2))
        t += 0.5
    cap.release()
    return (timed_pts, samples) if timed else (boxes, samples)


def _ema_smooth(pts: list[tuple[float, float]], alpha: float,
                sw: int, cw: int) -> list[tuple[float, int]]:
    """Apply EMA to (time, cx_frac) pairs and convert to (time, pixel_x) crop offsets."""
    if not pts:
        return []
    smoothed = []
    ema_x = pts[0][1]
    for t, cx in pts:
        ema_x = alpha * cx + (1 - alpha) * ema_x
        px = int(min(max(ema_x * sw - cw / 2, 0), sw - cw))
        smoothed.append((t, px))
    return smoothed


def _ema_track(pts: list[tuple[float, float]], alpha: float) -> list[tuple[float, float]]:
    """EMA over (time, cx_frac) pairs; returns smoothed (time, cx_frac)."""
    out, ema = [], None
    for t, cx in pts:
        ema = cx if ema is None else alpha * cx + (1 - alpha) * ema
        out.append((t, ema))
    return out


def _track_crop_expr(track: list[tuple[float, int]], seg_start: float) -> str:
    """Build an ffmpeg eval expression for the linearly-interpolated crop x offset."""
    if not track:
        return "0"
    pts = [(t - seg_start, x) for t, x in track]
    if len(pts) == 1:
        return str(pts[0][1])
    # Build nested if(lt(t, t_i), lerp(x_prev, x_i, ...), ...)
    expr = str(pts[-1][1])
    for (t0, x0), (t1, x1) in reversed(list(zip(pts[:-1], pts[1:]))):
        dt = max(t1 - t0, 0.01)
        lerp = f"({x0}+({x1}-{x0})*(t-{t0:.3f})/{dt:.3f})"
        expr = f"if(lt(t,{t1:.3f}),{lerp},{expr})"
    # Clamp to valid range before t0
    expr = f"if(lt(t,{pts[0][0]:.3f}),{pts[0][1]},{expr})"
    return expr


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[len(xs) // 2]


def facecam_rect(boxes: list, info: dict, panel_aspect: float) -> list[float]:
    """A crop around a small, static face (a streamer's facecam overlay) with the top panel's aspect ratio."""
    sw, sh = info["width"], info["height"]
    fx, fy, fw, fh = (median([b[i] for b in boxes]) for i in range(4))
    w = min(fw * sw * 3.2, sw)
    h = min(w / panel_aspect, sh)
    cx, cy = (fx + fw / 2) * sw, (fy + fh / 2) * sh + h * 0.12
    x = min(max(cx - w / 2, 0), sw - w)
    y = min(max(cy - h / 2, 0), sh - h)
    return [x / sw, y / sh, w / sw, h / sh]


def pick_layout(cfg: dict, source: dict, video: Path, info: dict, start: float,
                end: float) -> tuple[str, float | None, list[float] | None, list | None]:
    """Returns (layout, median_cx, facecam_rect, timed_track).
    timed_track is list[(t, cx_frac)] for center layout when track_face=True, else None."""
    rc = cfg["render"]
    layout = source.get("layout", "auto")
    if layout == "split" and source.get("facecam"):
        return "split", None, source["facecam"], None
    if layout == "fit":
        return "fit", None, None, None
    if layout == "shots":
        return "shots", None, None, None

    want_track = rc.get("track_face", False) and layout in ("center", "auto")
    if want_track:
        timed, samples = face_boxes(video, start, end, timed=True)
        boxes_for_static, _ = face_boxes(video, start, end)
    else:
        boxes_for_static, samples = face_boxes(video, start, end)
        timed = None

    ratio = len(boxes_for_static) / max(samples, 1)
    cx = median([b[0] + b[2] / 2 for b in boxes_for_static]) if boxes_for_static else None

    if layout == "center":
        return "center", cx, None, timed
    if ratio >= rc["face_min_ratio"]:
        xs = [b[0] + b[2] / 2 for b in boxes_for_static]
        ys = [b[1] + b[3] / 2 for b in boxes_for_static]
        spread = max(max(xs) - min(xs), max(ys) - min(ys)) if len(boxes_for_static) > 2 else 1.0
        small = median([b[2] for b in boxes_for_static]) < rc["facecam_max_face"]
        if small and spread < rc["facecam_max_spread"]:
            top_h = even(rc["height"] * rc["split_top_ratio"])
            return "split", None, facecam_rect(boxes_for_static, info, rc["width"] / top_h), None
        if not small:
            return "center", cx, None, timed
    return "fit", None, None, None


def crop_x_bounds(source: dict, sw: int, cw: int, crop_h: float) -> tuple[int, int]:
    """Allowed crop x range so the crop never shows a source's overlay rects (chat, viewer count,
    sub goal, logos). Rects are [x, y, w, h] fractions; only rects reaching into the crop height count."""
    lo, hi = 0, sw - cw
    for x, y, w, h in source.get("avoid", []):
        if y >= crop_h:
            continue  # below the crop (the bottom strip is cut off by crop_h)
        if x + w / 2 > 0.5:
            hi = min(hi, int(x * sw) - cw)
        else:
            lo = max(lo, int((x + w) * sw))
    if lo > hi:
        return 0, sw - cw  # cannot avoid all of them: fall back to the full range
    return lo, hi


def layout_filter(cfg: dict, info: dict, layout: str, cx: float | None, facecam: list[float] | None,
                  inp: str, out: str, track: list | None = None, seg_start: float = 0.0,
                  face_at: float = 0.5, source: dict | None = None) -> str:
    rc = cfg["render"]
    W, H = rc["width"], rc["height"]
    sw, sh = info["width"], info["height"]
    if layout == "fit":
        # Whole frame in the middle over a blurred, darkened copy: nothing important gets cropped away.
        return (
            f"[{inp}]split[{out}a][{out}b];"
            f"[{out}a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
            f"boxblur=20:2,eq=brightness=-0.12[{out}bg];"
            f"[{out}b]scale={even(W * rc['fit_zoom'])}:-2,crop={W}:ih[{out}fg];"
            f"[{out}bg][{out}fg]overlay=(W-w)/2:(H-h)/2,setsar=1[{out}]"
        )
    if layout == "split":
        fx, fy, fw, fh = facecam
        top_h = even(H * rc["split_top_ratio"])
        bot_h = H - top_h
        # Gameplay: centre crop of the full frame at the bottom panel's aspect ratio.
        gw = min(sw, even(sh * W / bot_h))
        return (
            f"[{inp}]split[{out}a][{out}b];"
            f"[{out}a]crop={even(fw * sw)}:{even(fh * sh)}:{even(fx * sw)}:{even(fy * sh)},"
            f"scale={W}:{top_h}:force_original_aspect_ratio=increase,crop={W}:{top_h}[{out}t];"
            f"[{out}b]crop={gw}:{sh}:{(sw - gw) // 2}:0,scale={W}:{bot_h}:force_original_aspect_ratio=increase,"
            f"crop={W}:{bot_h}[{out}g];"
            f"[{out}t][{out}g]vstack,setsar=1[{out}]"
        )
    src = source or {}
    # Source options for IRL/stream footage: crop_h drops the bottom strip (viewer count, sub goal),
    # avoid keeps overlay rects out of the crop, top_pad moves a face that sits at the very top of the
    # camera image below the top UI zone (the pad shows a blurred copy of the crop).
    crop_h = src.get("crop_h", 1.0)
    top_pad = int(src.get("top_pad", 0))
    ch = even(sh * crop_h)
    fg_h = H - top_pad
    cw = min(sw, even(ch * W / fg_h))
    lo, hi = crop_x_bounds(src, sw, cw, crop_h)
    face_at = face_at if source is None else FACE_X / W  # campaign sources: keep the face clear of the buttons
    rc = cfg["render"]
    if track and rc.get("track_face", False):
        smoothed = [(t, int(min(max(cxf * sw - cw * face_at, lo), hi)))
                    for t, cxf in _ema_track(track, rc.get("track_alpha", 0.15))]
        x_expr = _track_crop_expr(smoothed, seg_start)
    else:
        center = (cx if cx is not None else 0.5) * sw
        # face_at: where the face lands across the output width (0.5 = middle).
        x_expr = str(int(min(max(center - cw * face_at, lo), hi)))
    # Quoted: the tracking expression contains commas, which would otherwise split the filter chain.
    crop = f"crop=w={cw}:h={ch}:x='{x_expr}':y=0"
    if not top_pad:
        return f"[{inp}]{crop},scale={W}:{H},setsar=1[{out}]"
    return (f"[{inp}]{crop},split[{out}a][{out}b];"
            f"[{out}a]scale={W}:{H},boxblur=24:2,eq=brightness=-0.15[{out}bg];"
            f"[{out}b]scale={W}:{fg_h}[{out}fg];"
            f"[{out}bg][{out}fg]overlay=0:{top_pad},setsar=1[{out}]")


# ---------- per-shot layout ("shots") ----------
SCENE_CUT_SCORE = 4.0   # scdet score; real camera cuts score ~10, in-shot motion stays below 1
MIN_SHOT = 0.5          # s, shorter shots are merged into the previous one
DUO_MIN_SEP = 0.25
FACE_CROP_H = 0.9       # close-up crop uses the top 90% of the source frame
FACE_MIN_W = 0.0        # 0 = always face-crop a single person; fit puts edge faces in the button column      # two faces at least this far apart (fraction of width) -> stacked duo layout


def scene_cuts(video: Path, start: float, dur: float) -> list[tuple[float, int]]:
    """Camera cuts inside [start, start+dur] as (seconds from start, frame index).

    The frame index is the first frame of the new shot, counted exactly like the render input
    (same -ss), so layout switches can be placed on that frame with trim start_frame/end_frame.
    """
    res = subprocess.run(
        [tool("ffmpeg"), "-v", "error", "-ss", f"{start:.6f}", "-t", f"{dur:.6f}", "-i", str(video), "-an",
         "-vf", "setpts=PTS-STARTPTS,scale=320:-2,scdet=threshold=100,metadata=print:file=-", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
    cuts, t, n = [], None, None
    for line in res.stdout.splitlines():
        if "pts_time:" in line:
            n = int(line.split("frame:", 1)[1].split()[0])
            t = float(line.rsplit("pts_time:", 1)[1].split()[0])
        elif "scd.score=" in line and t is not None and float(line.split("=", 1)[1]) >= SCENE_CUT_SCORE:
            cuts.append((t, n))
    return cuts


def all_faces(cap, detector, t: float) -> list[tuple[float, float, float, float]]:
    """Every face in the frame at t as (x, y, w, h) fractions."""
    import cv2
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    if not ok:
        return []
    scale = 640 / frame.shape[1]
    small = cv2.resize(frame, None, fx=scale, fy=scale)
    H, W = small.shape[:2]
    detector.setInputSize((W, H))
    _, faces = detector.detect(small)
    if faces is None:
        return []
    return [(float(max(f[0], 0) / W), float(max(f[1], 0) / H), float(f[2] / W), float(f[3] / H))
            for f in faces if f[2] / W >= 0.02]


W_ASPECT_PANEL = 1080 / 1920  # output aspect; a duo panel is twice as wide relative to its height


def fps_of(video: Path) -> float:
    return probe(video)["fps"]


def plan_shots(video: Path, info: dict, start: float, end: float) -> list[dict]:
    """Split [start, end] at camera cuts and pick a static layout per shot.

    kind "face": one framing, 9:16 crop centred on the main face (static, no tracking inside a shot)
    kind "duo":  two people far apart (wide two-shot), stacked top/bottom
    kind "fit":  no usable face
    Times are relative to start.
    """
    import cv2
    dur = end - start
    all_cuts = scene_cuts(video, start, dur)
    # Cuts this close to a segment edge would leave a few frames of another camera (a flash).
    # Those frames are replaced by holding the neighbouring shot's edge frame (audio continues).
    lead = max([c for c in all_cuts if 0 < c[0] < MIN_SHOT], default=None, key=lambda c: c[0])
    tail = min([c for c in all_cuts if dur - MIN_SHOT < c[0] < dur], default=None, key=lambda c: c[0])
    cuts = [c for c in all_cuts if MIN_SHOT <= c[0] <= dur - MIN_SHOT]
    merged = [lead if lead else (0.0, 0)]
    for c in cuts:
        if c[0] - merged[-1][0] >= MIN_SHOT:
            merged.append(c)
    merged.append((tail[0], tail[1]) if tail else (dur, None))

    detector = cv2.FaceDetectorYN.create(str(ROOT / "models" / "face_detection_yunet_2023mar.onnx"), "",
                                         (640, 360), 0.6, 0.3, 50)
    cap = cv2.VideoCapture(str(video))
    shots = []
    for (t0, f0), (t1, f1) in zip(merged[:-1], merged[1:]):
        n = max(1, int((t1 - t0 - 0.3) / 0.5))
        times = [start + t0 + 0.15 + (t1 - t0 - 0.3) * (k + 0.5) / n for k in range(n)]
        samples = [all_faces(cap, detector, t) for t in times]
        with_faces = [f for f in samples if f]
        shot = {"t0": t0, "t1": t1, "f0": f0, "f1": f1, "kind": "fit"}
        if len(with_faces) * 2 >= len(samples):
            pairs = []
            for faces in with_faces:
                top2 = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[:2]
                if len(top2) == 2:
                    l, r = sorted(top2, key=lambda b: b[0])
                    if (r[0] + r[2] / 2) - (l[0] + l[2] / 2) >= DUO_MIN_SEP:
                        pairs.append((l, r))
            duo = None
            # Duo only if both people are visible nearly all the time: a person leaning out of
            # frame for a second leaves an empty panel.
            if len(pairs) >= 0.8 * len(samples):
                cxs = lambda k: [p[k][0] + p[k][2] / 2 for p in pairs]  # noqa: E731
                duo = {"kind": "duo",
                       "left": tuple(median([p[0][i] for p in pairs]) for i in range(4)),
                       "right": tuple(median([p[1][i] for p in pairs]) for i in range(4)),
                       # Horizontal range each person moves through in the shot: the static crop
                       # must cover it (someone leaning back laughing left a panel without a head).
                       "left_span": (min(cxs(0)), max(cxs(0))),
                       "right_span": (min(cxs(1)), max(cxs(1)))}
                if duo_panels(duo, info, 2 * W_ASPECT_PANEL) is None:
                    duo = None  # too soft: fall back to face/fit below
            if duo:
                shot.update(duo)
            else:
                # Face crop on the person who is visible most of the shot (largest-face-per-frame
                # can flip between two people and put the crop in the empty middle).
                centres = [b[0] + b[2] / 2 for f in with_faces for b in f]
                if max(centres) - min(centres) > DUO_MIN_SEP:
                    left = [c for c in centres if c < 0.5]
                    right = [c for c in centres if c >= 0.5]
                    side = left if len(left) > len(right) else right
                    cx = median(side)
                else:
                    cx = median(centres)
                # A small face (wide shot) blown up to 9:16 is visibly soft: show the frame instead.
                fws = [b[2] for f in with_faces for b in f]
                if FACE_MIN_W and median(fws) < FACE_MIN_W:
                    shot.update(kind="fit")
                else:
                    shot.update(kind="face", cx=cx)
        shots.append(shot)
    cap.release()
    if shots:
        shots[0]["hold_lead"] = lead[1] if lead else 0           # frames to fill with the first frame
        total = round(dur * fps_of(video))
        shots[-1]["hold_tail"] = (total - tail[1]) if tail else 0  # frames to fill with the last frame
    return shots


DUO_MIN_SRC_H = 600  # px of source height per duo panel; below this the ~2x upscale is visibly soft


def duo_panel(box: tuple, info: dict, aspect: float, min_h: float = 0,
              span: tuple[float, float] | None = None, bottom: bool = False) -> tuple[int, int, int, int]:
    """Source crop (w, h, x, y) around one person for a W x H/2 panel, face slightly below centre.

    span: (min, max) face centre x over the shot; the crop is widened so the face stays inside
    with 1.5 face widths of margin on both sides wherever the person moves.
    """
    sw, sh = info["width"], info["height"]
    fx, fy, fw, fh = box
    need_w = ((span[1] - span[0]) + 3 * fw) * sw if span else 0
    h = min(max(fh * sh * 4.5, sh * 0.45, min_h, need_w / aspect), sh)
    w = min(h * aspect, sw)
    h = w / aspect
    mid = (span[0] + span[1]) / 2 if span else fx + fw / 2
    # Top panel: face a bit below the panel centre (room for the hook, clear of the top UI).
    # Bottom panel: face in the upper third so it stays above the Shorts bottom UI (> ~1570 px).
    cx, cy = mid * sw, (fy + fh / 2) * sh + (h * 0.17 if bottom else -h * 0.08)
    x = min(max(cx - w / 2, 0), sw - w)
    y = min(max(cy - h / 2, 0), sh - h)
    return even(w), even(h), even(x), even(y)


def duo_panels(shot: dict, info: dict, aspect: float) -> tuple | None:
    """Both duo crops with at least DUO_MIN_SRC_H source pixels of height, or None.

    A panel is widened to DUO_MIN_SRC_H when the natural framing is tighter (sharper, less
    upscaling); if the two widened crops would overlap, the duo layout is not usable.
    """
    l = duo_panel(shot["left"], info, aspect, DUO_MIN_SRC_H, shot.get("left_span"))
    r = duo_panel(shot["right"], info, aspect, DUO_MIN_SRC_H, shot.get("right_span"), bottom=True)
    if min(l[1], r[1]) < DUO_MIN_SRC_H or l[2] + l[0] > r[2]:
        return None
    return l, r


def push_in(cfg: dict, fps: float, anchor: tuple[float, float] | None = None) -> str:
    """Slow push-in from the first frame of a shot: zoom 1.0 + rate*t, capped.

    perspective resamples with sub-pixel precision (cubic), so the zoom does not step/jitter
    like crop/zoompan with integer offsets. `in` is the frame count since the shot started.
    anchor (output px) stays fixed while everything else moves away from it: on close-ups it
    sits at the top safe line so the forehead is not pushed into the top UI zone.
    """
    rc = cfg["render"]
    rate, cap = rc.get("push_in_rate", 0.0), rc.get("push_in_max", 1.0)
    if rate <= 0 or cap <= 1.0:
        return ""
    W, H = rc["width"], rc["height"]
    ax, ay = anchor or (W / 2, H / 2)
    d = f"(1-1/min(1+{rate}*in/{fps:.5f},{cap}))"
    x0, x1 = f"{ax:.1f}*{d}", f"W-{W - ax:.1f}*{d}"
    y0, y2 = f"{ay:.1f}*{d}", f"H-{H - ay:.1f}*{d}"
    return (f",perspective=x0='{x0}':y0='{y0}':x1='{x1}':y1='{y0}':x2='{x0}':y2='{y2}'"
            f":x3='{x1}':y3='{y2}':interpolation=cubic:eval=frame")


def shot_filter(cfg: dict, info: dict, shot: dict, inp: str, out: str) -> str:
    rc = cfg["render"]
    W, H = rc["width"], rc["height"]
    if shot["kind"] == "face":
        # Face left of centre (the right ~160 px are platform buttons), and only the top 90% of the
        # frame: shows put their logo in the bottom corner, which a full-height crop cut in half.
        sw, sh = info["width"], info["height"]
        ch = even(sh * FACE_CROP_H)
        cw = min(sw, even(ch * W / H))
        x = int(min(max(shot["cx"] * sw - cw * FACE_X / W, 0), sw - cw))
        return f"[{inp}]crop={cw}:{ch}:{x}:0,scale={W}:{H},setsar=1[{out}]"
    if shot["kind"] == "duo":
        ph = H // 2
        (lw, lh, lx, ly), (rw, rh, rx, ry) = duo_panels(shot, info, W / ph)
        return (
            f"[{inp}]split[{out}a][{out}b];"
            f"[{out}a]crop={lw}:{lh}:{lx}:{ly},scale={W}:{ph}[{out}t];"
            f"[{out}b]crop={rw}:{rh}:{rx}:{ry},scale={W}:{H - ph}[{out}u];"
            f"[{out}t][{out}u]vstack,setsar=1[{out}]"
        )
    return layout_filter(cfg, info, "fit", None, None, inp, out)


def ass_time(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def ass_escape(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")")


def chunk_words(words: list[dict], max_words: int, max_sec: float) -> list[list[dict]]:
    chunks, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or w["s"] - cur[-1]["e"] > 0.6 or w["e"] - cur[0]["s"] > max_sec):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    # A chunk on screen for less than MIN_CAPTION_SEC only flashes: merge it into a neighbour.
    merged: list[list[dict]] = []
    for ch in chunks:
        if merged and ch[-1]["e"] - ch[0]["s"] < MIN_CAPTION_SEC and ch[0]["s"] - merged[-1][-1]["e"] <= 0.6:
            merged[-1] = merged[-1] + ch
        elif merged and merged[-1][-1]["e"] - merged[-1][0]["s"] < MIN_CAPTION_SEC \
                and ch[0]["s"] - merged[-1][-1]["e"] <= 0.6:
            merged[-1] = merged[-1] + ch
        else:
            merged.append(ch)
    return merged


MIN_CAPTION_SEC = 0.25


_NON_LATIN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af\u0400-\u04ff]")


_HESITATION = re.compile(r"(?i)^(um+|uh+|uhm+|erm+|hmm+|ah+|eh+)[,.!?]*$")


def caption_words(words: list[dict]) -> list[dict]:
    """Words that may appear in captions: no CJK/Cyrillic text (e.g. Japanese speech in an IRL stream)
    and no hesitation sounds (um, uh); real fillers like "like" stay."""
    return [w for w in words if not _NON_LATIN.search(w["w"]) and not _HESITATION.match(w["w"])]


def merge_punct(words: list[dict]) -> list[dict]:
    """Glue punctuation-only tokens onto the previous word: '50' '%' -> '50%'."""
    out: list[dict] = []
    for w in words:
        if out and (re.fullmatch(r"[^\w]+", w["w"]) or w["w"].startswith("-")):
            out[-1] = {**out[-1], "w": out[-1]["w"] + w["w"], "e": max(out[-1]["e"], w["e"])}
        else:
            out.append(dict(w))
    return out


_PROFANITY = re.compile(r"(?i)\b(motherf|f)u(ck\w*)|(?<![a-z])((?:bull)?sh)i(t\w*)|\b(b)i(tch\w*)|\b(c)u(nt\w*)")


AUDIO_FADE_IN = 0.018   # s, fade in at the start of every segment
AUDIO_FADE_OUT = 0.05   # s, fade out at the end of every segment (Whisper word ends run early)
SNAP_PRE = 0.09         # s of pre-roll before the first word, never before the previous word's end
SNAP_POST = 0.22        # s of tail after the last word, never past the next word's start - 0.05


def quietest(audio_fn, t0: float, t1: float, sr: int = 16000) -> float | None:
    """Time of the lowest-energy 20 ms window in [t0, t1] (5 ms hop), or None without audio."""
    import numpy as np
    if audio_fn is None or t1 - t0 < 0.03:
        return None
    a = audio_fn(t0, t1 - t0)
    win, hop = int(0.02 * sr), int(0.005 * sr)
    if len(a) < win:
        return None
    rms = [float(np.sqrt(np.mean(a[i:i + win] ** 2))) for i in range(0, len(a) - win + 1, hop)]
    k = min(range(len(rms)), key=rms.__getitem__)
    return t0 + (k * hop + win / 2) / sr


def first_silence(audio_fn, t0: float, t1: float, sr: int = 16000, min_len: float = 0.06) -> float | None:
    """Start (+30 ms) of the first silence of at least min_len in [t0, t1], or None.

    min_len skips the short closures inside words ("don't know" has a ~30 ms gap), which a single
    20 ms window would mistake for the end of speech.
    """
    import numpy as np
    if audio_fn is None or t1 - t0 < 0.03:
        return None
    a = audio_fn(t0, t1 - t0 + min_len)
    win, hop = int(0.02 * sr), int(0.005 * sr)
    quiet = [20 * np.log10(np.sqrt(np.mean(a[i:i + win] ** 2)) + 1e-9) < SILENCE_DB
             for i in range(0, max(len(a) - win, 0) + 1, hop)]
    need = max(1, int((min_len - 0.02) / 0.005) + 1)
    run = 0
    for k, q in enumerate(quiet):
        run = run + 1 if q else 0
        if run >= need:
            start = (k - run + 1) * hop / sr
            if start > t1 - t0:
                return None
            return t0 + start + 0.03
    return None


def silence_runs(audio_fn, t0: float, t1: float, min_len: float = 0.06, sr: int = 16000) -> list[tuple[float, float]]:
    """Silent stretches (20 ms RMS < SILENCE_DB) of at least min_len inside [t0, t1]."""
    import numpy as np
    if audio_fn is None or t1 - t0 < min_len:
        return []
    a = audio_fn(t0, t1 - t0)
    win, hop = int(0.02 * sr), int(0.005 * sr)
    quiet = [20 * np.log10(np.sqrt(np.mean(a[i:i + win] ** 2)) + 1e-9) < SILENCE_DB
             for i in range(0, max(len(a) - win, 0) + 1, hop)]
    runs, k = [], 0
    while k < len(quiet):
        if quiet[k]:
            j = k
            while j < len(quiet) and quiet[j]:
                j += 1
            a_, b_ = t0 + k * hop / sr, t0 + ((j - 1) * hop + win) / sr
            if b_ - a_ >= min_len:
                runs.append((a_, b_))
            k = j
        else:
            k += 1
    return runs


def level_db(audio_fn, t0: float, dur: float) -> float:
    import numpy as np
    a = audio_fn(t0, dur)
    return float(20 * np.log10(np.sqrt(np.mean(a ** 2)) + 1e-9)) if len(a) else -99.0


def snap_to_words(words: list[dict], start: float, end: float, sentence_end: bool = False,
                  max_trim: float | None = None, audio_fn=None, timed: list[dict] | None = None,
                  end_tol: float = 0.3) -> tuple[float, float]:
    """Snap a segment to word boundaries so a cut never lands mid-word.

    Start: beginning of the first word starting at or after start-0.05.
    End: end of the last word ending at or before end+0.3; with sentence_end, if one of the
    words in the segment ends a sentence (. ! ?), the segment closes at the last such word
    (with max_trim: only if that cuts at most max_trim seconds, so the main segment does
    not end on a dangling "So").
    Then SNAP_PRE / SNAP_POST of breathing room are added, bounded by the neighbouring words,
    because Whisper word ends are typically early and a hard cut bites off the last syllable.
    With audio_fn(start, duration) -> 16 kHz samples, the end is refined to the quietest point
    in [last word end, +0.3 s] (Whisper often glues the next word's start onto the previous end,
    so the next-word bound alone can leave no tail at all).
    Which words belong to the segment is decided on `words` (the timing the ranker saw);
    the cut times come from `timed` (onset-refined copy, same order) when given.
    Falls back to the given times when no word fits.
    """
    first = next((w for w in words if w["s"] >= start - 0.05), None)
    if first is None:
        return start, end
    # end_tol: slack for imprecise ends; quote-anchored ends are exact word ends (tiny slack only).
    inside = [w for w in words if w["s"] >= first["s"] and w["e"] <= end + end_tol]
    if not inside:
        return start, end
    last = inside[-1]
    if sentence_end:
        enders = [w for w in inside if w["w"].rstrip("\"'”’)").endswith((".", "!", "?"))
                  and (max_trim is None or last["e"] - w["e"] <= max_trim)]
        if enders:
            last = enders[-1]
    i0, i1 = words.index(first), words.index(last)
    if timed is not None:
        words = timed
        first, last = words[i0], words[i1]
    s0 = first["s"] - SNAP_PRE
    if i0 > 0:
        s0 = max(s0, words[i0 - 1]["e"])
        # Whisper's previous-word end is often early: cut at the quietest point just before the
        # first word so the segment does not open with the previous word's tail.
        q = quietest(audio_fn, max(words[i0 - 1]["s"] + 0.05, first["s"] - 0.25), first["s"])
        if q is not None:
            s0 = q
    e1 = last["e"] + SNAP_POST
    nxt = words[i1 + 1] if i1 + 1 < len(words) else None
    if nxt:
        e1 = min(e1, nxt["s"] - 0.05)
    limit = last["e"] + 0.3 if nxt is None else min(last["e"] + 0.3, nxt["e"] - 0.1)
    # End at the first silence after the last word (the next sentence can start 100 ms later and
    # Whisper's next-word start is unreliable); fall back to the quietest point.
    # Search no further than the next word's end: past that, the next sentence is surely playing.
    hi = last["e"] + 0.35 if nxt is None else min(last["e"] + 0.35, nxt["e"] - 0.05)
    # Whisper word ends can also be late (medium ran ~0.4 s late): start looking for the pause a bit
    # before the reported end, but not inside the first part of the word itself.
    lo = max(last["s"] + 0.15, last["e"] - 0.4)
    q = first_silence(audio_fn, lo, hi)
    if q is None:
        # Connected speech: cut at the quietest point before the next word really starts.
        q = quietest(audio_fn, last["e"], last["e"] + 0.3 if nxt is None else min(last["e"] + 0.3, nxt["s"] + 0.1))
    if q is not None:
        e1 = q
    s0 = min(s0, first["s"])
    e1 = max(e1, last["s"] + 0.15)
    if audio_fn is not None:
        # Safety net for wrong Whisper times: a cut must sit in silence. A loud start moves back
        # to the end of the nearest pause (<= 0.8 s), a loud end forward to the next pause.
        if level_db(audio_fn, s0, 0.03) > SILENCE_DB:
            runs = silence_runs(audio_fn, s0 - 0.3, s0 + 0.02)  # 0.8 s pulled in a stutter + dead air
            if runs:
                s0 = runs[-1][1] - 0.03
        if level_db(audio_fn, e1 - 0.03, 0.03) > -30.0:
            runs = silence_runs(audio_fn, e1 - 0.02, e1 + 0.8)
            if runs:
                e1 = runs[0][0] + 0.03
    return round(s0, 3), round(e1, 3)


EDGE_CUT = 0.25     # s; camera cuts this close to a segment edge become the edge
SILENCE_DB = -35.0   # 20 ms RMS below this counts as silence for word-timing refinement


def refine_words(words: list[dict], audio_fn, t0: float, t1: float) -> list[dict]:
    """Pin Whisper word starts to audible onsets (Whisper word times are often 0.2-0.6 s off).

    Every onset after >= 50 ms of silence is a word start. It is given to the word whose start
    is nearest (within 0.5 s, keeping word order); that word starts there and the previous word
    ends where the silence began. Words inside continuous speech keep Whisper's timing.
    """
    import numpy as np
    if audio_fn is None or not words:
        return words
    a = audio_fn(t0, t1 - t0)
    hop, win = 160, 320
    db = np.array([20 * np.log10(np.sqrt(np.mean(a[i:i + win] ** 2)) + 1e-9)
                   for i in range(0, max(len(a) - win, 0), hop)])
    speech = db >= SILENCE_DB
    onsets = []  # (onset time, silence start)
    for i in range(5, len(speech)):
        if speech[i] and not speech[i - 5:i].any():
            j = i - 1
            while j > 0 and not speech[j - 1]:
                j -= 1
            onsets.append((t0 + i * 0.01, t0 + j * 0.01))
    out = [dict(w) for w in words]
    taken: set[int] = set()
    for o, sil in onsets:
        # Near starts (0.5 s), or a word whose overlong Whisper span contains the onset (<= 1 s in).
        cands = [k for k, w in enumerate(out)
                 if (abs(w["s"] - o) <= 0.5 or w["s"] < o < w["e"] and o - w["s"] <= 1.0) and k not in taken
                 and (k == 0 or out[k - 1]["s"] < o - 0.02)
                 and (k + 1 == len(out) or o < out[k + 1]["s"] - 0.02 or (k + 1) not in taken)]
        if not cands:
            continue
        # A word whose Whisper start lies in the silence before the onset must start at the onset.
        # A word that ends a phrase ("day," before the pause) belongs before the silence even when
        # Whisper put its start late: the onset after the pause is the next word's.
        ends_phrase = lambda k: out[k]["w"].rstrip("\"'”’)").endswith((",", ".", "?", "!"))  # noqa: E731
        moved = []
        for k in cands:
            if ends_phrase(k) and k + 1 < len(out) and k + 1 not in taken and abs(out[k + 1]["s"] - o) <= 1.0:
                moved.append(k + 1)
            else:
                moved.append(k)
        cands = sorted(set(moved))
        in_gap = [k for k in cands if sil - 0.02 <= out[k]["s"] < o]
        k = in_gap[0] if in_gap else min(cands, key=lambda k: abs(out[k]["s"] - o))
        if k > 0 and ends_phrase(k - 1) and out[k - 1]["s"] >= sil - 0.02:
            # The phrase-ending word sits before the pause: pull it in front of the silence.
            prev_s = out[k - 2]["s"] if k > 1 else out[k - 1]["s"] - 0.5
            out[k - 1]["e"] = round(sil, 3)
            out[k - 1]["s"] = round(max(prev_s + 0.02, sil - 0.25), 3)
        out[k]["s"] = round(o, 3)
        out[k]["e"] = max(out[k]["e"], o + 0.05)
        if k > 0 and out[k - 1]["e"] > sil:
            out[k - 1]["e"] = round(max(sil, out[k - 1]["s"] + 0.05), 3)
        taken.add(k)
    # Keep order strictly increasing after the moves.
    for k in range(1, len(out)):
        if out[k]["s"] <= out[k - 1]["s"]:
            out[k]["s"] = round(out[k - 1]["s"] + 0.02, 3)
            out[k]["e"] = max(out[k]["e"], out[k]["s"] + 0.05)
    return out


def out_frame_time(t: float, out_fps: float = 0.0) -> float:
    """ASS time for a picture change at source-frame time t (the subtitles filter runs on source
    frames, before the 30 fps conversion): floored to the centisecond, minus 5 ms, so the text
    moves on exactly the first frame of the new shot."""
    import math
    return max(0.0, math.floor((t - 0.005) * 100) / 100)


def frame_grid(s: float, e: float, info: dict, out_fps: float = 30.0) -> tuple[float, int]:
    """Snap a segment onto the source frame grid: (start on a frame time, whole frame count).

    The first video frame then sits exactly at the segment start and video and audio have the
    same length, so concat adds no padding and captions computed from the segments stay in sync.
    """
    import math
    fps, st = info["fps"], info.get("v_start", 0.0)
    # Start rounds down and end rounds up: both move towards the silence around the speech.
    k = math.floor((s - st) * fps + 1e-6)
    s_q = st + k / fps
    n = max(1, math.ceil((e - s_q) * fps - 1e-6))
    # 60 fps sources rendered at 30: whole output frames per segment, or every segment adds half a
    # frame of audio/video mismatch (67 ms over two segments failed check_output).
    r = round(fps / out_fps) if out_fps else 1
    if r > 1 and abs(fps / out_fps - r) < 0.01:
        n = -(-n // r) * r
    return s_q, n


def fix_names(word: str, fixes: dict) -> str:
    """Correct names Whisper misspells (Lacey -> Lacy), keeping punctuation and possessives."""
    for wrong, right in fixes.items():
        word = re.sub(rf"(?i)\b{re.escape(wrong)}\b", right, word)
    return word


def censor(word: str) -> str:
    """Mask strong profanity in burned-in captions (audio is untouched): FUCK -> F*CK."""
    def mask(m: re.Match) -> str:
        g = [x for x in m.groups() if x is not None]
        return g[0] + "*" + g[1]
    return _PROFANITY.sub(mask, word)


# Strictest safe zone over YouTube Shorts, TikTok and Instagram Reels (1080x1920): keep text and
# faces out of the top ~220 px, the bottom ~420 px and the right ~160 px (action buttons).
SAFE_LEFT, SAFE_RIGHT_X = 90, 920
SAFE_TOP = 220
FACE_X = 475                              # close-up face centre x (left of the button column)
CAP_X = (SAFE_LEFT + SAFE_RIGHT_X) // 2   # text centre, shifted left of the frame centre


def build_ass(cfg: dict, clip: dict, segments: list[tuple[float, float, float]], layout: str,
              shot_spans: list[tuple[float, float, str]] | None = None) -> str:
    """ASS captions for the output timeline; segments map source ranges to output offsets (cold open + main).

    shot_spans (output start, end, kind) switch the caption position per camera shot in the
    shots layout: lower on close-ups (off the face), at the seam on stacked duo shots. A caption
    line that crosses a cut is split there so each part sits at its shot's position.
    """
    cc, hc, rc = cfg["captions"], cfg["hook"], cfg["render"]
    W, H = rc["width"], rc["height"]
    y = cc[f"y_{layout}"]
    shot_y = {"face": cc.get("y_shots_face", y), "duo": cc.get("y_shots_duo", y), "fit": cc.get("y_fit", y)}

    def placed(t0: float, t1: float) -> list[tuple[float, float, int]]:
        if not shot_spans:
            return [(t0, t1, y)]
        parts = []
        for a, b, kind in shot_spans:
            lo, hi = max(t0, a), min(t1, b)
            if hi - lo > 0.005:
                parts.append((lo, hi, shot_y.get(kind, y)))
        return parts or [(t0, t1, y)]

    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "WrapStyle: 0", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Cap,{cc['font']},{cc['size']},{cc['color']},&H000000FF,&H00000000,&H64000000,"
        f"-1,0,0,0,100,100,0,0,1,{cc['outline']},2,5,{SAFE_LEFT},{W - SAFE_RIGHT_X},0,1",
        f"Style: Hook,{cc['font']},{hc['size']},&H00000000,&H000000FF,{hc['box']},{hc['box']},"
        # Box padding (outline 18) included: the box itself stays inside the safe zone.
        f"-1,0,0,0,100,100,0,0,3,18,0,5,{SAFE_LEFT + 20},{W - SAFE_RIGHT_X + 20},0,1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    if hc["enabled"] and clip.get("hook"):
        # The hook stays up through the cold open and the first seconds of the setup.
        cold = segments[0][1] - segments[0][0] if len(segments) > 1 else 0.0
        # Per shot like the captions: over the chest on close-ups (the top of the frame is the
        # forehead there), at the top on duo/fit shots.
        hook_y = {"face": hc.get("y_face", hc["y"]), "duo": hc["y"], "fit": hc["y"]}
        spans = shot_spans or [(0.0, 1e9, "")]
        t_end = cold + hc["seconds"]
        for a, b, kind in spans:
            lo, hi = max(0.0, a), min(t_end, b)
            if hi - lo > 0.005:
                lines.append(
                    f"Dialogue: 1,{ass_time(lo)},{ass_time(hi)},Hook,,0,0,0,,"
                    f"{{\\pos({CAP_X},{hook_y.get(kind, hc['y'])})}}{ass_escape(clip['hook'])}"
                )
    for seg_s, seg_e, off in segments:
        # A word is captioned only if its middle is inside the segment (no half-heard word at a cut).
        # A word is captioned only if it starts inside the segment and its middle is inside too:
        # no word from before the cut (cold open began after "To this day,"), no half-heard word at the end.
        words = [w for w in caption_words(clip.get("words", []))
                 if w["s"] >= seg_s - 0.02 and (w["s"] + w["e"]) / 2 <= seg_e]
        chunks = chunk_words(words, cc["max_words"], cc["max_chunk_sec"])
        for ci, chunk in enumerate(chunks):
            next_start = chunks[ci + 1][0]["s"] if ci + 1 < len(chunks) else None
            texts = [ass_escape(censor(fix_names(w["w"], clip.get("name_fixes", {}))).upper() if cc["uppercase"]
                                else censor(fix_names(w["w"], clip.get("name_fixes", {}))))
                     for w in chunk]
            pop = cc.get("pop", 1.0)
            pop_tag = f"{{\\fscx{int(pop*100)}\\fscy{int(pop*100)}\\t(0,80,\\fscx100\\fscy100)}}" if pop > 1.0 else ""
            for i, w in enumerate(chunk):
                t0 = w["s"] if i else chunk[0]["s"]
                t1 = chunk[i + 1]["s"] if i + 1 < len(chunk) else chunk[-1]["e"]
                if next_start is not None:
                    t1 = min(t1, next_start)  # two caption lines must never be on screen together
                t0, t1 = max(t0, seg_s), min(t1, seg_e)
                if t1 <= t0:
                    continue
                parts = [f"{{\\c{cc['highlight']}}}{t}{{\\c{cc['color']}}}" if j == i else t
                         for j, t in enumerate(texts)]
                for a, b, yy in placed(t0 - seg_s + off, t1 - seg_s + off):
                    lines.append(
                        f"Dialogue: 0,{ass_time(a)},{ass_time(b)},Cap,,0,0,0,,"
                        f"{{\\pos({CAP_X},{yy})}}{pop_tag}{' '.join(parts)}"
                    )
    return "\n".join(lines) + "\n"


def music_ratio(jd: Path, start: float, end: float, threshold: float) -> float | None:
    """Share of seconds in the clip where YAMNet hears music (Content ID risk)."""
    import numpy as np

    tags = jd / "audio_tags.npz"
    if not tags.exists():
        return None
    music = np.load(tags)["music"][int(start):int(end) + 1]
    return round(float((music >= threshold).mean()), 2) if len(music) else None


def check_output(path: Path) -> None:
    """Fail loudly on broken audio timing: every AAC packet but the last must be 1024 samples,
    and audio and video lengths must agree within 60 ms."""
    import json as _json
    res = subprocess.run([tool("ffprobe"), "-v", "error", "-select_streams", "a:0", "-show_entries",
                          "packet=duration", "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    durs = [int(x) for x in res.stdout.split() if x.strip().isdigit()]
    bad = [i for i, d in enumerate(durs[:-1]) if d != 1024]
    st = _json.loads(subprocess.run([tool("ffprobe"), "-v", "error", "-show_entries", "stream=codec_type,duration",
                                     "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout)
    lens = {x["codec_type"]: float(x["duration"]) for x in st["streams"]}
    gap = abs(lens.get("audio", 0) - lens.get("video", 0))
    if bad or not durs or gap > 0.06:
        raise RuntimeError(f"RENDER CHECK FAILED {path}: {len(bad)} audio packets != 1024 samples "
                           f"(first at #{bad[:5]}), audio {lens.get('audio')} s vs video {lens.get('video')} s")
    log(f"  check ok: {len(durs)} audio packets x 1024, audio/video diff {gap * 1000:.0f} ms")


def render(cfg: dict, job_id: str) -> list[Path]:
    rc = cfg["render"]
    jd = job_dir(cfg, job_id)
    job = read_json(jd / "job.json")
    source = find_source(cfg, job["source"])
    video = source_video(jd)
    info = probe(video)
    clips = read_json(jd / "clips.json")
    if clips is None:
        raise SystemExit("run rank first")
    fonts = path_of(cfg, "fonts")

    outputs = []
    mc = cfg["music"]
    for clip in clips:
        clip["music"] = music_ratio(jd, clip["start"], clip["end"], mc["threshold"])
        if clip["music"] is not None and clip["music"] > mc["max_ratio"]:
            log(f"skipping {clip['id']}: music in {clip['music']:.0%} of the clip (Content ID risk)")
            continue
        cdir = jd / "clips" / clip["id"]
        cdir.mkdir(parents=True, exist_ok=True)
        out = cdir / "short.mp4"
        from .transcribe import load_audio
        audio_fn = lambda t0, d: load_audio(video, t0, d)  # noqa: E731
        clip = {**clip, "name_fixes": source.get("name_fixes", {})}
        raw_words = merge_punct(clip.get("words", []))
        words = raw_words
        if words and not clip.get("words_whisper"):
            # Old clips without forced alignment: fall back to onset pinning.
            words = refine_words(words, audio_fn, min(w["s"] for w in words) - 0.5, max(w["e"] for w in words) + 0.5)
            clip = {**clip, "words": words}
        tol = 0.02 if clip.get("quotes") else 0.3
        main = snap_to_words(raw_words, clip["start"], clip["end"], sentence_end=True, max_trim=1.5,
                             audio_fn=audio_fn, timed=words, end_tol=tol)
        cold = (clip.get("payoff_start"), clip.get("payoff_end"))
        if cold[0] is not None:
            cold = snap_to_words(raw_words, cold[0], cold[1], sentence_end=True, audio_fn=audio_fn, timed=words,
                                 end_tol=tol)
        raw = []
        if rc["cold_open"] and cold[0] is not None and cold[1] - cold[0] >= 0.5:
            raw.append(cold)
        raw.append(main)
        # Frame-exact segments: (start, end, output offset), each a whole number of frames.
        segments, nframes, off = [], [], 0.0
        for a_, b_ in raw:
            s_q, n = frame_grid(a_, b_, info, rc["fps"])
            segments.append((s_q, s_q + n / info["fps"], off))
            nframes.append(n)
            off += n / info["fps"]
        # A camera cut within EDGE_CUT of a segment edge leaves 1-6 frames of the other camera
        # (a flash): move the edge onto the cut.
        edge = int(EDGE_CUT * info["fps"])
        trimmed, off = [], 0.0
        for k, (s_, e_, _) in enumerate(segments):
            n = nframes[k]
            cuts = [f for _, f in scene_cuts(video, s_, e_ - s_)]
            lead = max([f for f in cuts if 0 < f <= edge], default=0)
            tail = min([f for f in cuts if n - edge <= f < n], default=n)
            fps = info["fps"]

            def speech_in(a: float, b: float) -> bool:
                return any(w["s"] < b and w["e"] > a for w in words)
            # Trimming more than a 2-frame flash must not cut speech (a camera cut on the last word
            # took "know" off "I actually don't know."); keep those frames as a quick cut instead.
            if lead > 2 and speech_in(s_, s_ + lead / fps):
                lead = 0
            if n - tail > 2 and speech_in(s_ + tail / fps, s_ + n / fps):
                tail = n
            # 60 fps source at 30 fps output: keep the segment a whole number of output frames
            # (lead rounds up, length rounds down), or A/V drift by 33 ms per odd trim.
            r = round(fps / rc["fps"]) if rc["fps"] else 1
            if r > 1 and abs(fps / rc["fps"] - r) < 0.01 and (lead or tail < n):
                lead = -(-lead // r) * r
                tail = lead + max(r, (tail - lead) // r * r)
            if lead or tail < n:
                log(f"  segment {k}: trimmed {lead} lead / {n - tail} tail frames at camera cuts")
            s_ = s_ + lead / info["fps"]
            n = tail - lead
            nframes[k] = n
            trimmed.append((s_, s_ + n / info["fps"], off))
            off += n / info["fps"]
        segments = trimmed
        layout, cx, cam, track = pick_layout(cfg, source, video, info, main[0], main[1])
        if track:
            log(f"  tracking: {len(track)} face samples")
        seg_shots, shot_spans = [], None
        if layout == "shots":
            shot_spans = []
            for s_, e_, off_ in segments:
                plan = plan_shots(video, info, s_, e_)
                seg_shots.append(plan)
                for sh in plan:
                    # Output times from the frame-exact source boundaries.
                    # Held edge frames belong to the first/last shot's layout.
                    a = off_ + (0.0 if sh is plan[0] else sh["f0"] / info["fps"])
                    b = off_ + (e_ - s_ if sh is plan[-1] else sh["f1"] / info["fps"])
                    shot_spans.append((out_frame_time(a, rc["fps"]), out_frame_time(b, rc["fps"]), sh["kind"]))
        (cdir / "captions.ass").write_text(build_ass(cfg, clip, segments, layout, shot_spans))

        inputs, graph, concat = [], [], ""
        for i, (s, e, _) in enumerate(segments):
            fps = info["fps"]
            # -t half a frame short of n frames: exactly n video frames; audio is padded to n/fps.
            inputs += ["-ss", f"{s:.6f}", "-t", f"{(nframes[i] - 0.5) / fps:.6f}", "-i", video.resolve()]
            if layout == "shots":
                # One static layout per camera shot; trim on a single input keeps every frame exactly once.
                shots = seg_shots[i]
                log(f"  segment {i}: " + ", ".join(f"{x['t0']:.2f}-{x['t1']:.2f} {x['kind']}" for x in shots))
                k = len(shots)
                graph.append(f"[{i}:v]setpts=PTS-STARTPTS,split={k}" + "".join(f"[s{i}_{j}]" for j in range(k)))
                for j, sh in enumerate(shots):
                    anchor = (FACE_X, SAFE_TOP) if sh["kind"] == "face" else (FACE_X, rc["height"] / 2)
                    zoom = push_in(cfg, info["fps"], anchor)
                    # Frame indices, not timestamps: rounding a time can move the switch by one frame.
                    end_opt = "" if sh["f1"] is None else f":end_frame={sh['f1']}"
                    graph.append(f"[s{i}_{j}]trim=start_frame={sh['f0']}{end_opt},setpts=PTS-STARTPTS[t{i}_{j}]")
                    graph.append(shot_filter(cfg, info, sh, f"t{i}_{j}", f"p{i}_{j}"))
                    pad = ""
                    if sh.get("hold_lead"):
                        pad += f",tpad=start_mode=clone:start={sh['hold_lead']}"
                    if sh.get("hold_tail"):
                        pad += f",tpad=stop_mode=clone:stop={sh['hold_tail']}"
                    graph.append(f"[p{i}_{j}]null{pad}{zoom}[l{i}_{j}]")
                graph.append("".join(f"[l{i}_{j}]" for j in range(k)) + f"concat=n={k}:v=1:a=0[l{i}]")
            else:
                # Each segment gets its own face track (the cold open lies outside the main range).
                seg_track = track
                if track and len(segments) > 1:
                    seg_track, _ = face_boxes(video, s, e, timed=True)
                graph.append(f"[{i}:v]setpts=PTS-STARTPTS[z{i}]")
                graph.append(layout_filter(cfg, info, layout, cx, cam, f"z{i}", f"l{i}",
                                           track=seg_track, seg_start=s,
                                           source=source if source.get("campaign_url") else None))
            # Short audio fades at every cut so word-boundary splices do not click.
            fi, fo, dur = AUDIO_FADE_IN, AUDIO_FADE_OUT, (nframes[i] - 0.5) / fps
            graph.append(f"[{i}:a:0]afade=t=in:st=0:d={fi},afade=t=out:st={max(dur - fo, 0):.4f}:d={fo},"
                         f"apad=whole_dur={nframes[i] / fps:.6f}[a{i}]")
            concat += f"[l{i}][a{i}]"
        # loudnorm alone let true peaks reach -0.7 dBFS; limiter at -3 dBFS leaves room for AAC overshoot (~1.5 dB).
        audio = ",loudnorm=I=-14:TP=-1.5:LRA=11,aresample=192000,alimiter=limit=0.708:level=false:attack=1:release=50,aresample=48000" if rc["loudnorm"] else ""
        graph.append(f"{concat}concat=n={len(segments)}:v=1:a=1[cv][ca0]")
        from .memes import pick_effects, meme_filter_chain
        meme_fx = pick_effects(cfg, clip)
        W, H = rc["width"], rc["height"]
        if meme_fx:
            # Payoff moment inside the main segment, in output-timeline seconds.
            ms, me, moff = segments[-1]
            ps = clip.get("payoff_start")
            window = None
            if ps is not None and ms <= ps < me:
                pe = min(clip.get("payoff_end") or ps + 1.5, me)
                window = (moff + ps - ms, moff + pe - ms)
            graph.append(meme_filter_chain(meme_fx, "cv", "cv_fx", W, H, window))
            cap_in = "cv_fx"
            log(f"  meme effects: {meme_fx} at {window}")
        else:
            cap_in = "cv"
        graph.append(f"[{cap_in}]subtitles=captions.ass:fontsdir={fonts}[v]")
        # Concat leaves gaps/overlaps in the audio timestamps (1-sample packets, one 2.36 s packet):
        # timestamp-driven players (YouTube, Telegram, Safari) then garble the start. Rebuild
        # continuous timestamps from the sample count.
        graph.append(f"[ca0]anull{audio},aresample=48000,asetpts=N/SR/TB[a]")
        cmd = [
            tool("ffmpeg"), "-v", "error", "-y", *inputs, "-filter_complex", ";".join(graph),
            "-map", "[v]", "-map", "[a]",
            "-r", rc["fps"], "-c:v", "libx264", "-preset", rc["preset"], "-crf", rc["crf"], "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        ]
        (cdir / "graph.txt").write_text(";\n".join(graph))
        run(cmd + ["-movflags", "+faststart", out.name], cwd=cdir)
        check_output(out)
        # Preview frame for the review step (taken after the hook disappears).
        run([tool("ffmpeg"), "-v", "error", "-y", "-ss", min(3.0, (clip["end"] - clip["start"]) / 2),
             "-i", out.name, "-frames:v", "1", "-q:v", "3", "preview.jpg"], cwd=cdir)
        write_json(cdir / "clip.json", clip)
        from .qa import run_all
        problems = run_all(cfg, cdir, audio_fn, [(e_, off_ + e_ - s_) for s_, e_, off_ in segments],
                           [(s_, off_) for s_, e_, off_ in segments])
        if problems:
            # Known defect types never reach the reviewer: the clip stays out of the queue.
            log(f"  QA FAILED {clip['id']} ({len(problems)}), see qa.txt:")
            for p_ in problems[:8]:
                log(f"    {p_}")
            continue
        log("  QA ok")
        review_queue.add(cfg, clip, source, out)
        outputs.append(out)
        log(f"rendered {clip['id']} ({sum(e - s for s, e, _ in segments):.1f}s, cold open: {len(segments) > 1}, layout: {layout})")
    return outputs
