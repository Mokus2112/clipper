"""Step 5: render each selected clip as a 1080x1920 Short with layout, word-by-word captions and a hook."""

import re
import subprocess
from pathlib import Path

from .common import ROOT, find_source, job_dir, log, path_of, probe, read_json, run, source_video, tool, write_json
from . import review_queue


def even(x: float) -> int:
    return int(x) // 2 * 2


def face_boxes(video: Path, start: float, end: float, timed: bool = False,
               identity: str | None = None) -> tuple[list, int]:
    """Largest face per frame as (x, y, w, h) fractions of the frame, sampled twice per second.
    If timed=True, returns ([(t, cx_frac), ...], samples) for dynamic tracking."""
    import cv2

    # YuNet (OpenCV zoo, 230 KB) finds small and angled facecam faces far more reliably than Haar cascades.
    detector = cv2.FaceDetectorYN.create(str(ROOT / "models" / "face_detection_yunet_2023mar.onnx"), "",
                                         (640, 360), 0.7, 0.3, 50)
    cap = cv2.VideoCapture(str(video))
    boxes: list = []
    timed_pts: list[tuple[float, float]] = []
    samples = 0
    ref = None
    locked = others = 0
    if identity:
        from . import identity as idm
        ref = idm.load_reference(identity)
        if ref is None:
            raise RuntimeError(f"identity '{identity}' has no reference (clipper identity build)")
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
        if ref is not None:
            # Identity lock: follow only the face that matches the streamer, never "the largest face".
            from . import identity as idm
            cands = idm.match_scores(frame, ref)
            good = [(s_, b_) for s_, b_ in cands if s_ >= idm.LOCK_COS]
            others += sum(1 for s_, _ in cands if s_ < idm.LOCK_COS)
            if good:
                bx = max(good, key=lambda c: c[0])[1]
                boxes.append(bx)
                timed_pts.append((t, bx[0] + bx[2] / 2))
                locked += 1
            t += 0.5
            continue
        _, faces = detector.detect(small)
        if faces is not None and len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])[:4]
            bx = (float(max(x, 0) / W), float(max(y, 0) / H), float(w / W), float(h / H))
            boxes.append(bx)
            timed_pts.append((t, bx[0] + bx[2] / 2))
        t += 0.5
    cap.release()
    face_boxes.last_identity = {"locked": locked, "other_faces": others, "samples": samples}
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
    if len(pts) >= 3:  # median of 3 drops single-sample outliers (a hand or a background face)
        cxs = [p[1] for p in pts]
        med = ([sorted(cxs[:3])[1]] + [sorted(cxs[i - 1:i + 2])[1] for i in range(1, len(cxs) - 1)]
               + [sorted(cxs[-3:])[1]])  # the ends use the 3 nearest samples (a cut can land on the last one)
        pts = [(p[0], m) for p, m in zip(pts, med)]
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
        ident = source.get("identity")
        timed, samples = face_boxes(video, start, end, timed=True, identity=ident)
        boxes_for_static, _ = face_boxes(video, start, end, identity=ident)
    else:
        boxes_for_static, samples = face_boxes(video, start, end, identity=source.get("identity"))
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
        zoom = 1.0 if (source or {}).get("montage") else rc["fit_zoom"]  # montage: no side crop, in-game titles stay whole
        # Whole frame in the middle over a blurred, darkened copy: nothing important gets cropped away.
        return (
            f"[{inp}]split[{out}a][{out}b];"
            f"[{out}a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
            f"boxblur=20:2,eq=brightness=-0.12[{out}bg];"
            f"[{out}b]scale={even(W * zoom)}:-2,crop={W}:ih[{out}fg];"
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
                                         (640, 360), 0.7, 0.3, 50)
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
                    near = [max((b for b in f if abs(b[0] + b[2] / 2 - cx) < 0.15),
                                key=lambda b: b[2] * b[3], default=None) for f in with_faces]
                    near = [b for b in near if b]
                    shot.update(kind="face", cx=cx)
                    if near:
                        shot["ft"] = median([b[1] for b in near])
                        shot["fb"] = median([b[1] + b[3] for b in near])
                        shot.update(face_frame(shot, info))
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


FACE_BOTTOM_MAX = 1070   # close-up: chin (YuNet box bottom + beard margin) must end above the captions (captions.y_shots_face 1195, top ~1140)
FACE_TOP_MIN = 300       # ... and the forehead stay below the top UI zone


def face_frame(shot: dict, info: dict, W: int = 1080, H: int = 1920) -> dict:
    """Crop (height ch, top y0, in source px) for a close-up so the face box ends above the captions.

    Default is the top 90% of the frame. A very tight close-up (face box reaching y1500+) zooms out
    toward the full frame height and shifts the window down (face moves up) until the box fits
    FACE_TOP_MIN..FACE_BOTTOM_MAX. tight=True when it still does not fit or needed any change:
    the hook then goes to the top and the push-in is capped by push_cap.
    """
    sh = info["height"]
    ft, fb = shot["ft"] * sh, shot["fb"] * sh
    best, ch0 = None, sh * FACE_CROP_H
    def out(ch, y0):
        k = H / ch
        return (ft - y0) * k, (fb - y0) * k
    base_top, base_bot = out(ch0, 0)
    if base_bot <= FACE_BOTTOM_MAX:
        return {"ch": even(ch0), "y0": 0, "ft_out": base_top, "fb_out": base_bot}
    cands = []
    for ch in [ch0 + (sh - ch0) * i / 10 for i in range(11)]:
        for y0 in [i * (sh - ch) / 10 for i in range(11)]:
            t, b = out(ch, y0)
            if t >= FACE_TOP_MIN and b <= FACE_BOTTOM_MAX:
                cands.append((ch, y0, t, b))
    if cands:
        ch, y0, t, b = max(cands, key=lambda c: c[0] * 1000 - c[1])  # least zoom-out, then least shift
    else:  # does not fit: most zoomed out, window lowest, accept
        ch = sh
        y0 = 0
        t, b = out(ch, y0)
    return {"ch": even(ch), "y0": even(y0), "ft_out": t, "fb_out": b, "tight": True}


HOOK_SHORT_S = 2.0  # close-up without head room: hook visible at most this long
MIN_WORD_SHOW = 4 / 30  # a highlight shorter than 4 frames borrows time from a neighbour (or merges into it)


def balance_highlights(evs: list[list[float]], min_show: float) -> list[list[float]]:
    """[t0, t1, word index] highlight events of one caption line: every word keeps its highlight.

    A highlight under min_show flashes. It first takes the missing time from a neighbour that stays
    >= min_show (so edge words of a 3-word line still light up); only when no neighbour can spare it
    does it merge into the neighbour (the old behaviour)."""
    evs = [list(e) for e in evs]
    guard = 0
    while len(evs) > 1 and guard < 20:
        guard += 1
        k = next((j for j, e in enumerate(evs) if e[1] - e[0] < min_show - 1e-9), None)
        if k is None:
            break
        need = min_show - (evs[k][1] - evs[k][0])
        done = False
        for n in sorted((j for j in (k - 1, k + 1) if 0 <= j < len(evs)), key=lambda j: -(evs[j][1] - evs[j][0])):
            contiguous = abs(evs[k][0] - evs[n][1]) < 1e-6 if n < k else abs(evs[k][1] - evs[n][0]) < 1e-6
            if contiguous and evs[n][1] - evs[n][0] - need >= min_show - 1e-9:
                if n < k:
                    evs[n][1] -= need
                    evs[k][0] -= need
                else:
                    evs[n][0] += need
                    evs[k][1] += need
                done = True
                break
        if done:
            continue
        if k > 0:
            evs[k - 1][1] = evs[k][1]
        else:
            evs[1][0] = evs[0][0]
        del evs[k]
    return evs
HOOK_BH_EST = 130  # hook box height estimate at the default size, for layout decisions before the ASS pass


def hook_window_fit(cfg: dict, info: dict, plan: list[dict], off: float, seg_len: float,
                    hook_end: float) -> list[dict]:
    """Close-ups inside the hook window with no head-free room for the hook become 'fit' shots.

    A face that fills the frame leaves no slot above the hair or under the chin, so the hook would
    only appear on a later shot (reviewer FAIL 080536-c2: hook at 3.0 s for 1.1 s). The whole frame
    over the blurred copy leaves the top clear. A shot running well past the window is split there,
    so the close-up comes back as a punch-in.
    """
    cap_top = cfg["captions"].get("y_shots_face", 1195) - 60
    fps, out = info["fps"], []
    for sh in plan:
        a = off + sh["f0"] / fps
        if sh["kind"] != "face" or "ft_out" not in sh or a >= hook_end - 0.3:
            out.append(sh)
            continue
        ft, fb = sh["ft_out"], sh["fb_out"]
        above = SAFE_TOP + HOOK_BH_EST <= ft - (HOOK_HAIR * (fb - ft) + 10)
        below = fb + 40 + HOOK_BH_EST + HOOK_CAP_GAP <= cap_top
        if above or below:
            out.append(sh)
            continue
        cut = round((hook_end - off) * fps)
        whole = {k: v for k, v in sh.items() if k != "tight"} | {"kind": "fit"}
        f_end = sh["f1"] if sh["f1"] is not None else round(seg_len * fps)
        if f_end - cut < 0.3 * fps:
            log(f"  hook window: close-up at {a:.2f}s shown whole (no head-free hook slot)")
            out.append(whole)
            continue
        log(f"  hook window: close-up at {a:.2f}s shown whole until {hook_end:.2f}s, then punch-in")
        first = {**whole, "f1": cut, "t1": sh["t0"] + (cut - sh["f0"]) / fps}
        first.pop("hold_tail", None)
        rest = {**sh, "f0": cut, "t0": first["t1"]}
        rest.pop("hold_lead", None)
        out += [first, rest]
    return out


def push_cap(cfg: dict, shot: dict) -> float:
    """Push-in cap for a close-up whose chin would otherwise reach the captions (anchor SAFE_TOP)."""
    cap = cfg["render"].get("push_in_max", 1.0)
    fb = shot.get("fb_out")
    if fb is None or fb <= SAFE_TOP:
        return cap
    return max(1.0, min(cap, (FACE_BOTTOM_MAX - SAFE_TOP) / (fb - SAFE_TOP)))


def push_in(cfg: dict, fps: float, anchor: tuple[float, float] | None = None, cap: float | None = None) -> str:
    """Slow push-in from the first frame of a shot: zoom 1.0 + rate*t, capped.

    perspective resamples with sub-pixel precision (cubic), so the zoom does not step/jitter
    like crop/zoompan with integer offsets. `in` is the frame count since the shot started.
    anchor (output px) stays fixed while everything else moves away from it: on close-ups it
    sits at the top safe line so the forehead is not pushed into the top UI zone.
    """
    rc = cfg["render"]
    rate = rc.get("push_in_rate", 0.0)
    cap = rc.get("push_in_max", 1.0) if cap is None else cap
    if rate <= 0 or cap <= 1.001:
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
        ch = shot.get("ch") or even(sh * FACE_CROP_H)
        y0 = shot.get("y0", 0)
        cw = min(sw, even(ch * W / H))
        x = int(min(max(shot["cx"] * sw - cw * FACE_X / W, 0), sw - cw))
        return f"[{inp}]crop={cw}:{ch}:{x}:{y0},scale={W}:{H},setsar=1[{out}]"
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


def chunk_words(words: list[dict], max_words: int, max_sec: float,
                fits=None) -> list[list[dict]]:
    """Caption lines of <= max_words words. fits(chunk) -> bool, when given, is the pixel check:
    a word that would push the line past the safe width starts a new line (a single word always fits)."""
    fits = fits or (lambda ch: True)
    chunks, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or w["s"] - cur[-1]["e"] > 0.6 or w["e"] - cur[0]["s"] > max_sec
                    or not fits(cur + [w])):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    # A chunk on screen for less than MIN_CAPTION_SEC only flashes: merge it into a neighbour,
    # but never past max_words (a merged 6-word line overflowed the safe width, 080536-c2).
    merged: list[list[dict]] = []
    for ch in chunks:
        ok = (bool(merged) and len(merged[-1]) + len(ch) <= max_words and ch[0]["s"] - merged[-1][-1]["e"] <= 0.6
              and fits(merged[-1] + ch))
        if ok and (ch[-1]["e"] - ch[0]["s"] < MIN_CAPTION_SEC or merged[-1][-1]["e"] - merged[-1][0]["s"] < MIN_CAPTION_SEC):
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


def _number_tail(prev: str, cur: str) -> bool:
    """True when cur continues a number split by the tokenizer: '17' ',500' or '17,' '500'."""
    if re.search(r"\d$", prev) and re.fullmatch(r"[.,]\d+[%\w]*", cur):
        return True
    return bool(re.fullmatch(r"\d{1,3}([.,]\d{3})*[.,]", prev) and re.fullmatch(r"\d{3}\b[%\w]*", cur))


def merge_punct(words: list[dict]) -> list[dict]:
    """Glue punctuation-only tokens onto the previous word: '50' '%' -> '50%'."""
    out: list[dict] = []
    for w in words:
        if out and (re.fullmatch(r"[^\w]+", w["w"]) or w["w"].startswith("-") or _number_tail(out[-1]["w"], w["w"])):
            out[-1] = {**out[-1], "w": out[-1]["w"] + w["w"], "e": max(out[-1]["e"], w["e"])}
        else:
            out.append(dict(w))
    return out


_PROFANITY = re.compile(r"(?i)\b(motherf|f)u(ck\w*)|(?<![a-z])((?:bull)?sh)i(t\w*)|\b(b)i(tch\w*)|\b(c)u(nt\w*)")


AUDIO_FADE_IN = 0.04    # s, fade in at the start of every segment (card D: 40 ms on both edges; SNAP_PRE leaves 90 ms of pre-roll)
AUDIO_FADE_OUT = 0.05   # s, fade out at the end of every segment (Whisper word ends run early)


MONTAGE_EDGE_FADE = 0.4   # s, music bed fades in at the clip start and out at the clip end
MONTAGE_CUT_FADE = 0.05   # s, internal cuts (black cuts, tightened gaps): a 0.4 s fade at each one made 7-11 dB dips


def montage_fades(i: int, n: int) -> tuple[float, float]:
    """(fade in, fade out) of segment i of n for a music montage: long only at the clip's own start and end."""
    return (MONTAGE_EDGE_FADE if i == 0 else MONTAGE_CUT_FADE,
            MONTAGE_EDGE_FADE if i == n - 1 else MONTAGE_CUT_FADE)


def audio_chain(idx: int, fi: float, fo: float, dur: float, a_dur: float) -> str:
    """Per-segment audio filter: fades at both cut edges, padded/trimmed to the exact output length."""
    return (f"[{idx}:a:0]afade=t=in:st=0:d={fi},afade=t=out:st={max(dur - fo, 0):.4f}:d={fo},"
            f"apad=whole_dur={a_dur:.6f},asetpts=PTS-STARTPTS,atrim=end={a_dur:.6f}")
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
            elif level_db(audio_fn, s0, 0.03) > SILENCE_DB:
                # Connected speech, no 60 ms pause: mlx word starts run up to 0.2 s late (Cotra c1 cut
                # inside "so" at -20 dB). Take the deepest dip just before, if it is a real one.
                # Scan the same 30 ms window qa.check_segment_starts measures.
                cand = [(level_db(audio_fn, t_, 0.03), -t_) for t_ in (s0 - 0.35 + 0.005 * k for k in range(70))]
                lvl, nt = min(cand)
                if lvl <= -40.0:
                    s0 = -nt
        if level_db(audio_fn, e1 - 0.03, 0.03) > -30.0:
            runs = silence_runs(audio_fn, e1 - 0.02, e1 + 0.8)
            if runs:
                e1 = runs[0][0] + 0.03
        # The end must sit in real quiet (60 ms below -46 dB), not in a word's decaying tail.
        if level_db(audio_fn, e1 - 0.06, 0.06) > END_QUIET_DB:
            stop = e1 + 0.6  # first real quiet wins; the aligner can put the next word inside this tail
            t_ = e1
            while t_ < stop:
                t_ += 0.01
                # 50 ms centered on the cut at <= -46 dB: a stop-consonant closure inside a word only dips to about -43
                # and the last 60 ms stay clear of the word's decay (qa.check_segment_ends measures -35 dB there)
                if level_db(audio_fn, t_ - 0.025, 0.05) <= END_QUIET_DB and level_db(audio_fn, t_ - 0.06, 0.06) <= -40.0:
                    e1 = t_
                    break
    return round(s0, 3), round(e1, 3)


EDGE_CUT = 0.25     # s; camera cuts this close to a segment edge become the edge
MIN_END_SHOT = 0.3  # s; montage sources: a shot shorter than this near the clip end is a leftover flash, not a shot
END_FLASH_WINDOW = 1.0  # s; how far from the end such a leftover is looked for
MONTAGE_BLACK_MIN = 0.3  # s; montage sources: black runs longer than this (title cards) are cut out


def trailing_flash(cuts: list[int], n: int, fps: float) -> int:
    """First frame of the earliest leftover shot (shorter than MIN_END_SHOT) in the last END_FLASH_WINDOW seconds,
    or n when there is none. cuts: first frames of the new shots inside the segment of n frames.
    CoD c2: a 0.17 s blurred shot at 20.13 s, then the first 0.47 s of the next scene fading in from black."""
    bounds = [0] + sorted(c for c in cuts if 0 < c < n) + [n]
    for a, b in zip(bounds, bounds[1:]):
        if a >= n - END_FLASH_WINDOW * fps and (b - a) < MIN_END_SHOT * fps and a > 0:
            return a
    return n


def black_runs(video: Path, start: float, end: float, min_len: float = MONTAGE_BLACK_MIN) -> list[tuple[float, float]]:
    """Source-time ranges inside [start, end] that are black for longer than min_len (blackdetect)."""
    res = subprocess.run(
        [tool("ffmpeg"), "-hide_banner", "-ss", f"{start:.6f}", "-t", f"{end - start:.6f}", "-i", str(video), "-an",
         "-vf", f"blackdetect=d={min_len}:pic_th=0.96:pix_th=0.10", "-f", "null", "-"],
        capture_output=True, text=True, check=True)
    return [(start + float(a), start + float(b))
            for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", res.stderr)]
END_QUIET_DB = -46.0  # last 60 ms of a clip end must be at most this loud
GAP_MAX, GAP_KEEP = 0.3, 0.1   # internal silences longer than GAP_MAX are cut down to 2 x GAP_KEEP
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
# faces out of the top ~220 px, the bottom ~420 px and the right ~220 px (action buttons; text stops at x860).
SAFE_LEFT, SAFE_RIGHT_X = 90, 860
HOOK_FLOOR = 80  # absolute minimum hook size (legacy long hooks only)
SAFE_TOP = 220
FACE_X = 475                              # close-up face centre x (left of the button column)
CAP_X = (SAFE_LEFT + SAFE_RIGHT_X) // 2   # text centre, shifted left of the frame centre


TOP_BLUR: dict[str, list[list[float]]] = {}  # clip id -> [t0, t1, x0, y0, x1, y1] blurred patches (source text in the top strip)
CAP_SHIFT: dict[str, list[list[float]]] = {}  # clip id -> [t0, t1, dy] caption moves (burned-in source text)


HOOK_BAR = 10          # yellow accent bar at the left edge of the hook box, px
HOOK_HAIR = 0.35       # close-up: hair above the face box, as a share of the face height
HOOK_DUO_GAP = 140     # duo: hook centre above the duo caption centre (reviewer: >= 140 centre gap)
H_HALF = 960           # duo seam (output px): the hook stays in the upper panel
HOOK_CAP_GAP = 90      # min px between the hook box bottom and the caption top (reviewer: >= 140 centre gap)


def _fonts(fonts_dir, family):
    if not fonts_dir:
        return None
    from .textfit import font_file
    try:
        return font_file(Path(fonts_dir), family)
    except FileNotFoundError:
        log(f"  WARN font {family!r} not in {fonts_dir}: text widths are estimated")
        return None


def _width(font, text, size, spacing=0.0, outline=0.0, shadow=0.0):
    if font is None:
        return len(text) * size * 0.45 + spacing * len(text) + 2 * outline + shadow
    from .textfit import drawn_width
    return drawn_width(font, text, size, spacing, outline, shadow)


def _cap_h(font, size):
    if font is None:
        return size * 0.7
    from .textfit import cap_height
    return cap_height(font, size)


def _cap_off(font, size):
    if font is None:
        return 0.0
    from .textfit import cap_center_offset
    return cap_center_offset(font, size)


def build_ass(cfg: dict, clip: dict, segments: list[tuple[float, float, float]], layout: str,
              shot_spans: list[tuple] | None = None, fonts_dir: Path | None = None) -> str:
    """ASS captions for the output timeline; segments map source ranges to output offsets (cold open + main).

    shot_spans (output start, end, kind[, geometry]) switch the caption position per camera shot in the
    shots layout: lower on close-ups (off the face), at the seam on stacked duo shots. A caption
    line that crosses a cut is split there so each part sits at its shot's position. geometry
    (close-ups) = {"ft", "fb": face box top/bottom in output px, "cap": push-in cap, "t0": shot start}
    places the hook above the hair or under the chin, never on the head.

    fonts_dir: where the caption/hook fonts live; widths are then measured from the real font
    (captions <= captions.max_width px incl. outline and shadow, hook shrinks to fit, min hook.min_size).
    """
    cc, hc, rc = cfg["captions"], cfg["hook"], cfg["render"]
    W, H = rc["width"], rc["height"]
    y = cc[f"y_{layout}"]
    shot_y = {"face": cc.get("y_shots_face", y), "face_tight": cc.get("y_shots_face", y), "duo": cc.get("y_shots_duo", y), "fit": cc.get("y_fit", y)}
    cap_font = _fonts(fonts_dir, cc["font"])
    size, spacing = cc["size"], cc.get("spacing", 0)
    outline, shadow = cc["outline"], cc.get("shadow", 2)
    bold = -1 if cc.get("bold", True) else 0

    MIN_PART = 0.12  # a caption piece shorter than this (< 4 frames) reads as a jump/duplicate

    def placed(t0: float, t1: float) -> list[tuple[float, float, int]]:
        parts = placed_base(t0, t1)
        if len(parts) > 1:
            # A sliver across a shot cut (usually the word's silent tail or a lead-in) is dropped,
            # not shown in a second place: the line never repeats at two positions.
            long_ = [p for p in parts if p[1] - p[0] >= MIN_PART]
            parts = long_ or [max(parts, key=lambda p: p[1] - p[0])]
        shifts = CAP_SHIFT.get(clip.get("id"), [])
        if not shifts:
            return parts
        out = []
        for lo, hi, yy in parts:
            cuts = sorted({lo, hi} | {c for a, b, _ in shifts for c in (a, b) if lo < c < hi})
            pieces = []
            for u, v in zip(cuts, cuts[1:]):
                dy = next((d for a, b, d in shifts if a <= (u + v) / 2 <= b), 0)
                pieces.append([u, v, int(yy + dy)])
            # Shift spans come from a 0.5 s grid, so their edges miss the cut by a few frames:
            # a short piece inside one shot joins its longer neighbour instead of jumping for 2 frames.
            merged = True
            while merged and len(pieces) > 1:
                merged = False
                for i, (u, v, _y) in enumerate(pieces):
                    if v - u < MIN_PART:
                        j = i - 1 if i == len(pieces) - 1 or (i > 0 and pieces[i - 1][1] - pieces[i - 1][0]
                                                               >= pieces[i + 1][1] - pieces[i + 1][0]) else i + 1
                        pieces[j][0], pieces[j][1] = min(pieces[j][0], u), max(pieces[j][1], v)
                        del pieces[i]
                        merged = True
                        break
            out += [tuple(p) for p in pieces]
        return out

    def placed_base(t0: float, t1: float) -> list[tuple[float, float, int]]:
        if not shot_spans:
            return [(t0, t1, y)]
        parts = []
        for a, b, kind, *_ in shot_spans:
            lo, hi = max(t0, a), min(t1, b)
            if hi - lo > 0.005:
                parts.append((lo, hi, shot_y.get(kind, y)))
        return parts or [(t0, t1, y)]

    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "WrapStyle: 2", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Cap,{cc['font']},{size},{cc['color']},&H000000FF,&H00000000,{cc.get('back', '&H64000000')},"
        f"{bold},0,0,0,100,100,{spacing},0,1,{outline},{shadow},5,{SAFE_LEFT},{W - SAFE_RIGHT_X},0,1",
        f"Style: HookBox,Arial,20,&H00111111,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1",
        f"Style: HookText,{hc.get('font', cc['font'])},{hc['size']},{hc.get('color', '&H00FFFFFF')},&H000000FF,"
        f"&H00000000,&H00000000,{-1 if hc.get('bold', False) else 0},0,0,0,100,100,{hc.get('spacing', 0)},0,1,0,0,4,0,0,0,1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    if hc["enabled"] and clip.get("hook"):
        lines += hook_events(cfg, clip, segments, shot_spans, fonts_dir, cap_font)

    max_w = cc.get("max_width", SAFE_RIGHT_X - SAFE_LEFT - 70)
    lead = cc.get("onset_lead", 0.03)
    sync = cc.get("sync_shift", 0.07)  # whisper on the finished audio: captions came median +69 ms late (reviewer #274)
    fade = cc.get("fade_ms", 70)
    blur = cc.get("blur", 1)
    hl_out = cc.get("highlight_outline")

    def shown(w: dict) -> str:
        t = censor(fix_names(w["w"], clip.get("name_fixes", {})))
        return ass_escape(t.upper() if cc["uppercase"] else t)

    def fits(ch: list[dict]) -> bool:
        return _width(cap_font, " ".join(shown(w) for w in ch), size, spacing, outline, shadow) <= max_w

    for seg_k, (seg_s, seg_e, off) in enumerate(segments):
        # A word is captioned only if it starts inside the segment and its middle is inside too:
        # no word from before the cut (cold open began after "To this day,"), no half-heard word at the end.
        # w["ov"][segment] (captionfix.retime_words) retimes a word replayed in two segments per occurrence.
        words = [{**w, "s": w.get("ov", {}).get(seg_k, (w["s"], w["e"]))[0] - sync,
                  "e": w.get("ov", {}).get(seg_k, (w["s"], w["e"]))[1] - sync}
                 for w in caption_words(merge_punct(clip.get("words", [])))
                 if w["s"] >= seg_s - 0.02 and (w["s"] + w["e"]) / 2 <= seg_e]
        chunks = chunk_words(words, cc["max_words"], cc["max_chunk_sec"], fits if cap_font else None)
        for ci, chunk in enumerate(chunks):
            # The line appears `lead` s before the first word's onset, so the fade-in is done when
            # the sound starts; the previous line gives way at the same moment (never two lines).
            next_start = chunks[ci + 1][0]["s"] - lead if ci + 1 < len(chunks) else None
            texts = [shown(w) for w in chunk]
            pop = cc.get("pop", 1.0)
            pop_tag = f"{{\\fscx{int(pop*100)}\\fscy{int(pop*100)}\\t(0,80,\\fscx100\\fscy100)}}" if pop > 1.0 else ""
            chunk_t0 = max(chunk[0]["s"] - lead, seg_s)
            chunk_t1 = min(chunk[-1]["e"] + 0.12, next_start if next_start is not None else 1e9, seg_e)
            first = True
            evs = []  # [t0, t1, highlighted word index]
            for i, w in enumerate(chunk):
                t0 = w["s"] if i else chunk_t0
                t1 = chunk[i + 1]["s"] if i + 1 < len(chunk) else chunk[-1]["e"] + 0.12  # a word's voiced tail outlasts its aligned end
                if next_start is not None:
                    t1 = min(t1, next_start)  # two caption lines must never be on screen together
                t0, t1 = max(t0, seg_s), min(t1, seg_e)
                if t1 > t0:
                    evs.append([t0, t1, i])
            evs = balance_highlights(evs, MIN_WORD_SHOW)
            for t0, t1, i in evs:
                if hl_out:
                    on, off_ = f"{{\\3c{hl_out}\\c{cc['highlight']}}}", f"{{\\3c&H00000000\\c{cc['color']}}}"
                else:
                    on, off_ = f"{{\\c{cc['highlight']}}}", f"{{\\c{cc['color']}}}"
                parts = [f"{on}{t}{off_}" if j == i else t for j, t in enumerate(texts)]
                # No fade on a line shorter than 0.2 s or a first word shown < 0.2 s: it would spend most of its frames half-transparent.
                fade_tag = f"\\fad({fade},0)" if fade and chunk_t1 - chunk_t0 >= 0.2 and t1 - t0 >= 0.2 else ""
                for a, b, yy in placed(t0 - seg_s + off, t1 - seg_s + off):
                    if b - a < 0.05:
                        continue  # a sliver flashes one frame at a cut, lit during silence
                    tags = f"\\pos({CAP_X},{yy})" + (f"\\blur{blur}" if blur else "") + (fade_tag if first else "")
                    first = False
                    lines.append(
                        f"Dialogue: 0,{ass_time(a)},{ass_time(b)},Cap,,0,0,0,,"
                        f"{{{tags}}}{pop_tag}{' '.join(parts)}"
                    )
    return "\n".join(lines) + "\n"


def hook_events(cfg: dict, clip: dict, segments, shot_spans, fonts_dir, cap_font) -> list[str]:
    """The hook: one line of text in a dark box with a yellow bar, left-aligned at the safe edge.

    The font shrinks to fit the safe width, never below hook.min_size (rank.py keeps hooks short
    enough). On close-ups the box goes above the hair or under the chin with room to the captions;
    if neither fits, that shot gets no hook (reviewer: the hook never sits on the head).
    """
    cc, hc, rc = cfg["captions"], cfg["hook"], cfg["render"]
    font = _fonts(fonts_dir, hc.get("font", cc["font"]))
    text = ass_escape(clip["hook"]).upper()
    pad = hc.get("pad", 18)
    avail = SAFE_RIGHT_X - SAFE_LEFT - HOOK_BAR - 2 * pad
    size = hc["size"]
    while size > hc.get("min_size", 80) and _width(font, text, size, hc.get("spacing", 0)) > avail:
        size -= 2
    tw = _width(font, text, size, hc.get("spacing", 0))
    while size > HOOK_FLOOR and tw > avail:
        # Hooks written before rank.py's length cap: shrink past min_size rather than leave the safe zone.
        size -= 2
        tw = _width(font, text, size, hc.get("spacing", 0))
    if tw > avail:
        log(f"  WARN hook {text!r} is {tw:.0f} px at {size} px, wider than {avail} px")
    ch = _cap_h(font, size)
    bh = ch + 2 * pad
    x0, x1 = SAFE_LEFT, SAFE_LEFT + HOOK_BAR + pad + min(tw, avail) + pad
    cap_top = cc.get("y_shots_face", 1195) - _cap_h(cap_font, cc["size"]) / 2 - cc["outline"]

    yc = hc["y"]  # ONE fixed position for the whole hook window, in every layout (reviewer #274: no jumping)
    y0, y1 = yc - bh / 2, yc + bh / 2
    cold = clip.get("cold_len", segments[0][1] - segments[0][0] if len(segments) > 1 else 0.0)
    t_end = cold + hc["seconds"]
    # Close-up with no room above the hair for the box: the hook is shown only HOOK_SHORT_S, not for the full window.
    for a, b, kind, *rest in (shot_spans or []):
        geo = rest[0] if rest else None
        if kind.startswith("face") and geo and max(a, 0.0) < t_end and b > cold and y1 > geo["ft"] - (HOOK_HAIR * (geo["fb"] - geo["ft"]) + 10):
            t_end = min(t_end, cold + HOOK_SHORT_S)
            log(f"  hook: head too high for the box at y{yc:.0f}, hook limited to {HOOK_SHORT_S:.1f} s")
            break
    off = _cap_off(font, size)
    t = f"{ass_time(0.0)},{ass_time(t_end)}"
    rect = lambda u0, u1: f"m {u0:.0f} {y0:.0f} l {u1:.0f} {y0:.0f} l {u1:.0f} {y1:.0f} l {u0:.0f} {y1:.0f}"
    out = [
        f"Dialogue: 1,{t},HookBox,,0,0,0,,{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\c{hc.get('fill', '&H00111111')}"
        f"\\1a{hc.get('fill_alpha', '&H26&')}}}{rect(x0, x1)}",
        f"Dialogue: 2,{t},HookBox,,0,0,0,,{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\c{hc['box']}}}{rect(x0, x0 + HOOK_BAR)}",
        f"Dialogue: 3,{t},HookText,,0,0,0,,{{\\an4\\pos({x0 + HOOK_BAR + pad},{yc - off:.0f})\\fs{size}}}{text}",
    ]
    shown = t_end
    if shown < 1.0:
        log(f"  WARN hook shown only {shown:.2f} s (no head-free slot on the close-ups)")
    return out


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


def inputs_idx(inputs: list) -> list:
    """Indices of the -i inputs in an ffmpeg arg list."""
    return [i for i, a in enumerate(inputs) if a == "-i"]


def logo_box(src: Path, width: int) -> Path:
    """Logo on a white padded box (readable on any background), cached next to the source logo."""
    out = src.with_name(f"{src.stem}_box{width}.png")
    if not out.exists():
        run([tool("ffmpeg"), "-v", "error", "-y", "-i", src, "-vf",
             f"scale={width}:-2,format=rgb24,pad=iw+40:ih+40:20:20:white", "-frames:v", "1", out])
    return out


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
    import os
    only = [x for x in os.environ.get("CLIPPER_ONLY", "").split(",") if x]
    for clip in clips:
        if only and clip["id"] not in only:
            continue
        clip["music"] = music_ratio(jd, clip["start"], clip["end"], mc["threshold"])
        if clip["music"] is not None and clip["music"] > mc["max_ratio"]:
            log(f"skipping {clip['id']}: music in {clip['music']:.0%} of the clip (Content ID risk)")
            continue
        cdir = jd / "clips" / clip["id"]
        cdir.mkdir(parents=True, exist_ok=True)
        out = cdir / "short.mp4"
        from .transcribe import load_audio
        audio_fn = lambda t0, d: load_audio(video, t0, d)  # noqa: E731
        # Source-level fixes first, the clip's own win (MacArthur is right for the general, not for McArthur Wheeler).
        clip = {**clip, "name_fixes": {**source.get("name_fixes", {}), **clip.get("name_fixes", {})}}
        raw_words = merge_punct(clip.get("words", []))
        words = raw_words
        if words and not clip.get("words_whisper"):
            # Old clips without forced alignment: fall back to onset pinning.
            words = refine_words(words, audio_fn, min(w["s"] for w in words) - 0.5, max(w["e"] for w in words) + 0.5)
            clip = {**clip, "words": words}
        if words:
            from .transcribe import snap_to_dips
            lo_ = min(w["s"] for w in words) - 0.5
            words = snap_to_dips(words, load_audio(video, lo_, max(w["e"] for w in words) + 0.5 - lo_), lo_)
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
        if source.get("montage"):  # title-card windows are dropped here (rank.py does the same for speech sources)
            from .blackwin import black_fraction, black_max
            black = black_fraction(video, main[0], main[1])
            log(f"  black_ratio {black:.3f} for {clip['id']}, limit {black_max(cfg):.2f}")
            if black > black_max(cfg):
                log(f"  DROP {clip['id']}: {black:.0%} of the window is black (title cards), limit {black_max(cfg):.0%}")
                continue
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
            if source.get("montage"):  # music trailers have no speech to protect: drop a leftover flash at the end
                tail = min(tail, trailing_flash(cuts, n, fps))

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
        # Dead air inside the main segment (a breath cut out of the source leaves digital silence
        # mid-sentence): drop the middle of any silence over GAP_MAX, keep GAP_KEEP each side.
        ms_, me_, moff_ = segments[-1]
        fps = info["fps"]
        r = round(fps / rc["fps"]) if rc["fps"] and abs(fps / rc["fps"] - round(fps / rc["fps"])) < 0.01 else 1
        n_main, cuts_f = nframes[-1], []
        # clip["cut_gaps"]: [start, end] source ranges of sound that is not speech (breath, noise), cut like silence.
        wgaps = [tuple(g) for g in clip.get("cut_gaps", [])]
        if source.get("montage"):  # title cards / fades through black are cut out (the loop below keeps GAP_KEEP each side)
            blk = black_runs(video, ms_, me_)
            wgaps += [(a - GAP_KEEP, b + GAP_KEEP) for a, b in blk]
            if blk:
                log(f"  black runs cut: {', '.join(f'{a - ms_:.2f}-{b - ms_:.2f}' for a, b in blk)} (segment-relative s)")
        for ga, gb in sorted(wgaps + ([] if clip.get("no_tighten") else silence_runs(audio_fn, ms_, me_, min_len=GAP_MAX))):
            if ga - ms_ < 0.25 or me_ - gb < 0.25:
                continue
            f0 = -(-int((ga + GAP_KEEP - ms_) * fps) // r) * r
            f1 = int((gb - GAP_KEEP - ms_) * fps) // r * r
            # Never cut into a word: pull the cut edges back from any word they touch.
            for w in words:
                if ms_ + f0 / fps < w["s"] < ms_ + f1 / fps:
                    f1 = int((w["s"] - 0.04 - ms_) * fps) // r * r
                if ms_ + f0 / fps < w["e"] < ms_ + f1 / fps:
                    f0 = -(-int((w["e"] + 0.04 - ms_) * fps) // r) * r
            if f1 - f0 >= r and (not cuts_f or f0 > cuts_f[-1][1]):
                cuts_f.append((f0, f1))
        if cuts_f:
            subs, pos = [], 0
            for f0, f1 in cuts_f + [(n_main, n_main)]:
                if f0 > pos:
                    subs.append((pos, f0))
                pos = f1
            segments, nframes, o = segments[:-1], nframes[:-1], moff_
            for a_, b_ in subs:
                segments.append((ms_ + a_ / fps, ms_ + b_ / fps, o))
                nframes.append(b_ - a_)
                o += (b_ - a_) / fps
            log(f"  tightened {len(cuts_f)} internal gaps: {sum(f1 - f0 for f0, f1 in cuts_f) / fps:.2f}s removed")
        clip = {**clip, "cold_len": segments[0][1] - segments[0][0] if len(raw) > 1 else 0.0}
        layout, cx, cam, track = pick_layout(cfg, source, video, info, main[0], main[1])
        if track:
            log(f"  tracking: {len(track)} face samples")
        if source.get("identity") and layout in ("center", "auto"):
            st = face_boxes.last_identity
            frac = st["locked"] / max(st["samples"], 1)
            # Source-based error (Codee #184): the streamer is missing or someone else shares the frame.
            if frac < 0.6 or st["other_faces"] > st["locked"] * 0.5:
                log(f"  DROP {clip['id']}: identity '{source['identity']}' present in {frac:.0%} of samples, "
                    f"{st['other_faces']} other faces vs {st['locked']} locked")
                continue
        seg_shots, shot_spans = [], None
        if layout == "shots":
            shot_spans = []
            hook_end = clip["cold_len"] + cfg["hook"]["seconds"]
            for s_, e_, off_ in segments:
                plan = hook_window_fit(cfg, info, plan_shots(video, info, s_, e_), off_, e_ - s_, hook_end)
                seg_shots.append(plan)
                for sh in plan:
                    # Output times from the frame-exact source boundaries.
                    # Held edge frames belong to the first/last shot's layout.
                    a = off_ + (0.0 if sh is plan[0] else sh["f0"] / info["fps"])
                    b = off_ + (e_ - s_ if sh is plan[-1] else sh["f1"] / info["fps"])
                    geo = ({"ft": sh["ft_out"], "fb": sh["fb_out"], "cap": push_cap(cfg, sh),
                            "t0": out_frame_time(a, rc["fps"])} if sh["kind"] == "face" and "ft_out" in sh else None)
                    if sh["kind"] == "duo":
                        # Upper panel: where the upper speaker's chin lands (output px), for the hook slot.
                        ph = rc["height"] // 2
                        (_lw, lh, _lx, ly), _r = duo_panels(sh, info, rc["width"] / ph)
                        fy, fh = sh["left"][1], sh["left"][3]
                        geo = {"fb_top": ((fy + fh) * info["height"] - ly) * ph / lh}
                    shot_spans.append((out_frame_time(a, rc["fps"]), out_frame_time(b, rc["fps"]),
                                       "face_tight" if sh.get("tight") else sh["kind"], geo))
        lg = source.get("logo") or {}
        cap_cfg = cfg if "caption_y" not in lg else {**cfg, "captions": {**cfg["captions"], "y_fit": lg["caption_y"], "y_center": lg["caption_y"]}}
        # Segment audio comes from one sample-exact WAV: input seek on the video file loses audio on some sources.
        from .segaudio import extract_range
        seg_wav = cdir / "segments_audio.wav"
        # shots layout: the WAV audio inputs hang ffmpeg's filter graph (solved-091333-c2, 15+ min at 100% CPU,
        # pre-G code 3 min); that layout keeps the audio of the video input (input seek) until this is understood.
        wav_audio = layout != "shots"
        wav_t0 = extract_range(video, min(sg[0] for sg in segments), max(sg[1] for sg in segments), seg_wav) if wav_audio else 0.0
        if wav_audio and clip.get("words") and cfg["captions"].get("retime_from_audio", True):
            # Forced alignment can sit 0.3-0.6 s off after fillers: move grossly misplaced words to the heard start.
            try:
                from .captionfix import heard_in_output, retime_words
                cw = caption_words(merge_punct(clip["words"]))
                heard = heard_in_output(seg_wav, wav_t0, segments)
                fixed, moved = retime_words(cw, segments, heard, bias=cfg["captions"].get("sync_shift", 0.07))
                import json as _json
                (cdir / "retime.json").write_text(_json.dumps(
                    {"segments": segments, "caption": [(w["w"], w["s"]) for w in cw], "heard": heard, "moved": moved}))
                if moved:
                    log(f"{clip['id']}: retimed {len(moved)} caption words from the finished audio: {', '.join(moved[:12])}")
                    clip = {**clip, "words": fixed}
            except Exception as exc:  # noqa: BLE001 - the check must never block a render
                log(f"{clip['id']}: caption retiming skipped ({exc})")
        (cdir / "captions.ass").write_text(build_ass(cap_cfg, clip, segments, layout, shot_spans, fonts))

        inputs, graph, concat = [], [], ""
        # -t half a frame short of n frames: exactly n video frames; audio is padded to n/fps.
        for i, (s, e, _) in enumerate(segments):
            inputs += ["-ss", f"{s:.6f}", "-t", f"{(nframes[i] - 0.5) / info['fps']:.6f}", "-i", video.resolve()]
        if wav_audio:
            for i, (s, e, _) in enumerate(segments):  # inputs len(segments)+i: the segment audio
                inputs += ["-ss", f"{s - wav_t0:.6f}", "-t", f"{(nframes[i] - 0.5) / info['fps']:.6f}", "-i", seg_wav.resolve()]
        for i, (s, e, _) in enumerate(segments):
            fps = info["fps"]
            if layout == "shots":
                # One static layout per camera shot; trim on a single input keeps every frame exactly once.
                shots = seg_shots[i]
                log(f"  segment {i}: " + ", ".join(f"{x['t0']:.2f}-{x['t1']:.2f} {x['kind']}" for x in shots))
                k = len(shots)
                graph.append(f"[{i}:v]setpts=PTS-STARTPTS,split={k}" + "".join(f"[s{i}_{j}]" for j in range(k)))
                for j, sh in enumerate(shots):
                    anchor = (FACE_X, SAFE_TOP) if sh["kind"] == "face" else (FACE_X, rc["height"] / 2)
                    zoom = push_in(cfg, info["fps"], anchor, push_cap(cfg, sh) if sh["kind"] == "face" else None)
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
                    seg_track, _ = face_boxes(video, s, e, timed=True, identity=source.get("identity"))
                # Cap at the exact output frame count: concat plus -r otherwise adds a frame per segment
                # on a 60 fps source (video longer than audio).
                n_out = round(nframes[i] * float(rc["fps"]) / info["fps"]) if rc["fps"] else nframes[i]
                graph.append(f"[{i}:v]setpts=PTS-STARTPTS,fps={rc['fps']},trim=end_frame={n_out}[z{i}]")
                graph.append(layout_filter(cfg, info, layout, cx, cam, f"z{i}", f"l{i}",
                                           track=seg_track, seg_start=s,
                                           source=source if (source.get("campaign_url") or source.get("montage")) else None))
            # Short audio fades at every cut so word-boundary splices do not click.
            fi, fo, dur = AUDIO_FADE_IN, AUDIO_FADE_OUT, (nframes[i] - 0.5) / fps
            # Audio exactly as long as the output video frames (n_out / output fps): any difference lets -r 30 duplicate frames.
            a_dur = n_out / float(rc["fps"]) if layout != "shots" and rc["fps"] else nframes[i] / fps
            if source.get("montage"):
                fi, fo = montage_fades(i, len(segments))  # music montage: fade the bed at the clip edges only
            graph.append(audio_chain(len(segments) + i if wav_audio else i, fi, fo, dur, a_dur) + f"[a{i}]")
            concat += f"[l{i}][a{i}]"
        # loudnorm alone let true peaks reach -0.7 dBFS; limiter at -3 dBFS leaves room for AAC overshoot (~1.5 dB).
        audio = ",loudnorm=I=-14:TP=-1.5:LRA=11,aresample=192000,alimiter=limit=0.708:level=false:attack=1:release=50,aresample=48000" if rc["loudnorm"] else ""
        graph.append(f"{concat}concat=n={len(segments)}:v=1:a=1[cv][ca0]")
        from .memes import pick_effects, meme_filter_chain
        meme_fx = pick_effects(cfg, clip)
        W, H = rc["width"], rc["height"]
        if meme_fx:
            # Payoff moment inside the main segment, in output-timeline seconds.
            ps = clip.get("payoff_start")
            ms, me, moff = next((sg for sg in segments[::-1] if ps is not None and sg[0] <= ps < sg[1]), segments[-1])
            window = None
            if ps is not None and ms <= ps < me:
                pe = min(clip.get("payoff_end") or ps + 1.5, me)
                window = (moff + ps - ms, moff + pe - ms)
            graph.append(meme_filter_chain(meme_fx, "cv", "cv_fx", W, H, window))
            cap_in = "cv_fx"
            log(f"  meme effects: {meme_fx} at {window}")
        else:
            cap_in = "cv"
        # Caption-free low-res copy for the burned-in-text QA (same frames, no subtitles, no logo).
        for k, (t0_, t1_, bx0, by0, bx1, by1) in enumerate(TOP_BLUR.get(clip["id"], [])):
            bw_, bh_ = int(bx1 - bx0), int(by1 - by0)
            # Feathered patch: alpha ramps over 24 px on the sides that do not touch the frame edge.
            terms = ["9999"]
            if bx0 > 0:
                terms.append("(X+1)")
            if bx1 < 1080:
                terms.append(f"({bw_}-X)")
            if by0 > 0:
                terms.append("(Y+1)")
            terms.append(f"({bh_}-Y)")
            dist = terms[0]
            for t_ in terms[1:]:
                dist = f"min({dist},{t_})"
            alpha = f"255*min(1,{dist}/56)"
            graph.append(f"[{cap_in}]split[tb{k}a][tb{k}b];[tb{k}b]crop={bw_}:{bh_}:{int(bx0)}:{int(by0)},gblur=sigma=40:steps=4,"
                         f"format=yuva420p,geq=lum='lum(X,Y)':cb='cb(X,Y)':cr='cr(X,Y)':a='{alpha}'[tb{k}c];"
                         f"[tb{k}a][tb{k}c]overlay={int(bx0)}:{int(by0)}:enable='between(t,{t0_:.2f},{t1_:.2f})'[tb{k}]")
            cap_in = f"tb{k}"
        # round=up: with the default round=near the fps filter keeps the LAST input frame of each
        # 0.5 s slot, so nocap frame i showed t=i/2+0.23 and QA placed caption shifts too early.
        graph.append(f"[{cap_in}]split[cap_a][cap_b];[cap_b]fps=2:round=up,scale=540:960[nc]")
        cap_in = "cap_a"
        logo = source.get("logo")
        if logo:
            # Campaign-mandated logo (Alliance): white padded box, centred under the video panel, whole clip.
            box = logo_box(Path(logo["file"]), int(logo.get("width", 340)))
            inputs += ["-loop", "1", "-i", box.resolve()]
            graph.append(f"[{cap_in}]subtitles=captions.ass:fontsdir={fonts}[cs]")
            graph.append(f"[cs][{len(inputs_idx(inputs)) - 1}:v]overlay=(W-w)/2:{int(logo.get('y', 1360))}:shortest=1[v]")
        else:
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
        nocap = ["-map", "[nc]", "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p",
                 "nocap.mp4"]
        (cdir / "graph.txt").write_text(";\n".join(graph))
        try:
            run(cmd + ["-movflags", "+faststart", out.name] + nocap, cwd=cdir)
        finally:
            seg_wav.unlink(missing_ok=True)  # temp segment audio, never part of the clip folder's output
        check_output(out)
        # Preview frame for the review step (taken after the hook disappears).
        run([tool("ffmpeg"), "-v", "error", "-y", "-ss", min(3.0, (clip["end"] - clip["start"]) / 2),
             "-i", out.name, "-frames:v", "1", "-q:v", "3", "preview.jpg"], cwd=cdir)
        write_json(cdir / "clip.json", clip)
        from .qa import plan_caption_shift, plan_top_blur, run_all
        if clip["id"] not in TOP_BLUR:
            blurs = plan_top_blur(cdir)
            TOP_BLUR[clip["id"]] = blurs
            if blurs:
                log(f"  burned-in source text in the top strip, blurring: {blurs}")
                prev = os.environ.get("CLIPPER_ONLY")
                os.environ["CLIPPER_ONLY"] = clip["id"]
                try:
                    outputs += render(cfg, job_id)
                finally:
                    if prev is None:
                        os.environ.pop("CLIPPER_ONLY", None)
                    else:
                        os.environ["CLIPPER_ONLY"] = prev
                continue
        if clip["id"] not in CAP_SHIFT:
            shifts = plan_caption_shift(cdir, fonts)
            if shifts:
                # Source lower-third/inset under the captions: move them clear and render this clip again.
                CAP_SHIFT[clip["id"]] = shifts
                log(f"  burned-in source text under captions, moving captions: {shifts}")
                prev = os.environ.get("CLIPPER_ONLY")
                os.environ["CLIPPER_ONLY"] = clip["id"]
                try:
                    outputs += render(cfg, job_id)
                finally:
                    if prev is None:
                        os.environ.pop("CLIPPER_ONLY", None)
                    else:
                        os.environ["CLIPPER_ONLY"] = prev
                continue
        problems = run_all(cfg, cdir, audio_fn, [(e_, off_ + e_ - s_) for s_, e_, off_ in segments],
                           [(s_, off_) for s_, e_, off_ in segments],
                           montage=bool(source.get("montage")),
                           logo=((logo_box(Path(lg["file"]), int(lg.get("width", 340))), int(lg.get("y", 1360))) if lg.get("file") else None))
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
