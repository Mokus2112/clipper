"""Step 5: render each selected clip as a 1080x1920 Short with layout, word-by-word captions and a hook."""

import re
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


def layout_filter(cfg: dict, info: dict, layout: str, cx: float | None, facecam: list[float] | None,
                  inp: str, out: str, track: list | None = None, seg_start: float = 0.0) -> str:
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
    cw = min(sw, even(sh * W / H))
    # Dynamic tracking: EMA-smoothed crop expression if we have timed face data.
    rc = cfg["render"]
    if track and rc.get("track_face", False):
        smoothed = _ema_smooth(track, rc.get("track_alpha", 0.15), sw, cw)
        x_expr = _track_crop_expr(smoothed, seg_start)
    else:
        center = (cx if cx is not None else 0.5) * sw
        x_expr = str(int(min(max(center - cw / 2, 0), sw - cw)))
    return f"[{inp}]crop={cw}:{sh}:{x_expr}:0,scale={W}:{H},setsar=1[{out}]"


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
    return chunks


_PROFANITY = re.compile(r"(?i)\b(motherf|f)u(ck\w*)|(?<![a-z])((?:bull)?sh)i(t\w*)|\b(b)i(tch\w*)|\b(c)u(nt\w*)")


AUDIO_FADE = 0.018  # s, fade in/out on every segment before concat


def snap_to_words(words: list[dict], start: float, end: float, sentence_end: bool = False,
                  max_trim: float | None = None) -> tuple[float, float]:
    """Snap a segment to word boundaries so a cut never lands mid-word.

    Start: beginning of the first word starting at or after start-0.05.
    End: end of the last word ending at or before end+0.3; with sentence_end, if one of the
    words in the segment ends a sentence (. ! ?), the segment closes at the last such word
    (with max_trim: only if that cuts at most max_trim seconds, so the main segment does
    not end on a dangling "So").
    Falls back to the given times when no word fits.
    """
    first = next((w for w in words if w["s"] >= start - 0.05), None)
    if first is None:
        return start, end
    inside = [w for w in words if w["s"] >= first["s"] and w["e"] <= end + 0.3]
    if not inside:
        return start, end
    last = inside[-1]
    if sentence_end:
        enders = [w for w in inside if w["w"].rstrip("\"'”’)").endswith((".", "!", "?"))
                  and (max_trim is None or last["e"] - w["e"] <= max_trim)]
        if enders:
            last = enders[-1]
    return first["s"], last["e"]


def censor(word: str) -> str:
    """Mask strong profanity in burned-in captions (audio is untouched): FUCK -> F*CK."""
    def mask(m: re.Match) -> str:
        g = [x for x in m.groups() if x is not None]
        return g[0] + "*" + g[1]
    return _PROFANITY.sub(mask, word)


def build_ass(cfg: dict, clip: dict, segments: list[tuple[float, float, float]], layout: str) -> str:
    """ASS captions for the output timeline; segments map source ranges to output offsets (cold open + main)."""
    cc, hc, rc = cfg["captions"], cfg["hook"], cfg["render"]
    W, H = rc["width"], rc["height"]
    y = cc[f"y_{layout}"]

    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "WrapStyle: 0", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Cap,{cc['font']},{cc['size']},{cc['color']},&H000000FF,&H00000000,&H64000000,"
        f"-1,0,0,0,100,100,0,0,1,{cc['outline']},2,5,90,160,0,1",
        f"Style: Hook,{cc['font']},{hc['size']},&H00000000,&H000000FF,{hc['box']},{hc['box']},"
        f"-1,0,0,0,100,100,0,0,3,18,0,5,90,90,0,1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    if hc["enabled"] and clip.get("hook"):
        # The hook stays up through the cold open and the first seconds of the setup.
        cold = segments[0][1] - segments[0][0] if len(segments) > 1 else 0.0
        lines.append(
            f"Dialogue: 1,{ass_time(0)},{ass_time(cold + hc['seconds'])},Hook,,0,0,0,,"
            f"{{\\pos({W // 2},{hc['y']})}}{ass_escape(clip['hook'])}"
        )
    for seg_s, seg_e, off in segments:
        words = [w for w in clip.get("words", []) if w["s"] >= seg_s - 0.05 and w["e"] <= seg_e + 0.05]
        for chunk in chunk_words(words, cc["max_words"], cc["max_chunk_sec"]):
            texts = [ass_escape(censor(w["w"].upper() if cc["uppercase"] else w["w"])) for w in chunk]
            pop = cc.get("pop", 1.0)
            pop_tag = f"{{\\fscx{int(pop*100)}\\fscy{int(pop*100)}\\t(0,80,\\fscx100\\fscy100)}}" if pop > 1.0 else ""
            for i, w in enumerate(chunk):
                t0 = w["s"] if i else chunk[0]["s"]
                t1 = chunk[i + 1]["s"] if i + 1 < len(chunk) else chunk[-1]["e"]
                t0, t1 = max(t0, seg_s), min(t1, seg_e)
                if t1 <= t0:
                    continue
                parts = [f"{{\\c{cc['highlight']}}}{t}{{\\c{cc['color']}}}" if j == i else t
                         for j, t in enumerate(texts)]
                lines.append(
                    f"Dialogue: 0,{ass_time(t0 - seg_s + off)},{ass_time(t1 - seg_s + off)},Cap,,0,0,0,,"
                    f"{{\\pos({W // 2},{y})}}{pop_tag}{' '.join(parts)}"
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
        main = snap_to_words(clip.get("words", []), clip["start"], clip["end"], sentence_end=True, max_trim=1.5)
        cold = (clip.get("payoff_start"), clip.get("payoff_end"))
        if cold[0] is not None:
            cold = snap_to_words(clip.get("words", []), cold[0], cold[1], sentence_end=True)
        segments = []
        if rc["cold_open"] and cold[0] is not None and cold[1] - cold[0] >= 0.5:
            segments.append((cold[0], cold[1], 0.0))
        segments.append((main[0], main[1], sum(e - s for s, e, _ in segments)))
        layout, cx, cam, track = pick_layout(cfg, source, video, info, main[0], main[1])
        if track:
            log(f"  tracking: {len(track)} face samples")
        (cdir / "captions.ass").write_text(build_ass(cfg, clip, segments, layout))

        inputs, graph, concat = [], [], ""
        for i, (s, e, _) in enumerate(segments):
            inputs += ["-ss", f"{s:.2f}", "-t", f"{e - s:.2f}", "-i", video.resolve()]
            graph.append(layout_filter(cfg, info, layout, cx, cam, f"{i}:v", f"l{i}",
                                       track=track, seg_start=s))
            # Short audio fades at every cut so word-boundary splices do not click.
            fd, dur = AUDIO_FADE, e - s
            graph.append(f"[{i}:a:0]afade=t=in:st=0:d={fd},afade=t=out:st={max(dur - fd, 0):.3f}:d={fd}[a{i}]")
            concat += f"[l{i}][a{i}]"
        audio = ",loudnorm=I=-14:TP=-1.5:LRA=11" if rc["loudnorm"] else ""
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
        graph.append(f"[ca0]anull{audio}[a]")
        cmd = [
            tool("ffmpeg"), "-v", "error", "-y", *inputs, "-filter_complex", ";".join(graph),
            "-map", "[v]", "-map", "[a]",
            "-r", rc["fps"], "-c:v", "libx264", "-preset", rc["preset"], "-crf", rc["crf"], "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        ]
        run(cmd + ["-movflags", "+faststart", out.name], cwd=cdir)
        # Preview frame for the review step (taken after the hook disappears).
        run([tool("ffmpeg"), "-v", "error", "-y", "-ss", min(3.0, (clip["end"] - clip["start"]) / 2),
             "-i", out.name, "-frames:v", "1", "-q:v", "3", "preview.jpg"], cwd=cdir)
        write_json(cdir / "clip.json", clip)
        review_queue.add(cfg, clip, source, out)
        outputs.append(out)
        log(f"rendered {clip['id']} ({sum(e - s for s, e, _ in segments):.1f}s, cold open: {len(segments) > 1}, layout: {layout})")
    return outputs
