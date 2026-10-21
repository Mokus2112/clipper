"""Step 5: render each selected clip as a 1080x1920 Short with layout, word-by-word captions and a hook."""

from pathlib import Path

from .common import find_source, job_dir, log, path_of, probe, read_json, run, source_video, tool, write_json
from . import review_queue


def even(x: float) -> int:
    return int(x) // 2 * 2


def face_center_x(video: Path, start: float, end: float) -> float | None:
    """Median horizontal face position (0..1) over the clip, sampled twice per second."""
    import cv2

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(str(video))
    xs = []
    t = start
    while t < end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        t += 0.5
        if not ok:
            continue
        scale = 480 / frame.shape[1]
        small = cv2.resize(frame, None, fx=scale, fy=scale)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(24, 24))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            xs.append((x + w / 2) / small.shape[1])
    cap.release()
    if not xs:
        return None
    xs.sort()
    return xs[len(xs) // 2]


def layout_filter(cfg: dict, source: dict, info: dict, cx: float | None, inp: str, out: str) -> str:
    rc = cfg["render"]
    W, H = rc["width"], rc["height"]
    sw, sh = info["width"], info["height"]
    if source.get("layout") == "split" and source.get("facecam"):
        fx, fy, fw, fh = source["facecam"]
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
    center = (cx if cx is not None else 0.5) * sw
    x = int(min(max(center - cw / 2, 0), sw - cw))
    return f"[{inp}]crop={cw}:{sh}:{x}:0,scale={W}:{H},setsar=1[{out}]"


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


def build_ass(cfg: dict, clip: dict, segments: list[tuple[float, float, float]], split: bool) -> str:
    """ASS captions for the output timeline; segments map source ranges to output offsets (cold open + main)."""
    cc, hc, rc = cfg["captions"], cfg["hook"], cfg["render"]
    W, H = rc["width"], rc["height"]
    y = cc["y_split"] if split else cc["y_center"]

    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "WrapStyle: 0", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Cap,{cc['font']},{cc['size']},{cc['color']},&H000000FF,&H00000000,&H64000000,"
        f"-1,0,0,0,100,100,0,0,1,{cc['outline']},2,5,90,160,0,1",
        f"Style: Hook,{cc['font']},{hc['size']},&H00000000,&H000000FF,&H00FFFFFF,&H00FFFFFF,"
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
            texts = [ass_escape(w["w"].upper() if cc["uppercase"] else w["w"]) for w in chunk]
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
                    f"{{\\pos({W // 2},{y})}}{' '.join(parts)}"
                )
    return "\n".join(lines) + "\n"


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
    split = source.get("layout") == "split" and bool(source.get("facecam"))

    outputs = []
    for clip in clips:
        cdir = jd / "clips" / clip["id"]
        cdir.mkdir(parents=True, exist_ok=True)
        out = cdir / "short.mp4"
        main = (clip["start"], clip["end"])
        cold = (clip.get("payoff_start"), clip.get("payoff_end"))
        segments = []
        if rc["cold_open"] and cold[0] is not None and cold[1] - cold[0] >= 0.5:
            segments.append((cold[0], cold[1], 0.0))
        segments.append((main[0], main[1], sum(e - s for s, e, _ in segments)))
        (cdir / "captions.ass").write_text(build_ass(cfg, clip, segments, split))

        cx = None if split else face_center_x(video, main[0], main[1])
        inputs, graph, concat = [], [], ""
        for i, (s, e, _) in enumerate(segments):
            inputs += ["-ss", f"{s:.2f}", "-t", f"{e - s:.2f}", "-i", video.resolve()]
            graph.append(layout_filter(cfg, source, info, cx, f"{i}:v", f"l{i}"))
            concat += f"[l{i}][{i}:a:0]"
        audio = ",loudnorm=I=-14:TP=-1.5:LRA=11" if rc["loudnorm"] else ""
        graph.append(f"{concat}concat=n={len(segments)}:v=1:a=1[cv][ca0]")
        graph.append(f"[cv]subtitles=captions.ass:fontsdir={fonts}[v]")
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
        log(f"rendered {clip['id']} ({sum(e - s for s, e, _ in segments):.1f}s, cold open: {len(segments) > 1})")
    return outputs
