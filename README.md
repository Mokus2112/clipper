# clipper

Streamer VOD -> YouTube Shorts pipeline. Only use sources you have permission to clip (e.g. Whop Content Rewards campaigns).

## Steps

1. `fetch`: download the VOD (yt-dlp) or link a local file into `data/jobs/<job>/`
2. `detect`: loudness spikes (+ YouTube live chat velocity if available) -> candidate windows
3. `transcribe`: faster-whisper word timestamps, only for the candidate windows
4. `rank`: `claude -p` picks the best moments, writes hook + title, scores 1-10
5. `render`: 1080x1920, face-tracked crop or facecam/gameplay split, word-by-word captions, hook overlay
6. `review` / `approve` / `reject`: human gate, every clip waits in `data/queue.json`
7. `upload`: YouTube Data API, approved clips only, `upload.daily_limit` per day

## Usage

```
uv run clipper run <source> <url-or-file> [--section 01:10:00-01:40:00]
uv run clipper review
uv run clipper approve <clip-id>
uv run clipper upload
```

Each step can be re-run on its own: `uv run clipper rank <job> --min-score 5`.

## Setup

- Tools in `~/.local/bin`: `ffmpeg`/`ffprobe` (BtbN build; the johnvansickle static build segfaults on network input), `yt-dlp`, `uv`. YouTube extraction uses `bun` as the JS runtime.
- Add a source per permitted streamer/campaign in `config.yaml` (`layout: center` or `split` with a `facecam` rectangle).
- YouTube upload: create a Google Cloud project, enable YouTube Data API v3, configure the OAuth consent screen (add your account as test user), create an OAuth client of type "Desktop app", save it as `secrets/client_secret.json`, then run `uv run clipper auth`.

## Known limits

- CPU only: Whisper `small` takes ~13 s per 45 s window; rendering runs at roughly 0.6x realtime on 4 cores.
- Face crop is static per clip (median face position), no dynamic tracking yet.
- No music/Content ID detection yet; keep clips under 60 s.
