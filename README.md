# clipper

Streamer VOD -> YouTube Shorts pipeline. Only use sources you have permission to clip (e.g. Whop Content Rewards campaigns).

## Steps

1. `fetch`: download the VOD (yt-dlp) or link a local file into `data/jobs/<job>/`
2. `detect`: loudness spikes (+ YouTube live chat velocity if available), used as hints for the ranker
3. `transcribe`: fast full-VOD transcript (whisper base); `captions`: accurate word timestamps for chosen clips
4. `rank`: `claude -p` reads the whole transcript, picks self-contained setup -> payoff stories, scores 1-10 (keeps 7+)
5. `render`: 1080x1920, cold open (payoff first), auto layout (face crop / blurred-background fit / facecam split),
   word-by-word captions, hook overlay
6. `notify`: sends new clips to Telegram for review
7. `review` / `approve` / `reject`: human gate, every clip waits in `data/queue.json`
8. `upload`: YouTube Data API, approved clips only, `upload.daily_limit` per day
9. `cleanup`: `run` deletes the source VOD automatically when done (`fetch.keep_source`)

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
- Section downloads (`--section`) stream through ffmpeg at network speed: ~25 min for 45 min of 1080p60.
- YouTube: per the current videos.insert docs, unverified projects are no longer locked private and uploads have
  their own quota bucket (100/day). Not yet confirmed with a real public upload.
- No music/Content ID detection yet; keep clips under 60 s.
