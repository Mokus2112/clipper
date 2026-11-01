#!/bin/bash
# Weekly meme library refresh -- runs every Sunday at 03:00.
set -euo pipefail
cd "$(dirname "$0")/.."
/opt/homebrew/bin/uv run clipper memes refresh
