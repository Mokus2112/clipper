#!/bin/bash
# Daily Telegram summary (launchd hu.badi.clipper-summary, 20:00).
set -euo pipefail
cd "$(dirname "$0")/.."
/opt/homebrew/bin/uv run clipper summary
