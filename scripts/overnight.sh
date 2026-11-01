#!/bin/bash
# Overnight practice batch -- runs full pipeline for all practice sources.
# Cron: 0 2 * * * /path/to/overnight.sh >> /path/to/overnight.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
/opt/homebrew/bin/uv run clipper overnight
