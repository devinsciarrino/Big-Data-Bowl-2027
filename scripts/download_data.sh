#!/usr/bin/env bash
# Download and unzip the competition data into data/raw/ (~489 MB zip, ~2.3 GB unzipped).
# Prereqs: accept the rules on the Kaggle competition page, and a Kaggle API token at
# ~/.kaggle/access_token (or ~/.kaggle/kaggle.json), chmod 600.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
uv run kaggle competitions download -c nfl-big-data-bowl-2027 -p data/raw -q
unzip -oqj data/raw/nfl-big-data-bowl-2027.zip -d data/raw   # -j: flatten the zip's subfolder
rm data/raw/nfl-big-data-bowl-2027.zip
ls -lh data/raw
