#!/usr/bin/env bash
# One command to start everything: sets up a virtual environment, installs
# dependencies, (optionally) trains the disease classifier, then launches the
# web GUI dashboard in your browser.
#
#   bash run_all.sh
set -e

cd "$(dirname "$0")"            # project root (this script's folder)

# 1. Virtual environment + dependencies
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

cd code

# 2. Optional: if a labelled dataset exists, (re)train the disease classifier
if [ -d "../input/dataset" ]; then
    python train.py
fi

# 3. Launch the desktop GUI
echo
echo "Starting the Leaf Disease Analyzer Desktop GUI ..."
python gui.py
