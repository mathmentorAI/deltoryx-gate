#!/usr/bin/env bash
# Downloads the PR-MCI replication package (Gong, Pinna, Bian, Zhang — MSR '26) pinned to the commit used here.
set -euo pipefail
DEST="${PRMCI_DIR:-./data/PR-MCI}"
if [ ! -d "$DEST/.git" ]; then git clone https://github.com/gjz78910/PR-MCI "$DEST"; fi
git -C "$DEST" checkout -q ac6bbf4
echo "PR-MCI ready at $DEST (commit ac6bbf4). Export PRMCI_DIR=$DEST before running run.py"
