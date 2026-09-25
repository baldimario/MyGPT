#!/bin/sh
# riparte da out/last.pt se il training muore (es. Xid 8 del driver), al massimo 5 volte
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
uv run src/mygpt/train.py && exit 0
for n in 1 2 3 4 5; do
  echo "=== crash, ripresa $n/5 $(date +%H:%M) ==="
  sleep 30
  uv run src/mygpt/train.py --resume && exit 0
done
