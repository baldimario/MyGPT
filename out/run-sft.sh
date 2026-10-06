#!/bin/sh
# aspetta la fine di ctx1024, aggiunge i token della chat e lancia l'SFT
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
until grep -q "miglior val loss" out/ctx1024.log; do sleep 20; done
uv run scratch/add_chat_tokens.py out/ctx1024/ckpt.pt out/ctx1024-chat-init.pt
uv run src/mygpt/train.py config/sft.py
