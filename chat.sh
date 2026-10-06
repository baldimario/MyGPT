#!/bin/sh
# chat con il modello SFT in llama.cpp (template e token speciali sono dentro il GGUF)
# uso:   ./chat.sh                         chat interattiva, Ctrl+C per uscire
#        ./chat.sh -st -p "Ciao, chi sei?"  una sola domanda e via
#        MODEL=out/mygpt-chat-f32.gguf ./chat.sh
# altri flag di llama-cli si aggiungono in coda e vincono su questi
LLAMA_CPP="${LLAMA_CPP:-$HOME/llm/llama.cpp}"
MODEL="${MODEL:-out/mygpt-chat-Q8_0.gguf}"
cd "$(dirname "$0")"
# -c 1024: il contesto con cui e' stato addestrato. Sampling: gli stessi default di generate.py
exec "$LLAMA_CPP/build/bin/llama-cli" -m "$MODEL" -c 1024 --jinja -cnv \
  --temp 0.7 --top-k 40 --repeat-penalty 1.15 "$@"
