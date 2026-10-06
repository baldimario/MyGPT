MyGPT is a simple toy llm implementation in torch without high level layers implementation

Download tinyshakespeare dataset
```
mkdir -p data && curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Train
```
uv run src/mygpt/train.py
```

Generate
```
uv run src/mygpt/generate.py --prompt "EDWARD" --tokens 160 --temperature 0.6 
```

you can run `--no-cache` to disable KV cache or `--sample N` to see the cache in action


download wikikmedia/wikipedia it dataset
```
curl -L -C - -o data/it-wiki-00.parquet https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.it/train-00000-of-00010.parquet
```

then extract the first chunk with

```
uv run scratch/extract_wikimeia.py
```

---

```
uv run src/mygpt/generate.py --prompt "Il processore Pentium 4" --tokens 120 --temperature 0.6 --seed 31337 
--top-k 95 --top-p 1.0 --repetition-penalty 1.2
# out/ckpt.pt: iter 60000, val loss 2.6099
# temp 0.6 | top-k 95 | top-p 1.0 | rep 1.2
# 120 token in 0.99s = 121 tok/s (kv cache, batch 1)
============================================================
Il processore Pentium 4 fu sostituito da un nuovo coprocessore a 64 bit GPU, il sistema operativo RISC 3.0.

La nuova architettura era molto simile al predecessore: 1ª versione (LX) e 2ª versione (MS) erano basati sullo stesso core Camden, ma con prestazioni superiori rispetto al precedente modello. I cofani erano stati ridisegnati per funzionare in modo indipendente, ed adottavano la tecnologia ABM o ARM di seconda generazione, che permetteva una migliore combustione della batteria
```

## GGUF conversion

```
uv run --with gguf scratch/export_gguf.py out/ckpt.pt out/mygpt-moe-f32.gguf
```

quantize using llama-quantize

```
~/llm/llama.cpp/build/bin/llama-quantize out/mygpt-moe-f32.gguf out/mygpt-moe-Q8_0.gguf Q8_0
~/llm/llama.cpp/build/bin/llama-quantize out/mygpt-moe-f32.gguf out/mygpt-moe-Q4_K_M.gguf Q4_K_M
```

## SFT

```
$ MAX_TOKENS=100 ./chat.sh

Loading model...  


▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                    ██    ██
                                    ▀▀    ▀▀

build      : b9512-0dbfa66a1
model      : mygpt-chat-Q8_0.gguf
modalities : text

available commands:
  /exit or Ctrl+C     stop or exit
  /regen              regenerate the last response
  /clear              clear the chat history
  /read <file>        add a text file
  /glob <pattern>     add text files using globbing pattern


> chi è il regista del signore degli anelli?

Il protagonista è Gandalf.

Prima dell'inizio dei titoli, Gandalf aveva già scritto un libro intitolato "Storie dell'Artiglieria", ma aveva subito solo qualche modifica per adattarlo alle sue esigenze e qualità. 

Durante la sua prima pubblicazione, nel 600 a.C., Gandalf si trovò di fronte ad alcune sfide: la distruzione del suo castello, l'uso di enigmi e la perdita

[ Prompt: 262.9 t/s | Generation: 405.0 t/s ]
```