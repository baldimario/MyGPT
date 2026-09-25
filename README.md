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

### next improvements

gguf to run on llama.cpp
