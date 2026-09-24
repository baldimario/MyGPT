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
uv run src/mygpt/generate.py --prompt "Il processore Pentium 4" --tokens 120 --temperature 0.6 --seed 13 --top
-k 95 --top-p 1.0 --repetition-penalty 1.2
# out/ckpt.pt: iter 19500, val loss 3.1310
# temp 0.6 | top-k 95 | top-p 1.0 | rep 1.2
# 120 token in 0.42s = 283 tok/s (kv cache, batch 1)
============================================================
Il processore Pentium 4 è stato progettato per essere utilizzato in sistemi basati sui canali di rendering e i dati delle versioni precedenti.

Le versioni successive furono prodotte nel mercato dei computer, con le versioni del Macintosh, mentre una versione più costosa fu la Sony XP, fondata dalla società Automatic Computing Company. L'azienda ha iniziato a produrre il sistema operativo Intel, che introduceva un nuovo sistema operativo. Il G4MP venne commercializzato sul mercato dapprima come "DOS" o "DSM", ma successivamente anche
```

### next improvements


1. RMSNorm
2. SwiGLU
3. MoE con n_expert e top_k parametrizzati, aux loss e conteggio dei token per esperto nei log
4. Dataset: tutti gli shard, memmap uint16
5. Modello più grande, dimensionato sui token che abbiamo
