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
