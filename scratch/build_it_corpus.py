# corpus italiano completo, tokenizzato con un tokenizer qualsiasi -> data/<prefix>-train.bin + data/<prefix>-val.bin (uint16)
# uso: build_it_corpus.py data/bpe-it-gpt2.json it-gpt2
# la val e' sempre lo STESSO TESTO: l'ultimo 10% di input-it.bin (il primo corpus, scritto col tokenizer legacy),
# decodificato e ricodificato. Cosi' i bpb restano confrontabili fra tokenizer diversi e con tutti i run passati.
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from mygpt.data import BPETokenizer

TOKENIZER, PREFIX = sys.argv[1], sys.argv[2]
LEGACY = "data/bpe-it.json"  # quello con cui e' scritto input-it.bin
SKIP0 = 116_000  # articoli dello shard 0 gia' in input-it.bin: extract_wikimedia.py si e' fermato li'
VAL, TRAIN0 = -1, -2  # task che non sono shard


def legacy_split() -> tuple[np.ndarray, np.ndarray]:
    old = np.fromfile("data/input-it.bin", dtype=np.uint16)
    n_train = int(len(old) * 0.9)  # stesso taglio di data._split
    return old[:n_train], old[n_train:]


def encode_task(i: int) -> tuple[int, int]:
    tok = BPETokenizer.load(TOKENIZER)  # ogni processo ha la sua cache dei chunk
    out = f"data/{PREFIX}-part{i:02d}.bin"
    n_tok = 0
    with open(out, "wb") as dst:
        if i in (VAL, TRAIN0):
            train0, val = legacy_split()
            text = BPETokenizer.load(LEGACY).decode((val if i == VAL else train0).tolist())
            if i == VAL:
                # stringa unica: la tokenizzazione e' esattamente quella che darebbe llama.cpp sullo stesso testo
                pieces = [text]
            else:
                # 8 MB alla volta, tagliando a fine riga: un token diverso ogni tanto sul bordo, come encode_to_bin
                pieces, start = [], 0
                while start < len(text):
                    end = text.find("\n", start + 8 * 1024**2)
                    end = len(text) if end < 0 else end + 1
                    pieces.append(text[start:end])
                    start = end
            for piece in pieces:
                ids = np.array(tok.encode(piece), dtype=np.uint16)
                ids.tofile(dst)
                n_tok += len(ids)
            return i, n_tok
        pf = pq.ParquetFile(f"data/it-wiki-{i:02d}.parquet")
        seen = 0
        for batch in pf.iter_batches(batch_size=1000, columns=["title", "text"]):
            titles = batch.column("title").to_pylist()
            texts = batch.column("text").to_pylist()
            # stesso formato di extract_wikimedia.py
            docs = [f"= {t} =\n\n{x}\n\n" for t, x in zip(titles, texts)]
            start = max(0, SKIP0 - seen) if i == 0 else 0
            seen += len(docs)
            if start >= len(docs):
                continue
            ids = np.array(tok.encode("".join(docs[start:])), dtype=np.uint16)
            ids.tofile(dst)
            n_tok += len(ids)
    return i, n_tok


if __name__ == "__main__":
    assert BPETokenizer.load(TOKENIZER).vocab_size <= 65536, "uint16 non basta per questo vocabolario"
    t = time.time()
    tasks = [TRAIN0, VAL] + list(range(10))
    with Pool(6) as pool:
        for i, n in pool.imap_unordered(encode_task, tasks):
            name = {VAL: "val", TRAIN0: "train shard 0 (gia' in input-it.bin)"}.get(i, f"shard {i:02d}")
            print(f"{name}: {n:,} token  ({time.time() - t:.0f}s)", flush=True)

    def part(i: int) -> str:
        return f"data/{PREFIX}-part{i:02d}.bin"

    with open(f"data/{PREFIX}-train.bin", "wb") as dst:
        for i in [TRAIN0] + list(range(10)):
            np.fromfile(part(i), dtype=np.uint16).tofile(dst)
    np.fromfile(part(VAL), dtype=np.uint16).tofile(f"data/{PREFIX}-val.bin")
    for i in tasks:
        Path(part(i)).unlink()

    train = np.memmap(f"data/{PREFIX}-train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(f"data/{PREFIX}-val.bin", dtype=np.uint16, mode="r")
    print(f"train {len(train):,} token | val {len(val):,} token | {time.time() - t:.0f}s")
