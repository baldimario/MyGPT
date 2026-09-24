# corpus italiano completo: i 10 shard di wikipedia -> data/it-train.bin + data/it-val.bin (uint16)
# la val resta ESATTAMENTE quella di input-it.bin (il suo ultimo 10%), cosi' i bpb restano confrontabili coi run vecchi
import time
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

from mygpt.data import BPETokenizer

SKIP0 = 116_000  # articoli dello shard 0 gia' in input-it.bin: extract_wikimedia.py si e' fermato li'


def encode_shard(i: int) -> tuple[int, int]:
    tok = BPETokenizer.load("data/bpe-it.json")  # ogni processo ha la sua cache dei chunk
    pf = pq.ParquetFile(f"data/it-wiki-{i:02d}.parquet")
    seen = n_tok = 0
    with open(f"data/it-wiki-{i:02d}.bin", "wb") as dst:
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
    t = time.time()
    with Pool(5) as pool:
        for i, n in pool.imap_unordered(encode_shard, range(10)):
            print(f"shard {i:02d}: {n:,} token  ({time.time() - t:.0f}s)", flush=True)

    old = np.fromfile("data/input-it.bin", dtype=np.uint16)
    n_train = int(len(old) * 0.9)  # stesso taglio di data._split
    old[n_train:].tofile("data/it-val.bin")
    with open("data/it-train.bin", "wb") as dst:
        old[:n_train].tofile(dst)
        for i in range(10):
            np.fromfile(f"data/it-wiki-{i:02d}.bin", dtype=np.uint16).tofile(dst)

    train = np.memmap("data/it-train.bin", dtype=np.uint16, mode="r")
    val = np.memmap("data/it-val.bin", dtype=np.uint16, mode="r")
    print(f"train {len(train):,} token | val {len(val):,} token | {time.time() - t:.0f}s")
