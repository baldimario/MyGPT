# dati per l'SFT: conversazioni -> righe di T+1 token impacchettate, con una maschera di cosa va predetto
#   <|user|> domanda <|end|> <|assistant|> risposta <|end|>
#   maschera:  0 ...   0       0            1 ... 1   1      <- loss solo su risposta e sul suo <|end|> (impara a fermarsi)
# uso: build_sft.py  -> data/sft-{train,val}.bin (uint16) + data/sft-{train,val}.mask.bin (uint8)
import random

import numpy as np
import pyarrow.parquet as pq

from mygpt.data import BPETokenizer

T = 1024
ROW = T + 1  # x = riga[:-1], y = riga[1:]
tok = BPETokenizer.load("data/bpe-it-gpt2-chat.json")
U, A, E = (tok.special_id[s] for s in ("<|user|>", "<|assistant|>", "<|end|>"))

convs = pq.read_table("data/sft-capybara-ita.parquet").column("conversations").to_pylist()
random.seed(1337)
random.shuffle(convs)
n_val = len(convs) * 3 // 100


def encode(conv: list[dict]) -> tuple[list[int], list[int]] | None:
    # tiene gli scambi interi (domanda + risposta) finche' entrano in una riga; mai una risposta tagliata a meta',
    # altrimenti il modello imparerebbe risposte senza <|end|>
    ids, mask = [], []
    for user, asst in zip(conv[::2], conv[1::2]):
        assert user["role"] == "user" and asst["role"] == "assistant"
        q = [U] + tok.encode(user["content"].strip()) + [E, A]
        a = tok.encode(asst["content"].strip()) + [E]
        if len(ids) + len(q) + len(a) > ROW:
            break
        ids += q + a
        mask += [0] * len(q) + [1] * len(a)
    return (ids, mask) if ids else None  # None: neanche il primo scambio entra in 1024


def pack(split: list[list[dict]], name: str) -> None:
    rows, rmask, cur, cmask = [], [], [], []
    used = pad = 0
    for conv in split:
        if (enc := encode(conv)) is None:
            continue
        ids, mask = enc
        used += 1
        if len(cur) + len(ids) > ROW:  # la conversazione non entra: chiudo la riga riempiendo con <|end|> mascherati
            pad += ROW - len(cur)
            rows.append(cur + [E] * (ROW - len(cur)))
            rmask.append(cmask + [0] * (ROW - len(cmask)))
            cur, cmask = [], []
        cur += ids
        cmask += mask
    if cur:
        pad += ROW - len(cur)
        rows.append(cur + [E] * (ROW - len(cur)))
        rmask.append(cmask + [0] * (ROW - len(cmask)))
    r, m = np.array(rows, dtype=np.uint16), np.array(rmask, dtype=np.uint8)
    r.tofile(f"data/sft-{name}.bin")
    m.tofile(f"data/sft-{name}.mask.bin")
    print(
        f"{name}: {used:,}/{len(split):,} conversazioni -> {len(rows):,} righe | "
        f"token di risposta {m.mean():.0%} | riempimento {1 - pad / r.size:.0%}"
    )


pack(convs[n_val:], "train")
pack(convs[:n_val], "val")
