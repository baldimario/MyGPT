import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch


class CharTokenizer:
    UNK = "\ufffd"

    def __init__(self, itos: list[str]) -> None:
        self.itos = itos
        self.stoi = {c: i for i, c in enumerate(itos)}
        self.unk_id = self.stoi[self.UNK]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls([cls.UNK] + sorted(set(text) - {cls.UNK}))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> list[int]:
        return [self.stoi.get(c, self.unk_id) for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.itos, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    # sostituisce ogni occorrenza di pair con new_id, da sinistra e senza overlap
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    # BPE byte-level, il vocabolario parte dai 256 byte, quindi NON esiste l'UNK token, qualunque testo e' rappresentabile. La pre-tokenizzazione spezza il testo prima di fondere, cosi' nessun token scavalca un confine di parola, lo spazio resta attaccato alla parola che segue, come in GPT-2
    PAT = re.compile(r" ?\w+| ?[^\w\s]+|\s+")

    def __init__(self, merges: list[tuple[int, int]]) -> None:
        self.merges = [tuple(pair) for pair in merges]
        # il rank e' l'ordine di creazione: serve a rifondere nello stesso ordine
        self.ranks = {pair: 256 + i for i, pair in enumerate(self.merges)}
        self.itob = [bytes([i]) for i in range(256)]  # id -> byte rappresentati
        for a, b in self.merges:
            self.itob.append(self.itob[a] + self.itob[b])
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.itob)

    @classmethod
    def from_text(
        cls, text: str, vocab_size: int = 1024, min_freq: int = 1
    ) -> "BPETokenizer":
        assert vocab_size >= 256, "i 256 byte sono il punto di partenza obbligato"
        # si allena sui chunk UNICI pesati per frequenza: 15k invece di 295k
        freqs = Counter(cls.PAT.findall(text))
        if min_freq > 1:
            # i chunk rari costano quanto gli altri a ogni giro ma pesano
            # pochissimo sul conteggio delle coppie. Su it.wikipedia, min_freq=5
            # butta il 75% dei chunk unici e conserva il 96% delle occorrenze.
            freqs = Counter({c: n for c, n in freqs.items() if n >= min_freq})
        seqs = {chunk: list(chunk.encode()) for chunk in freqs}

        merges = []
        for new_id in range(256, vocab_size):
            pairs = Counter()
            for chunk, ids in seqs.items():
                for pair in zip(ids, ids[1:]):
                    pairs[pair] += freqs[chunk]
            if not pairs:
                break  # niente piu' da fondere: il corpus e' tutto token singoli
            best = max(pairs, key=pairs.__getitem__)
            merges.append(best)
            # ricontiamo tutte le coppie a ogni fusione. 10s una tantum
            # su tinyshakespeare; con un indice coppia -> chunk si passerebbe a
            # aggiornamenti locali, da fare solo se il corpus cresce di un ordine
            for chunk, ids in seqs.items():
                if len(ids) > 1:
                    seqs[chunk] = merge(ids, best, new_id)
        return cls(merges)

    def _encode_chunk(self, chunk: str) -> list[int]:
        if chunk not in self._cache:
            ids = list(chunk.encode())
            while len(ids) > 1:
                # si fonde sempre la coppia col rank piu' basso, cioe' quella nata
                # prima in training: ricostruisce la stessa storia di fusioni
                pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, 1 << 30))
                if pair not in self.ranks:
                    break  # nessuna coppia rimasta e' nel vocabolario
                ids = merge(ids, pair, self.ranks[pair])
            self._cache[chunk] = ids
        return self._cache[chunk]

    def encode(self, s: str) -> list[int]:
        return [i for chunk in self.PAT.findall(s) for i in self._encode_chunk(chunk)]

    def decode(self, ids: list[int]) -> str:
        # errors="replace": generando, il modello puo' fermarsi a meta' di un
        # carattere multi-byte. Meglio un tofu che un'eccezione.
        return b"".join(self.itob[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.merges))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        return cls(json.loads(Path(path).read_text()))


def encode_to_bin(
    tokenizer: BPETokenizer,
    txt_path: str | Path,
    bin_path: str | Path,
    chunk_mb: int = 8,
) -> int:
    # codifica a pezzi e scrive uint16. Su mezzo giga di testo la lista Python di
    # ~130M interi costerebbe ~5 GB di RAM; su disco sono 260 MB. Va rigenerato
    # ogni volta che cambia il tokenizer: gli id non vogliono piu' dire lo stesso.
    assert tokenizer.vocab_size <= 65536, "uint16 non basta per questo vocabolario"
    n = 0
    with open(txt_path, encoding="utf-8") as src, open(bin_path, "wb") as dst:
        while block := src.read(chunk_mb * 1024**2):
            block += src.readline()  # non spezzare una riga a meta'
            ids = np.array(tokenizer.encode(block), dtype=np.uint16)
            ids.tofile(dst)
            n += len(ids)
    return n


def load_data(
    path: str | Path,
    tokenizer: "CharTokenizer | BPETokenizer",
    device: str = "cuda",
    val_frac: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".bin":  # gia' codificato da encode_to_bin
        ids = np.fromfile(path, dtype=np.uint16).astype(np.int64)
        return _split(torch.from_numpy(ids).to(device), val_frac)
    text = path.read_text(encoding="utf-8")
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)
    return _split(data, val_frac)


def _split(data: torch.Tensor, val_frac: float) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(len(data) * (1 - val_frac))
    return data[:n], data[n:]


def get_batch(
    data: torch.Tensor, batch_size: int, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(
        len(data) - block_size, (batch_size,), device=data.device
    )  # (B,)
    off = torch.arange(block_size, device=data.device)  # (T,)
    idx = ix[:, None] + off  # (B, T)
    return data[idx], data[idx + 1]


if __name__ == "__main__":
    text = Path("data/input.txt").read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)

    # assert that tokenizer is a bijection
    assert tok.decode(tok.encode(text)) == text

    # assert unknown chars encoded correctly
    assert tok.encode("日本語") == [0, 0, 0]

    # assert vocab survives round-trip on disk
    tok.save("data/vocab.json")
    assert CharTokenizer.load("data/vocab.json").itos == tok.itos

    print(f"vocab_size = {tok.vocab_size}")
    print(repr("".join(tok.itos[1:])))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val = load_data("data/input.txt", tok, device)
    print(
        f"train = {len(train):,} token | val = {len(val):,} token | device = {train.device}"
    )

    # BPE: si allena una volta e si salva, 10s
    bpe_path = Path("data/bpe.json")
    if bpe_path.exists():
        bpe = BPETokenizer.load(bpe_path)
    else:
        bpe = BPETokenizer.from_text(text, vocab_size=1024)
        bpe.save(bpe_path)

    assert bpe.vocab_size == 1024
    assert bpe.decode(bpe.encode(text)) == text  # bigezione sul corpus
    assert bpe.decode(bpe.encode("日本語")) == "日本語"  # byte-level: nessun UNK
    assert BPETokenizer.load(bpe_path).merges == bpe.merges

    ids = bpe.encode(text)
    print(
        f"BPE: vocab {bpe.vocab_size} | {len(ids):,} token | "
        f"{len(text.encode()) / len(ids):.2f} byte/token | "
        f"{len(text.encode()) / len(ids) / 1:.2f}x piu' corto del char-level"
    )
    print("  token piu' lunghi:", sorted(bpe.itob, key=len)[-6:])
    print(
        "  'To be, or not to be' ->",
        [bpe.itob[i] for i in bpe.encode("To be, or not to be")],
    )

    torch.manual_seed(1337)
    x, y = get_batch(train, batch_size=4, block_size=8)
    assert x.shape == y.shape == (4, 8)
    assert x.dtype == torch.int64
    assert (x[:, 1:] == y[:, :-1]).all()

    for t in range(8):
        print(
            f"{tok.decode(x[0, : t + 1].tolist())!r:12} -> {tok.decode([y[0, t].item()])!r}"
        )
