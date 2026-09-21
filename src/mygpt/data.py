import json
from pathlib import Path
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


def load_data(
    path: str | Path,
    tokenizer: CharTokenizer,
    device: str = "cuda",
    val_frac: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    text = Path(path).read_text(encoding="utf-8")
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)
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

    torch.manual_seed(1337)
    x, y = get_batch(train, batch_size=4, block_size=8)
    assert x.shape == y.shape == (4, 8)
    assert x.dtype == torch.int64
    assert (x[:, 1:] == y[:, :-1]).all()

    for t in range(8):
        print(
            f"{tok.decode(x[0, : t + 1].tolist())!r:12} -> {tok.decode([y[0, t].item()])!r}"
        )
