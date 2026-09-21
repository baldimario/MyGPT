import math
import time
from pathlib import Path

import torch

from mygpt.data import CharTokenizer, get_batch, load_data
from mygpt.model import GPT

# config
n_embd, n_head, n_layer = 384, 6, 6
block_size = 256
dropout = 0.2

batch_size = 64
max_iters = 2000
learning_rate = 1e-3
min_lr = 1e-4
warmup_iters = 100
weight_decay = 0.1
betas = (0.9, 0.99)
grad_clip = 1.0

eval_interval = 250
eval_iters = 200
compile_model = True
out_dir = Path("out")

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
torch.set_float32_matmul_precision("high")  # abilita i tensor core TF32


def sync() -> None:
    if device == "cuda":
        torch.cuda.synchronize()


# dati e modello
tok = CharTokenizer.load("data/vocab.json")
train_data, val_data = load_data("data/input.txt", tok, device)

raw_model = GPT(tok.vocab_size, n_embd, n_head, n_layer, block_size, dropout).to(device)
print(f"{sum(p.numel() for p in raw_model.parameters()):,} parametri")

# weight decay solo sui tensori 2D+ (le matrici dei matmul), non su bias e LayerNorm
decay = [p for p in raw_model.parameters() if p.dim() >= 2]
no_decay = [p for p in raw_model.parameters() if p.dim() < 2]
print(f"  decay:    {len(decay):3d} tensori, {sum(p.numel() for p in decay):>10,}")
print(
    f"  no decay: {len(no_decay):3d} tensori, {sum(p.numel() for p in no_decay):>10,}"
)

optimizer = torch.optim.AdamW(
    [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=learning_rate,
    betas=betas,
)

model = torch.compile(raw_model) if compile_model else raw_model


# learning rate scheduler
def get_lr(it: int) -> float:
    if it < warmup_iters:  # rampa lineare
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > max_iters:
        return min_lr
    ratio = (it - warmup_iters) / (max_iters - warmup_iters)  # 0 -> 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))  # 1 -> 0
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss() -> dict[str, float]:
    model.eval()
    out = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                _, loss = model(x, y)
            losses[k] = loss
        out[name] = losses.mean().item()
    model.train()
    return out


# learning loop
best_val = float("inf")
t_last, it_last = time.time(), 0

for it in range(max_iters + 1):
    lr = get_lr(it)
    for group in optimizer.param_groups:
        group["lr"] = lr

    if it % eval_interval == 0:
        sync()
        elapsed = time.time() - t_last
        n_it = it - it_last
        ms = elapsed * 1000 / n_it if n_it else float("nan")
        tok_s = batch_size * block_size * n_it / elapsed if n_it else 0.0

        losses = estimate_loss()
        print(
            f"step {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
            f"| lr {lr:.2e} | {ms:6.1f} ms/it | {tok_s / 1e3:5.0f}k tok/s"
        )

        if losses["val"] < best_val and it > 0:
            best_val = losses["val"]
            out_dir.mkdir(exist_ok=True)
            torch.save(
                {
                    "model": raw_model.state_dict(),
                    "config": dict(
                        vocab_size=tok.vocab_size,
                        n_embd=n_embd,
                        n_head=n_head,
                        n_layer=n_layer,
                        block_size=block_size,
                    ),
                    "iter": it,
                    "val_loss": best_val,
                },
                out_dir / "ckpt.pt",
            )
        t_last, it_last = time.time(), it  # l'eval non conta nel tempo/iter

    x, y = get_batch(train_data, batch_size, block_size)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        _, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
    optimizer.step()

print(f"\nmiglior val loss: {best_val:.4f}  ->  {out_dir / 'ckpt.pt'}")

# sampling
raw_model.eval()
ctx = torch.tensor([tok.encode("\n")], dtype=torch.long, device=device)
print("=" * 60)
print(tok.decode(raw_model.generate(ctx, 1000, temperature=0.8, top_k=40)[0].tolist()))
