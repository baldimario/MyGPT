import torch
from mygpt.data import CharTokenizer, get_batch, load_data
from mygpt.model import BigramLM

batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 5000
eval_iters = 200
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)

tok = CharTokenizer.load("data/vocab.json")
train_data, val_data = load_data("data/input.txt", tok, device)

model = BigramLM(tok.vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


@torch.no_grad()
def estimate_loss() -> dict[str, float]:
    model.eval()
    out = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size)
            _, loss = model(x, y)
            losses[k] = loss
        out[name] = losses.mean().item()
    model.train()
    return out


for it in range(max_iters + 1):
    if it % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")

    x, y = get_batch(train_data, batch_size, block_size)
    _, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

start = torch.tensor([tok.encode("\n")], dtype=torch.long, device=device)  # (1, 1)
print("\n" + "=" * 60)
print(tok.decode(model.generate(start, max_new_tokens=500)[0].tolist()))
