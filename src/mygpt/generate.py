import argparse
import time

import torch

from mygpt.data import BPETokenizer
from mygpt.model import GPT

parser = argparse.ArgumentParser(description="Genera testo da un checkpoint di mygpt.")
parser.add_argument("--ckpt", default="out/ckpt.pt")
parser.add_argument("--vocab", default="data/bpe.json")
parser.add_argument("--prompt", default="\n", help="testo iniziale")
parser.add_argument("--tokens", type=int, default=500, help="quanti token generare")
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top-k", type=int, default=40)
parser.add_argument("--samples", type=int, default=1)
parser.add_argument("--no-cache", action="store_true", help="disattiva la KV cache")
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()

assert args.prompt, "il prompt non puo' essere vuoto: serve almeno un token di contesto"

device = "cuda" if torch.cuda.is_available() else "cpu"
if args.seed is not None:
    torch.manual_seed(args.seed)

# weights_only=True: il checkpoint viene deserializzato senza eseguire codice
ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
tok = BPETokenizer.load(args.vocab)

# dropout non e' nel config: resta al default 0.0, che e' quello che vuoi in inferenza
model = GPT(**ckpt["config"]).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

print(f"# {args.ckpt}: iter {ckpt['iter']}, val loss {ckpt['val_loss']:.4f}")

ids = tok.encode(args.prompt)  # byte-level: nessun prompt e' fuori vocabolario

ctx = torch.tensor([ids] * args.samples, dtype=torch.long, device=device)

if device == "cuda":
    torch.cuda.synchronize()  # i kernel sono asincroni: senza questo misuri il lancio
t0 = time.perf_counter()
out = model.generate(
    ctx,
    args.tokens,
    temperature=args.temperature,
    top_k=args.top_k,
    use_cache=not args.no_cache,
)
if device == "cuda":
    torch.cuda.synchronize()
dt = time.perf_counter() - t0

cache = "no cache" if args.no_cache else "kv cache"
n = args.tokens * args.samples
print(f"# {n} token in {dt:.2f}s = {n / dt:.0f} tok/s ({cache}, batch {args.samples})")

for row in out:
    print("=" * 60)
    print(tok.decode(row.tolist()))
