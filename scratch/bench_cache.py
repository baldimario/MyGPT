import time, torch
from mygpt.data import CharTokenizer
from mygpt.model import GPT
ck = torch.load("out/ckpt.pt", map_location="cuda", weights_only=True)
tok = CharTokenizer.load("data/vocab.json")
m = GPT(**ck["config"]).cuda(); m.load_state_dict(ck["model"]); m.eval()
ctx = torch.tensor([tok.encode("EDWARD:\n")], device="cuda")
for use_cache in (False, True, False, True):
    torch.manual_seed(0); torch.cuda.synchronize(); t = time.time()
    out = m.generate(ctx, 500, temperature=0.8, top_k=40, use_cache=use_cache)
    torch.cuda.synchronize(); dt = time.time() - t
    print(f"use_cache={use_cache!s:5s}  {dt:5.2f}s  {500/dt:6.1f} tok/s  len={out.size(1)}")
