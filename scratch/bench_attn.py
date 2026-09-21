# confronto 7c: MultiHeadAttention naive vs CausalSelfAttention fusa+SDPA
# stesse condizioni di train.py: 64x256, bf16 autocast, TF32, torch.compile, fwd+bwd
import time, torch
import mygpt.model as M
from mygpt.data import CharTokenizer

torch.manual_seed(1337)
torch.set_float32_matmul_precision("high")
dev = "cuda"
tok = CharTokenizer.load("data/vocab.json")
B, T = 64, 256
FUSED = M.CausalSelfAttention  # il vero riferimento, Block la cerca per nome a runtime

def bench(attn_cls, label, iters=60, warmup=15):
    M.CausalSelfAttention = attn_cls          # monkeypatch: stessa firma
    torch.manual_seed(1337)
    raw = M.GPT(tok.vocab_size, 384, 6, 6, T, 0.2).to(dev)
    model = torch.compile(raw)
    opt = torch.optim.AdamW(raw.parameters(), lr=1e-3)
    x = torch.randint(tok.vocab_size, (B, T), device=dev)
    y = torch.randint(tok.vocab_size, (B, T), device=dev)
    for i in range(warmup + iters):
        if i == warmup:
            torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
        opt.step()
    torch.cuda.synchronize()
    ms = (time.time() - t0) * 1000 / iters
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"{label:34s} {ms:6.2f} ms/it   {B*T/ms*1000/1e3:5.0f}k tok/s   peak {peak:.2f} GB")
    del raw, model, opt
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return ms

a = bench(M.MultiHeadAttention, "naive (ModuleList di Head)")
b = bench(FUSED,                "fusa c_attn + SDPA")
print(f"\nspeedup: {a/b:.2f}x   ({a-b:+.1f} ms/it)")
