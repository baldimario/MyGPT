# ms/it del passo di training vero (fwd+bwd+step), come in train.py
import sys, time, torch
from mygpt.data import BPETokenizer
from mygpt.model import GPT

torch.manual_seed(1337)
torch.set_float32_matmul_precision("high")
tok = BPETokenizer.load("data/bpe-it.json")
B, T = 64, 256
raw = GPT(tok.vocab_size, 384, 6, 6, T, 0.0).cuda()
model = torch.compile(raw)
opt = torch.optim.AdamW(raw.parameters(), lr=1e-3)
x = torch.randint(tok.vocab_size, (B, T), device="cuda")
y = torch.randint(tok.vocab_size, (B, T), device="cuda")
warmup, iters = 20, 80
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
print(f"{sys.argv[1] if len(sys.argv)>1 else '':12s} {ms:6.2f} ms/it   {B*T/ms*1000/1e3:5.0f}k tok/s")
