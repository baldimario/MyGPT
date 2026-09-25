# confronta il modello PyTorch con llama.cpp: stessa PPL di llama-perplexity (finestre di n_ctx, si contano solo
# le previsioni della seconda meta') sugli STESSI token. Se la conversione dei pesi e' giusta i due numeri coincidono.
# uso: verify_gguf.py ckpt.pt val.txt ids_llama.txt
import ast, math, sys

import torch
import torch.nn.functional as F

from mygpt.data import BPETokenizer
from mygpt.model import GPT

ckpt_path, txt_path, ids_path = sys.argv[1:4]
ck = torch.load(ckpt_path, map_location="cuda", weights_only=True)
model = GPT(**ck["config"]).cuda().eval()
model.load_state_dict(ck["model"])
tok = BPETokenizer.load(ck.get("tokenizer", "data/bpe-it.json"))
n_ctx = ck["config"]["block_size"]


@torch.no_grad()
def score(ids: list[int]) -> tuple[float, float]:
    # (PPL alla llama-perplexity, bit per byte sugli stessi token contati)
    n_chunk = len(ids) // n_ctx
    x = torch.tensor(ids[: n_chunk * n_ctx], device="cuda").view(n_chunk, n_ctx)
    first = n_ctx // 2
    nll, nbytes = 0.0, 0
    for i in range(0, n_chunk, 64):
        xb = x[i : i + 64]
        logits, _ = model(xb)  # fp32, niente autocast: il GGUF e' F32
        lp = F.log_softmax(logits[:, first:-1].float(), dim=-1)
        tgt = xb[:, first + 1 :]
        nll -= lp.gather(-1, tgt[..., None]).sum().item()
        nbytes += sum(len(tok.itob[t]) for t in tgt.flatten().tolist())
    n = n_chunk * (n_ctx - 1 - first)
    return math.exp(nll / n), nll / (math.log(2) * nbytes)


ids_llama = ast.literal_eval(open(ids_path).read())
ids_ours = tok.encode(open(txt_path, encoding="utf-8").read())
same = sum(a == b for a, b in zip(ids_llama, ids_ours))
print(f"token: llama.cpp {len(ids_llama):,} | nostri {len(ids_ours):,} | uguali nello stesso posto {same:,}")
print(f"primo punto di divergenza: {next(i for i, (a, b) in enumerate(zip(ids_llama, ids_ours)) if a != b)}")
ppl, bpb = score(ids_llama)
print(f"PyTorch sui token di llama.cpp: PPL {ppl:.4f}  bpb {bpb:.4f}")
ppl, bpb = score(ids_ours)
print(f"PyTorch sui NOSTRI token:       PPL {ppl:.4f}  bpb {bpb:.4f}")
