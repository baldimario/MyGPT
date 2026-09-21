# addestra il BPE italiano su un campione del corpus e lo salva
import sys, time
from mygpt.data import BPETokenizer

sample_mb, vocab, min_freq = 50, 8192, 5
with open("data/input-it.txt", encoding="utf-8") as f:
    sample = f.read(sample_mb * 1024**2)

t = time.time()
tok = BPETokenizer.from_text(sample, vocab_size=vocab, min_freq=min_freq)
tok.save("data/bpe-it.json")
print(f"vocab {tok.vocab_size} da {sample_mb} MB in {(time.time()-t)/60:.1f} min", flush=True)
print("token piu' lunghi:", sorted(tok.itob, key=len)[-8:], flush=True)
ids = tok.encode(sample[: 5 * 1024**2])
print(f"compressione: {len(sample[: 5*1024**2].encode())/len(ids):.2f} byte/token", flush=True)
