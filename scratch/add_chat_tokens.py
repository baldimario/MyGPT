# aggiunge i token speciali della chat a un checkpoint: l'embedding (e lm_head, che e' lo stesso tensore) cresce di
# una riga per token. Le righe nuove partono dalla MEDIA degli embedding esistenti (Hewitt 2021): con valori casuali
# il loro logit sarebbe arbitrario e all'inizio dell'SFT il modello li sparerebbe a caso
# uso: add_chat_tokens.py in.pt out.pt
import sys

import torch

from mygpt.data import BPETokenizer

SPECIALS = ["<|user|>", "<|assistant|>", "<|end|>"]
CHAT_TOKENIZER = "data/bpe-it-gpt2-chat.json"

src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu", weights_only=True)
base = BPETokenizer.load(ck["tokenizer"])
chat = BPETokenizer(base.merges, base.pattern, SPECIALS)
chat.save(CHAT_TOKENIZER)

wte = ck["model"]["wte.weight"]
assert wte.shape[0] == base.vocab_size, "il checkpoint ha gia' token in piu'?"
new = wte.mean(dim=0, keepdim=True).repeat(len(SPECIALS), 1)
ck["model"]["wte.weight"] = torch.cat([wte, new])
ck["model"]["lm_head.weight"] = ck["model"]["wte.weight"]  # weight tying: nello state_dict sono due chiavi
ck["config"]["vocab_size"] = chat.vocab_size
ck["tokenizer"] = CHAT_TOKENIZER
torch.save(ck, dst)
print(f"{src} -> {dst}: vocab {base.vocab_size} -> {chat.vocab_size}, {chat.special_id}")
