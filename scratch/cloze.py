# benchmark fattuale "cloze" sulle voci di film della val: anno ("è un film del ____") e regista ("diretto da ____").
# tre condizioni: senza retrieval (memoria nei pesi), retrieval BM25 compresa la voce stessa (sa leggere?),
# retrieval esclusa la voce stessa (il fatto va trovato in altre voci).
# esatto = a ogni posizione della risposta il token piu' probabile e' quello giusto, cioe' la generazione greedy la
# scriverebbe identica. Un forward per domanda invece di una generazione.
# uso: cloze.py ckpt.pt [train]   (serve data/it-wiki-leads.sqlite, da scratch/build_wiki_index.py)
# con "train" le domande vengono da voci VISTE in training (primi 60M token del train): misura la conoscenza nei pesi
import re
import sqlite3
import sys

import numpy as np
import torch
import torch.nn.functional as F

from mygpt.data import BPETokenizer
from mygpt.model import GPT

ck = torch.load(sys.argv[1], map_location="cuda", weights_only=True)
model = GPT(**ck["config"]).cuda().eval()
model.load_state_dict(ck["model"])
tok = BPETokenizer.load(ck.get("tokenizer", "data/bpe-it.json"))
n_ctx = ck["config"]["block_size"]
db = sqlite3.connect("data/it-wiki-leads.sqlite")

split = sys.argv[2] if len(sys.argv) > 2 else "val"
prefix = "it-gpt2" if tok.pattern == BPETokenizer.GPT2_PAT else "it"
data = np.memmap(f"data/{prefix}-{split}.bin", dtype=np.uint16, mode="r")
text = tok.decode(data[:60_000_000].tolist())  # la val intera e' 14.7M token, il train si campiona dall'inizio
articles = re.split(r"\n\n(?== .+ =\n\n)", text)

WORK = re.compile(
    r"^= (.+?) =\n\n[^\n]{0,120}? è un (?:film|album|singolo|romanzo|videogioco|brano musicale)[^.\n]{0,60}? del (\d{4})\b"
)
DIRECTOR = re.compile(r"^= (.+?) =\n\n[^\n]{0,200}? diretto da ([A-ZÀ-Ý][\w.'-]+ [A-ZÀ-Ý][\w'-]+)")
items = []  # (task, titolo, prompt, risposta)
for a in articles:
    for task, rx in (("anno", WORK), ("regista", DIRECTOR)):
        if m := rx.match(a):
            prompt, answer = a[: m.start(2) - 1], " " + m.group(2)
            if m.group(2) not in prompt:  # es. "= The Killer (film 1989) =": la risposta e' gia' nel titolo
                items.append((task, m.group(1), prompt, answer))


def retrieve(title: str, exclude_self: bool, k: int = 3) -> str:
    q = '"' + re.sub(r" \(.*\)$", "", title).replace('"', "") + '"'  # frase esatta, senza "(film 1989)"
    rows = db.execute(
        "select title, body from leads where leads match ? and title != ? order by rank limit ?",
        (q, title if exclude_self else "", k),
    ).fetchall()
    return "".join(f"= {t} =\n\n{b.strip()}\n\n" for t, b in rows)


@torch.no_grad()
def score(ctx: str, prompt: str, answer: str) -> tuple[bool, float, bool] | None:
    p_ids, a_ids = tok.encode(prompt), tok.encode(answer)
    if tok.encode(prompt + answer) != p_ids + a_ids:
        return None  # il confine prompt/risposta cade dentro un token: domanda non valida
    c_ids = tok.encode(ctx)[: max(0, n_ctx - len(p_ids) - len(a_ids))]  # i passaggi migliori per primi
    ids = torch.tensor([c_ids + p_ids + a_ids], device="cuda")
    logits, _ = model(ids)
    start = len(c_ids) + len(p_ids) - 1  # la posizione che predice il primo token della risposta
    lp = F.log_softmax(logits[0, start : start + len(a_ids)].float(), dim=-1)
    tgt = torch.tensor(a_ids, device="cuda")
    in_ctx = answer.strip() in tok.decode(c_ids)  # la risposta c'era davvero nel contesto rimasto dopo il taglio?
    return bool((lp.argmax(-1) == tgt).all()), lp.gather(-1, tgt[:, None]).sum().item(), in_ctx


conditions = {
    "senza retrieval": lambda t: "",
    "retrieval, voce compresa": lambda t: retrieve(t, exclude_self=False),
    "retrieval, voce esclusa": lambda t: retrieve(t, exclude_self=True),
}
print(f"{ck['config']} | {split} | {len(items)} domande")
for name, ctx_fn in conditions.items():
    res = {"anno": [], "regista": []}
    for task, title, prompt, answer in items:
        if (r := score(ctx_fn(title), prompt, answer)) is not None:
            res[task].append(r)
    for task, r in res.items():
        ok = np.array([e for e, _, _ in r])
        hit = np.array([h for _, _, h in r])
        # separa i due fallimenti: la ricerca non ha portato la risposta, o c'era ma il modello non l'ha usata
        read = f"se nel contesto {100 * ok[hit].mean():5.1f}%" if hit.any() else ""
        print(
            f"{name:26s} {task:8s} esatto {100 * ok.mean():5.1f}% (n={len(r)}, logp {np.mean([lp for _, lp, _ in r]):6.2f})"
            f" | risposta nel contesto {100 * hit.mean():5.1f}% | {read}"
        )
