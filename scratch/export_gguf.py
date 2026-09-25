# esporta un checkpoint MoE in GGUF per llama.cpp, come architettura "llama" con esperti (= Mixtral)
# uv run --with /home/localghost/llm/llama.cpp/gguf-py scratch/export_gguf.py out/ckpt.pt out/mygpt.gguf
import sys

import gguf
import numpy as np
import torch

from mygpt.data import BPETokenizer

ckpt_path, out_path = sys.argv[1], sys.argv[2]
ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
cfg, sd = ck["config"], ck["model"]
C, nh, L, E = cfg["n_embd"], cfg["n_head"], cfg["n_layer"], cfg["n_expert"]
assert E, "solo MoE: il ramo denso (ffn_gate/up/down) non e' esportato"
hs = C // nh
H = sd["blocks.0.mlp.experts.0.proj.weight"].shape[1]
tok = BPETokenizer.load(ck.get("tokenizer", "data/bpe-it.json"))
V = tok.vocab_size


def np32(t: torch.Tensor) -> np.ndarray:
    return t.float().numpy()


def permute(w: torch.Tensor) -> torch.Tensor:
    # la nostra RoPE ruota le coppie (i, i + hs/2) dentro ogni testa, llama.cpp (ROPE_TYPE_NORM) le coppie (2i, 2i+1):
    # riordino le righe di Wq e Wk, e' la stessa permutazione di convert_hf_to_gguf.py
    return w.reshape(nh, 2, hs // 2, C).swapaxes(1, 2).reshape(C, C)


def bytes_to_unicode() -> dict[int, str]:
    # GPT-2: ogni byte diventa un carattere stampabile, cosi' vocabolario e merge sono stringhe
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


b2u = bytes_to_unicode()
tokens = ["".join(b2u[b] for b in t) for t in tok.itob]
merges = [f"{tokens[a]} {tokens[b]}" for a, b in tok.merges]

w = gguf.GGUFWriter(out_path, "llama")
w.add_name("mygpt-moe")
w.add_context_length(cfg["block_size"])
w.add_embedding_length(C)
w.add_block_count(L)
w.add_feed_forward_length(H)
w.add_head_count(nh)
w.add_head_count_kv(nh)
w.add_layer_norm_rms_eps(1e-5)
w.add_rope_freq_base(10000.0)
w.add_rope_dimension_count(hs)
w.add_expert_count(E)
w.add_expert_used_count(cfg["top_k"])
w.add_vocab_size(V)
w.add_file_type(gguf.LlamaFileType.ALL_F32)

# llama.cpp ha solo pre-tokenizer con nome: il GGUF e' fedele solo per i tokenizer addestrati con GPT2_PAT
if tok.pattern != BPETokenizer.GPT2_PAT:
    print("ATTENZIONE: tokenizer legacy, llama.cpp lo spezzera' con la regex gpt-2 (~+8% bpb sul nostro modello)")
w.add_tokenizer_model("gpt2")
w.add_tokenizer_pre("gpt-2")  # il nome che llama.cpp mappa su LLAMA_VOCAB_PRE_TYPE_GPT2
w.add_token_list(tokens)
w.add_token_types([gguf.TokenType.NORMAL] * V)
w.add_token_merges(merges)
w.add_add_bos_token(False)

# lm_head e' legata a wte: llama.cpp senza output.weight riusa token_embd
w.add_tensor("token_embd.weight", np32(sd["wte.weight"]))
w.add_tensor("output_norm.weight", np32(sd["ln_f.weight"]))
for i in range(L):
    p, g = f"blocks.{i}.", f"blk.{i}."
    q, k, v = sd[p + "attn.c_attn.weight"].split(C, dim=0)
    w.add_tensor(g + "attn_norm.weight", np32(sd[p + "ln1.weight"]))
    w.add_tensor(g + "attn_q.weight", np32(permute(q)))
    w.add_tensor(g + "attn_k.weight", np32(permute(k)))
    w.add_tensor(g + "attn_v.weight", np32(v))
    w.add_tensor(g + "attn_output.weight", np32(sd[p + "attn.proj.weight"]))
    w.add_tensor(g + "attn_output.bias", np32(sd[p + "attn.proj.bias"]))
    w.add_tensor(g + "ffn_norm.weight", np32(sd[p + "ln2.weight"]))
    w.add_tensor(g + "ffn_gate_inp.weight", np32(sd[p + "mlp.router.weight"]))
    fc = torch.stack([sd[p + f"mlp.experts.{e}.fc.weight"] for e in range(E)])  # (E, 2H, C)
    down = torch.stack([sd[p + f"mlp.experts.{e}.proj.weight"] for e in range(E)])  # (E, C, H)
    w.add_tensor(g + "ffn_gate_exps.weight", np32(fc[:, :H]))  # la prima meta' di fc e' il gate (chunk in SwiGLU)
    w.add_tensor(g + "ffn_up_exps.weight", np32(fc[:, H:]))
    w.add_tensor(g + "ffn_down_exps.weight", np32(down))

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"{out_path}: {L} layer, {E} esperti top-{cfg['top_k']}, vocab {V}, step {ck['iter']}")
