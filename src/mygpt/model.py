import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim) * 0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.weight[idx]  # idx (...) integer -> out (..., embedding_dim)


class BigramLM(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding_table = Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            return logits, None

        B, T, V = logits.shape
        # le B*T psoizioni sono B*T esempi indipendenti
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx)  # (B, T, V)
            logits = logits[:, -1, :]  # (B, V) solo l'ultima posizione
            probs = F.softmax(logits, dim=-1)  # (B, V)
            next_idx = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, next_idx), dim=1)  # (B, T+1)
        return idx


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = logits - logits.max(dim=-1, keepdim=True).values  # shift-invariant
    logprobs = logits - logits.exp().sum(dim=-1, keepdim=True).log()  # log_softmax
    return -logprobs[torch.arange(len(targets)), targets].mean()


class Linear(nn.Module):
    # y = x @ W.T + b
    # W ha spare (out_features, in_features), non il contrario è la convenzione di torch, serve per il weight tying dopo

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features) -> (..., out_features)
        out = x @ self.weight.T
        return out if self.bias is None else out + self.bias


class Dropout(nn.Module):
    # azzera ogni attivazione con probabilità p durante il training, i sopravvissuti
    # vengono riscalati di 1/(1-p) così il valore atteso dell'output è identico
    # in training e in inferenza (inverted dropout)

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = (torch.rand_like(x) > self.p).to(x.dtype) / (1.0 - self.p)
        return x * mask


class Head(nn.Module):
    # single head attention

    def __init__(
        self, n_embd: int, head_size: int, block_size: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.key = Linear(n_embd, head_size, bias=False)
        self.query = Linear(n_embd, head_size, bias=False)
        self.value = Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)  # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        v = self.value(x)  # (B, T, hs)

        # affinità di ogni query  contro ogni key
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        return wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)


class MultiHeadAttention(nn.Module):
    # n_head teste in paralello contatenate e riproiettate

    def __init__(
        self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        assert n_embd % n_head == 0, "n_embd deve essere divisibile per n_head"
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = Linear(n_embd, n_embd)
        self.dropout = Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, nh*hs) = (B, T, C)
        return self.dropout(self.proj(out))


class LayerNorm(nn.Module):
    # normalize every feature vector to mean 0 and variance 1 and then  rescale it, it works on the last dimension, for each token in the sequence indipendently

    def __init__(self, ndim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))  # gamma: parte dell'identità
        self.bias = nn.Parameter(torch.zeros(ndim))  # beta
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)  # (..., 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # (..., 1)
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * xhat + self.bias


class MLP(nn.Module):
    # espande 4x, non linearità, ricomprime, nessuna comunicazione tra token
    def __init__(self, n_embd: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc = Linear(n_embd, 4 * n_embd)
        self.proj = Linear(4 * n_embd, n_embd)
        self.dropout = Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    # un blocco transformer, comunicazione e computazione pre-ln

    def __init__(
        self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.ln1 = LayerNorm(n_embd)
        # self.attn = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))  # i token si parlano
        x = x + self.mlp(self.ln2(x))  # ogni token elabora quello che ha sentito
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.block_size = block_size

        self.wte = Embedding(vocab_size, n_embd)  # token -> vector
        self.wpe = Embedding(block_size, n_embd)  # position -> vector
        self.drop = Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = LayerNorm(n_embd)  # obbligatoria in pre-LN
        self.lm_head = Linear(n_embd, vocab_size, bias=False)

        # weight tying, LO STESSO tensore, non una copia
        self.lm_head.weight = self.wte.weight

        # GPT-2: le proiezioni che scrivono sul residual stream partono più piccole
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        # da che posizione ripartono i token nuovi: quanti ce n'e' gia' in cache
        cache = self.blocks[0].attn.cache
        t0 = 0 if cache is None else min(cache[0].size(2), self.block_size - T)
        assert t0 + T <= self.block_size, (
            f"sequenza di {T} token a partire da {t0}, block_size è {self.block_size}"
        )

        pos = torch.arange(t0, t0 + T, device=idx.device)  # (T,)
        x = self.drop(self.wte(idx) + self.wpe(pos))  # (B, T, C) + (T, C) -> (B, T, C)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, C)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    def set_cache(self, enabled: bool) -> None:
        # svuota sempre: una cache vecchia e' peggio di nessuna cache
        for block in self.blocks:
            block.attn.use_cache = enabled
            block.attn.cache = None

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        self.set_cache(use_cache)
        for i in range(max_new_tokens):
            # con la cache: il primo giro macina tutto il prompt (prefill), poi un
            # token alla volta. Senza: si rifa' tutta la finestra ogni volta.
            idx_cond = (
                (idx if i == 0 else idx[:, -1:])
                if use_cache
                else idx[:, -self.block_size :]  # il modello non vede oltre
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, V)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.set_cache(False)  # il modello torna stateless, la memoria si libera
        return idx


class CausalSelfAttention(nn.Module):
    # stessa matematica della MultiHeadAttentio ma con 1 matmul per q, k e v invece di 3*n_head, e l amatrice (B, nh, T, T) no viene mai scritta in memoria

    def __init__(
        self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        assert n_embd % n_head == 0, "n_embd deve essere divisibile per n_head"
        self.n_head = n_head
        self.dropout_p = dropout

        self.c_attn = Linear(n_embd, 3 * n_embd, bias=False)  # q, k, v insieme
        self.proj = Linear(n_embd, n_embd)
        self.dropout = Dropout(dropout)

        # KV cache: spenta durante il training, accesa da GPT.set_cache in generate
        self.block_size = block_size
        self.use_cache = False
        self.cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)  # 3 * (B, T, C)

        # (B, T, C) -> (B, T, nh, hx) -> (B, nh, T, hs): la head diventa una dim di batch
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        if self.use_cache:
            if self.cache is not None:  # i k,v del passato sono gia' calcolati
                k = torch.cat((self.cache[0], k), dim=2)  # (B, nh, T_past+T, hs)
                v = torch.cat((self.cache[1], v), dim=2)
            if k.size(2) > self.block_size:  # finestra scorrevole
                k, v = k[:, :, -self.block_size :], v[:, :, -self.block_size :]
            self.cache = (k, v)

        # is_causal ancora la maschera IN ALTO A SINISTRA: ha senso solo se q e k
        # sono lunghe uguali. In decode q e' 1 token contro tutto il passato, e
        # non c'e' nessun futuro da nascondere.
        assert q.size(2) == k.size(2) or q.size(2) == 1, (
            "decode a piu' token per volta richiederebbe una maschera esplicita"
        )
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=q.size(2) == k.size(2),
        )  # (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # riconcatena le teste
        return self.dropout(self.proj(y))


if __name__ == "__main__":
    import math
    from mygpt.data import CharTokenizer, get_batch, load_data

    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # our cross_entropy is the same of torch cross_entropy
    _logits, _targets = torch.randn(32, 66), torch.randint(66, (32,))
    assert torch.allclose(
        cross_entropy(_logits, _targets), F.cross_entropy(_logits, _targets)
    )

    tok = CharTokenizer.load("data/vocab.json")
    train, _val = load_data("data/input.txt", tok, device)

    m = BigramLM(tok.vocab_size).to(device)
    x, y = get_batch(train, batch_size=4, block_size=8)
    logits, loss = m(x, y)

    print(f"parametri = {sum(p.numel() for p in m.parameters()):,}")
    print(f"logits = {tuple(logits.shape)}")
    print(
        f"loss = {loss.item():.4f} (attesa ~ ln({tok.vocab_size}) = {math.log(tok.vocab_size):.4f})"
    )

    # shape e causalita' della head
    B, T, C, hs = 4, 8, 32, 16
    head = Head(C, hs, block_size=T).to(device)
    xh = torch.randn(B, T, C, device=device)
    out1 = head(xh)
    assert out1.shape == (B, T, hs)

    # alterare il FUTURO non deve cambiare il passato
    xh2 = xh.clone()
    xh2[:, T // 2 :] = torch.randn(B, T - T // 2, C, device=device)
    out2 = head(xh2)
    assert torch.allclose(out1[:, : T // 2], out2[:, : T // 2], atol=1e-6)
    print("Head: shape ok, causalita' ok")

    # MultiHeadAttention
    n_head = 4
    mha = MultiHeadAttention(C, n_head, block_size=T).to(device)
    o1 = mha(xh)
    assert o1.shape == (B, T, C)  # larghezza del residual stream preservata

    o2 = mha(xh2)  # stesso futuro alterato di prima
    assert torch.allclose(o1[:, : T // 2], o2[:, : T // 2], atol=1e-6)

    # 3*C^2 per q,k,v + C^2 + C per la proiezione, INDIPENDENTE da n_head
    n_params = sum(p.numel() for p in mha.parameters())
    assert n_params == 4 * C * C + C, n_params
    print(f"MHA: shape ok, causalita' ok, {n_params} parametri (= 4C^2 + C)")

    # LayerNorm: identica a quella di torch
    ln = LayerNorm(C).to(device)
    assert torch.allclose(
        ln(xh), F.layer_norm(xh, (C,), ln.weight, ln.bias, ln.eps), atol=1e-6
    )

    # e la proprieta' che la definisce (vale finche' weight=1, bias=0)
    normed = ln(xh)
    assert normed.mean(dim=-1).abs().max() < 1e-5
    assert (normed.std(dim=-1, unbiased=False) - 1).abs().max() < 1e-3
    print("LayerNorm: identica a F.layer_norm, media 0 / var 1 ok")

    # MLP: shape, parametri, e NESSUNA comunicazione tra token
    mlp = MLP(C).to(device)
    assert mlp(xh).shape == (B, T, C)
    assert sum(p.numel() for p in mlp.parameters()) == 8 * C * C + 5 * C

    xh3 = xh.clone()
    xh3[:, 3] = torch.randn(B, C, device=device)  # cambia SOLO la posizione 3
    m1, m3 = mlp(xh), mlp(xh3)
    assert torch.allclose(m1[:, :3], m3[:, :3], atol=1e-6)
    assert torch.allclose(m1[:, 4:], m3[:, 4:], atol=1e-6)
    print("MLP: shape ok, per-token ok (posizione 3 alterata, le altre invariate)")

    # Block: shape, causalita', e "parte come identita'"
    block = Block(C, n_head, block_size=T).to(device)
    b1 = block(xh)
    assert b1.shape == (B, T, C)

    b2 = block(xh2)
    assert torch.allclose(b1[:, : T // 2], b2[:, : T // 2], atol=1e-6)

    delta = (b1 - xh).std().item() / xh.std().item()
    assert delta < 0.1, delta
    print(
        f"Block: shape ok, causalita' ok, modifica il residual stream del {delta:.2%}"
    )

    # GPT completo
    n_embd, n_head_g, n_layer, bs = 384, 6, 6, 256
    gpt = GPT(tok.vocab_size, n_embd, n_head_g, n_layer, bs).to(device)

    # il weight tying e' sullo STESSO tensore
    assert gpt.lm_head.weight is gpt.wte.weight

    n = sum(p.numel() for p in gpt.parameters())
    expected = (
        tok.vocab_size * n_embd  # wte (lm_head e' legata, non conta due volte)
        + bs * n_embd  # wpe
        + n_layer * (12 * n_embd**2 + 10 * n_embd)  # blocchi
        + 2 * n_embd  # ln_f
    )
    assert n == expected, (n, expected)
    print(f"GPT: {n:,} parametri (formula ok)")

    xg, yg = get_batch(train, batch_size=8, block_size=bs)
    _, lg = gpt(xg, yg)
    print(
        f"loss init = {lg.item():.4f}   (ln({tok.vocab_size}) = {math.log(tok.vocab_size):.4f})"
    )
    assert abs(lg.item() - math.log(tok.vocab_size)) < 0.4

    # generate oltre block_size non deve esplodere
    ctx = torch.tensor([tok.encode("\n")], dtype=torch.long, device=device)
    assert gpt.generate(ctx, max_new_tokens=5, top_k=10).shape == (1, 6)
    print("GPT: generate ok")

    # KV cache: prefill + decode danno gli stessi logit del forward completo
    gpt.eval()
    seq = torch.randint(tok.vocab_size, (2, 10), device=device)
    full, _ = gpt(seq)  # senza cache, tutto in una volta
    gpt.set_cache(True)
    pieces = [gpt(seq[:, :7])[0]] + [gpt(seq[:, i : i + 1])[0] for i in (7, 8, 9)]
    gpt.set_cache(False)
    cached = torch.cat(pieces, dim=1)
    assert torch.allclose(full, cached, atol=1e-4), (full - cached).abs().max()
    print("GPT: KV cache ok")

    # Dropout: identita' in eval, azzera ~p in train, media preservata
    drop = Dropout(0.2)
    z = torch.ones(10_000, device=device)

    drop.eval()
    assert torch.equal(drop(z), z)

    drop.train()
    out = drop(z)
    assert abs((out == 0).float().mean().item() - 0.2) < 0.02  # ~20% azzerati
    assert abs(out.mean().item() - 1.0) < 0.02  # media preservata
    print("Dropout: identita' in eval, ~20% azzerato in train, media 1.0 ok")

    # CausalSelfAttention: identica alla naive a pesi uguali, stessi parametri
    mha = MultiHeadAttention(C, 4, T).to(device)
    csa = CausalSelfAttention(C, 4, T).to(device)
    hs = C // 4
    with torch.no_grad():
        # c_attn impila [q; k; v] per righe; la testa i occupa hs*i : hs*(i+1)
        for attr, off in (("query", 0), ("key", C), ("value", 2 * C)):
            for i, h in enumerate(mha.heads):
                csa.c_attn.weight[off + i * hs : off + (i + 1) * hs] = getattr(
                    h, attr
                ).weight
        csa.proj.weight.copy_(mha.proj.weight)
        csa.proj.bias.copy_(mha.proj.bias)
    mha.eval(), csa.eval()
    assert torch.allclose(mha(xh), csa(xh), atol=1e-5)
    assert sum(p.numel() for p in csa.parameters()) == 4 * C * C + C
