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


class RMSNorm(nn.Module):
    # LayerNorm senza centratura: divide per la radice della media dei quadrati, niente media, niente bias (Zhang & Sennrich 2019)

    def __init__(self, ndim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))  # gamma, unico parametro
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # la somma dei quadrati in bf16 perde cifre: si calcola in fp32 e si torna al dtype d'ingresso
        xf = x.float()
        rms_inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # (..., 1)
        return self.weight * (xf * rms_inv).type_as(x)


class MLP(nn.Module):
    # espande 4x, non linearità, ricomprime, nessuna comunicazione tra token
    def __init__(self, n_embd: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc = Linear(n_embd, 4 * n_embd)
        self.proj = Linear(4 * n_embd, n_embd)
        self.dropout = Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class SwiGLU(nn.Module):
    # MLP con porta moltiplicativa (Shazeer 2020): proj(silu(gate(x)) * up(x)), tre matrici invece di due, niente bias come LLaMA
    # hidden = 8/3 C invece di 4C: tre matrici C x hidden costano come le due 4C di prima (8C^2)
    def __init__(self, n_embd: int, dropout: float = 0.0, multiple_of: int = 64) -> None:
        super().__init__()
        # arrotondato a un multiplo di 64, i tensor core lavorano a tile
        self.hidden = multiple_of * math.ceil(8 * n_embd / 3 / multiple_of)
        self.fc = Linear(n_embd, 2 * self.hidden, bias=False)  # gate e up in un matmul solo, come c_attn
        self.proj = Linear(self.hidden, n_embd, bias=False)
        self.dropout = Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.fc(x).chunk(2, dim=-1)  # (..., hidden) ciascuno
        return self.dropout(self.proj(F.silu(gate) * up))


class MoE(nn.Module):
    # Mixture of Experts: n_expert SwiGLU indipendenti, un router sceglie i top_k per ogni token
    # parametri x n_expert, calcolo per token x top_k (Shazeer 2017, Switch 2021, Mixtral 2024)
    def __init__(
        self, n_embd: int, n_expert: int, top_k: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        assert 1 <= top_k <= n_expert
        self.n_expert, self.top_k = n_expert, top_k
        self.router = Linear(n_embd, n_expert, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(n_embd, dropout) for _ in range(n_expert)]
        )
        self.aux_loss = torch.zeros(())  # load balancing, lo somma GPT.forward
        self.load = torch.zeros(n_expert)  # frazione di token per esperto, solo per i log

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x = x.reshape(B * T, C)  # il routing e' per token: B e T non contano piu'
        probs = F.softmax(self.router(x), dim=-1, dtype=torch.float32)  # (N, E)
        w, idx = probs.topk(self.top_k, dim=-1)  # (N, k) pesi e indici degli esperti scelti
        w = (w / w.sum(dim=-1, keepdim=True)).type_as(x)  # rinormalizzati sui k scelti

        # ponytail: un ciclo sugli esperti con gather/scatter, niente grouped GEMM, shape dinamiche per compile
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            tok, slot = (idx == e).nonzero(as_tuple=True)  # quali token hanno scelto e, e in quale dei k posti
            out.index_add_(0, tok, expert(x[tok]) * w[tok, slot, None])

        # Switch Transformer: E * sum_e f_e * P_e, minima (=1) quando il carico e' uniforme
        # f_e = quota di assegnazioni andate a e (conteggio, non derivabile), P_e = prob media (derivabile)
        f = F.one_hot(idx, self.n_expert).float().mean(dim=(0, 1))  # (E,), somma 1
        P = probs.mean(dim=0)  # (E,), somma 1
        self.aux_loss = self.n_expert * (f * P).sum()
        self.load = f.detach()
        return out.reshape(B, T, C)


class Block(nn.Module):
    # un blocco transformer, comunicazione e computazione pre-ln

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
        n_expert: int = 0,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        # self.attn = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = RMSNorm(n_embd)
        # n_expert=0: MLP densa, altrimenti MoE con lo stesso SwiGLU come esperto
        self.mlp = (
            MoE(n_embd, n_expert, top_k, dropout) if n_expert else SwiGLU(n_embd, dropout)
        )

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
        n_expert: int = 0,
        top_k: int = 2,
        aux_coef: float = 0.01,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.aux_coef = aux_coef

        self.wte = Embedding(vocab_size, n_embd)  # token -> vector
        # niente wpe: la posizione la mette RoPE dentro l'attention, ruotando q e k
        self.drop = Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                Block(n_embd, n_head, block_size, dropout, n_expert, top_k)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = RMSNorm(n_embd)  # obbligatoria in pre-LN
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
        assert T <= self.block_size, (
            f"sequenza di {T} token, block_size è {self.block_size}"
        )

        x = self.drop(self.wte(idx))  # (B, T) -> (B, T, C)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, C)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        # la aux loss solo in training: in eval la loss resta cross entropy pura, confrontabile col denso e in bpb
        moes = [b.mlp for b in self.blocks if isinstance(b.mlp, MoE)]
        if moes and self.training:
            loss = loss + self.aux_coef * sum(m.aux_loss for m in moes) / len(moes)
        return logits, loss

    def set_cache(self, enabled: bool) -> None:
        # svuota sempre: una cache vecchia e' peggio di nessuna cache
        for block in self.blocks:
            block.attn.use_cache = enabled
            block.attn.cache = None
            block.attn.pos = 0

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        use_cache: bool = True,
    ) -> torch.Tensor:
        self.set_cache(use_cache)
        for i in range(max_new_tokens):
            # con la cache: il primo giro macina tutto il prompt (prefill), poi un
            # token alla volta. Senza: si rifa' tutta la finestra ogni volta.
            idx_cond = (
                (idx[:, -self.block_size :] if i == 0 else idx[:, -1:])
                if use_cache
                else idx[:, -self.block_size :]  # il modello non vede oltre
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, V)

            if repetition_penalty != 1.0:
                # CTRL (Keskar et al. 2019): si penalizzano i token gia' usciti.
                # Divisione se il logit e' positivo, moltiplicazione se negativo,
                # cosi' la penalita' abbassa la probabilita' in tutti e due i casi.
                seen = torch.zeros_like(logits, dtype=torch.bool)
                seen.scatter_(1, idx, True)
                logits = torch.where(
                    seen,
                    torch.where(
                        logits > 0,
                        logits / repetition_penalty,
                        logits * repetition_penalty,
                    ),
                    logits,
                )

            logits = logits / temperature

            if top_k is not None:  # taglia a un numero FISSO di candidati
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            if top_p is not None:
                # nucleus (Holtzman et al. 2019): tiene i token piu' probabili
                # finche' la loro massa non copre top_p. La coda tagliata cambia
                # taglia a ogni passo, ed e' esattamente il punto: dove il modello
                # e' sicuro restano pochi candidati, dove e' incerto ne restano molti
                srt, order = F.softmax(logits, dim=-1).sort(dim=-1, descending=True)
                # cum - srt e' la massa PRIMA di questo token: cosi' il primo
                # candidato sopravvive sempre, anche se da solo supera top_p
                drop_srt = (srt.cumsum(-1) - srt) > top_p
                drop = torch.zeros_like(drop_srt).scatter(1, order, drop_srt)
                logits = logits.masked_fill(drop, -float("inf"))

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.set_cache(False)  # il modello torna stateless, la memoria si libera
        return idx


def rope_tables(
    head_size: int, t0: int, T: int, device: torch.device, theta: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    # la coppia i ruota di pos * theta^(-2i/hs): le prime coppie girano veloci
    # (distanze corte), le ultime lentissime (contesto lungo). Nessun parametro.
    inv_freq = theta ** (-torch.arange(0, head_size, 2, device=device) / head_size)
    ang = torch.arange(t0, t0 + T, device=device)[:, None] * inv_freq[None, :]
    return ang.cos(), ang.sin()  # (T, hs/2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, nh, T, hs). Una rotazione 2D per ogni coppia (i, i + hs/2), di un
    # angolo proporzionale alla posizione. Cosi' q @ k.T dipende solo dalla
    # DIFFERENZA delle due posizioni: e' questo che rende la cache traslabile.
    x1, x2 = x.chunk(2, dim=-1)  # (B, nh, T, hs/2)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


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
        self.pos = 0  # token gia' visti: NON e' la lunghezza della cache, che si tronca

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)  # 3 * (B, T, C)

        # (B, T, C) -> (B, T, nh, hx) -> (B, nh, T, hs): la head diventa una dim di batch
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        # posizione assoluta da cui ripartono questi token. Deve continuare a
        # crescere anche quando la cache si tronca: se la bloccassi a block_size,
        # le q resterebbero indietro rispetto alle k gia' in cache e le distanze
        # relative diventerebbero negative.
        # pos si muove SOLO in decode: in training e' costante, e torch.compile
        # tratta gli interi di un nn.Module come statici -> ogni valore nuovo era
        # una ricompilazione, fino a sbattere contro recompile_limit
        if self.use_cache:
            t0 = self.pos
            self.pos += T
        else:
            t0 = 0
        cos, sin = rope_tables(hs, t0, T, x.device)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)  # v NON si ruota

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

    # RMSNorm: identica a quella di torch, e la media dei quadrati diventa 1 (la media NON va a 0)
    rms = RMSNorm(C).to(device)
    assert torch.allclose(rms(xh), F.rms_norm(xh, (C,), rms.weight, rms.eps), atol=1e-6)
    rn = rms(xh + 3.0)  # sposto tutto di +3: LayerNorm lo cancellerebbe, RMSNorm no
    assert ((rn.pow(2).mean(dim=-1)) - 1).abs().max() < 1e-4
    assert rn.mean(dim=-1).min() > 0.5
    print(f"RMSNorm: identica a F.rms_norm, rms 1 ok, media di x+3 resta {rn.mean().item():.2f}")

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

    # SwiGLU: la formula scritta a mano, i parametri, e di nuovo nessuna comunicazione tra token
    sw = SwiGLU(C).to(device)
    H = sw.hidden
    assert H % 64 == 0 and H >= 8 * C / 3
    assert sum(p.numel() for p in sw.parameters()) == 3 * C * H
    Wg, Wu = sw.fc.weight[:H], sw.fc.weight[H:]
    g = xh @ Wg.T
    ref = (g * torch.sigmoid(g) * (xh @ Wu.T)) @ sw.proj.weight.T  # silu(g) = g * sigmoid(g)
    assert torch.allclose(sw(xh), ref, atol=1e-6)
    s1, s3 = sw(xh), sw(xh3)
    assert torch.allclose(s1[:, :3], s3[:, :3], atol=1e-6)
    assert torch.allclose(s1[:, 4:], s3[:, 4:], atol=1e-6)
    print(f"SwiGLU: formula ok, hidden {H} (8/3 C = {8 * C / 3:.1f}), {3 * C * H} parametri, per-token ok")

    # MoE: con un esperto solo e' esattamente quell'esperto
    moe1 = MoE(C, n_expert=1, top_k=1).to(device)
    assert torch.allclose(moe1(xh), moe1.experts[0](xh), atol=1e-6)

    # MoE: confronto col calcolo ingenuo token per token, somma pesata dei k esperti scelti
    E, k = 4, 2
    moe = MoE(C, n_expert=E, top_k=k).to(device)
    mo = moe(xh)
    xf = xh.reshape(-1, C)
    pr = F.softmax(moe.router(xf), dim=-1)
    ref = torch.zeros_like(xf)
    for n in range(xf.size(0)):
        wn, en = pr[n].topk(k)
        for wi, ei in zip(wn / wn.sum(), en):
            ref[n] += wi * moe.experts[ei](xf[n : n + 1])[0]
    assert torch.allclose(mo, ref.view(B, T, C), atol=1e-6)
    assert sum(p.numel() for p in moe.parameters()) == E * 3 * C * H + C * E

    # per-token anche lui: il routing guarda solo il proprio vettore
    mo3 = moe(xh3)
    assert torch.allclose(mo[:, :3], mo3[:, :3], atol=1e-6)
    assert torch.allclose(mo[:, 4:], mo3[:, 4:], atol=1e-6)

    # il gradiente arriva al router attraverso i pesi w e la aux loss
    (moe(xh).sum() + moe.aux_loss).backward()
    assert moe.router.weight.grad.abs().sum() > 0
    assert all(e.fc.weight.grad is not None for e in moe.experts)

    # aux loss: ~1 col carico uniforme dell'init, n_expert quando collassa tutto su un esperto
    assert abs(moe.aux_loss.item() - 1) < 0.2, moe.aux_loss.item()
    assert torch.isclose(moe.load.sum(), torch.tensor(1.0, device=device))
    moec = MoE(C, n_expert=E, top_k=1).to(device)
    with torch.no_grad():
        moec.router.weight.zero_()
        moec.router.weight[0] = 1.0
    moec(xh.abs() + 5.0)  # sum(x) grande e positiva: l'esperto 0 vince ovunque
    assert moec.load[0] == 1.0 and abs(moec.aux_loss.item() - E) < 1e-3
    print(
        f"MoE: 1 esperto = SwiGLU, uguale al calcolo ingenuo, per-token ok, "
        f"aux {moe.aux_loss.item():.3f} uniforme / {moec.aux_loss.item():.3f} collassata"
    )

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
        + n_layer
        * (
            4 * n_embd**2 + n_embd  # attention: c_attn 3C^2 + proj C^2 + C
            + 3 * n_embd * gpt.blocks[0].mlp.hidden  # SwiGLU, niente bias
            + 2 * n_embd  # ln1, ln2 (RMSNorm solo gamma, RoPE nessun parametro)
        )
        + n_embd  # ln_f
    )
    assert n == expected, (n, expected)
    print(f"GPT: {n:,} parametri (formula ok)")

    # GPT MoE: ogni blocco ha n_expert SwiGLU al posto di uno, piu' il router C x E
    gm = GPT(tok.vocab_size, n_embd, n_head_g, n_layer, bs, n_expert=8, top_k=2).to(device)
    Hg = gpt.blocks[0].mlp.hidden
    nm = sum(p.numel() for p in gm.parameters())
    assert nm == n + n_layer * (7 * 3 * n_embd * Hg + 8 * n_embd), nm
    gm.train()
    _, lm = gm(*get_batch(train, batch_size=8, block_size=bs))
    gm.eval()
    _, le = gm(*get_batch(train, batch_size=8, block_size=bs))
    print(f"GPT MoE 8x top-2: {nm:,} parametri (formula ok), loss train {lm.item():.4f} (con aux) eval {le.item():.4f}")

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

    # top-p: con p minuscolo sopravvive solo il token piu' probabile, quindi deve
    # dare esattamente lo stesso testo di top_k=1. Se lo shift `cum - srt` fosse
    # sbagliato taglierebbe anche il primo e multinomial esploderebbe su tutti -inf
    ctx_p = torch.tensor([tok.encode("\n")], dtype=torch.long, device=device)
    torch.manual_seed(0)
    greedy = gpt.generate(ctx_p, 20, top_k=1)
    torch.manual_seed(0)
    nucleus = gpt.generate(ctx_p, 20, top_k=None, top_p=1e-6)
    assert torch.equal(greedy, nucleus), (greedy, nucleus)

    # repetition_penalty enorme: nessun token puo' uscire due volte
    torch.manual_seed(0)
    rep = gpt.generate(ctx_p, 20, top_k=1, repetition_penalty=1e6)[0].tolist()
    assert len(set(rep)) == len(rep), rep
    print("sampling: top-p e repetition_penalty ok")

    # RoPE: q . k dipende SOLO dalla distanza, non dalla posizione assoluta
    qv, kv = (
        torch.randn(1, 1, 1, 64, device=device),
        torch.randn(1, 1, 1, 64, device=device),
    )

    def rope_dot(m: int, n: int) -> torch.Tensor:
        return (
            apply_rope(qv, *rope_tables(64, m, 1, qv.device))
            * apply_rope(kv, *rope_tables(64, n, 1, kv.device))
        ).sum()

    assert torch.allclose(rope_dot(5, 3), rope_dot(105, 103), atol=1e-4)
    assert not torch.allclose(rope_dot(5, 3), rope_dot(5, 4), atol=1e-4)
    print("RoPE: invariante per traslazione ok")

    # KV cache oltre block_size: con RoPE la finestra tagliata deve dare ESATTAMENTE
    # i logit del path senza cache, che la finestra la ricalcola da zero ogni volta
    small = GPT(tok.vocab_size, 64, 4, 2, 16).to(device)
    small.eval()
    seq = torch.randint(tok.vocab_size, (1, 1), device=device)
    plain = []
    for _ in range(40):  # senza cache: riparte dagli ultimi 16 token ogni volta
        lg, _ = small(seq[:, -16:])
        plain.append(lg[:, -1])
        seq = torch.cat((seq, lg[:, -1].argmax(-1, keepdim=True)), dim=1)
    small.set_cache(True)
    cached = [small(seq[:, :1])[0][:, -1]]
    for t in range(1, 40):
        cached.append(small(seq[:, t : t + 1])[0][:, -1])
    small.set_cache(False)
    err = max((a - b).abs().max().item() for a, b in zip(plain, cached))
    assert err < 1e-3, err  # 40 token su una finestra di 16: scorre 24 volte
    print(f"KV cache: finestra scorrevole esatta (err max {err:.1e})")

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

    # CausalSelfAttention: stessi parametri della naive (4C^2 + C), ma da RoPE in poi NON e' piu' la stessa funzione
    csa = CausalSelfAttention(C, 4, T).to(device)
    csa.eval()
    assert csa(xh).shape == (B, T, C)
    assert torch.allclose(csa(xh)[:, : T // 2], csa(xh2)[:, : T // 2], atol=1e-6)
    assert sum(p.numel() for p in csa.parameters()) == 4 * C * C + C
