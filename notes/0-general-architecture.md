Useremo `torch` perché ci da solo 3 cose che ci semplificano la vita, tensori, GPU e autograd più qualche kernel ottimizzato (matmul, softmax), tutto il resto come attention, layernorm, blocchi lo scriveremo noi, è la minima astrazione per capire la matematica sotto che implementeremo noi e non avere cose automagiche sopra.
Non usiamo `tensorflow/keras` perché astrae troppo, potremmo essere tentati da usare keras.layers.MultiHeadAttention che nasconde il cuore di quello che vorremo imparare, di solito gli llm li fanno con questo perché è già bello e collaudato.
Potremmo voler usare `numpy` per qualche esercizietto tipo forward e backward pass a mano di un MLP, capire autograd, ma sono cose che ho già appreso nei corsi base della stanford come il CS231N e non voglio perdere tempo in cose che posso ripassare rileggendo i syllabus velocemente.

torch.nn.Linear, torch.nn.Embedding, torch.nnLayer.Norm possiamo volerli usare per fare prima, sono solo layer lineari xW+b e normalizzazioni, nessun concetto particolare nascosto, evitiamo nn.MultiheadAttention e nn.TransformeBlock, quelli vorrei scriverli io, probabilmente scriverà anche anche linear, embedding e layernorm per completezza.

---

Schema generale di GPT-3, che è decoder-only transformer, pre-LayerNorm, questo è il flusso completo:

```
 testo: "il gatto dorme"
      │
      ▼
┌─────────────────────┐
│  TOKENIZER (BPE)    │   str → List[int]
└─────────────────────┘
      │  idx: (B, T)          B=batch, T=context length
      ▼
┌─────────────────────┐
│ wte: Embedding      │   (V, C)  lookup table token → vettore
│ wpe: Embedding      │   (T, C)  lookup table posizione → vettore
│        x = wte[idx] + wpe[0:T]
└─────────────────────┘
      │  x: (B, T, C)         C = n_embd (dim. del modello)
      ▼
┌──────────────────────────────────────────┐
│  BLOCCO TRANSFORMER   × n_layer          │
│                                          │
│   x = x + Attn( LayerNorm(x) )           │  ← comunicazione tra token
│   x = x + MLP ( LayerNorm(x) )           │  ← computazione per token
│                                          │
└──────────────────────────────────────────┘
      │  x: (B, T, C)
      ▼
┌─────────────────────┐
│  LayerNorm finale   │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ lm_head: Linear     │   (C, V)   senza bias, pesi legati a wte
└─────────────────────┘
      │  logits: (B, T, V)
      ▼
   cross_entropy(logits, targets)     targets = idx shiftato di 1
```

Le shape non cambiano mai dentro lo stack (B, T, C) entra (B , T C) esce, per ogni blocco, il residual stream un canale di dimensione C a cui ogni blocco aggiunge qualcosa (x = x + f(x)), attention e MLP sono solo scritture su quel bus.

Ogni posizione t produce una predizione per t+1, un batch (B, T) non ci da B esempi di training, ci da B*T, il vincolo è ch ela posizione t non può vedere t+1...T è proprio il senso della maschera causale.

Ci sono due sottoblocchi:

Causal Self-Attention che è l'unico posto dove i token si parlano
x (B,T,C)  -> Linear(C, 3C) -> split -> q, k, v
          -> reshape in n_head teste: (B, nh, T, hs)  dove hs = C/nh
  att = (q @ k^T) / sqrt(hs)            (B, nh, T, T)
  att = att.masked_fill(mask==0, -inf) <- triangolare inferiore
  att = softmax(att)
  y   = att @ v
      -> riassembla (B, T, C) -> Linear(C, C) (proiezione d'uscita)

MLP, non c'è alcuna comunicazione, agisce su ogni token isolato, espande 4x, è non-lineare, ricomprime.
x -> Linear(C, 4C) -> GELU -> Linear(4C, C)

---

Ipermarametri di GPT-3 rispetto quello che addestriamo

| | n_layer | n_head | n_embd | ctx | param |
|---|---|---|---|---|---|
| GPT-3 175B | 96 | 96 | 12288 | 2048 | 175B |
| GPT-3 Small | 12 | 12 | 768 | 2048 | 125M |
| myGPT | 4-6 | 4-6 | 128-256 | 128-256 | ~1-10M |

i vincoli sono: n_embd % n_head == 0. head_size = n_embd / n_head (in GPT-3 è sempre 128 nei modelli grandi)

la differenza tra GPT-3 e GPT-2 è poca e sono quasi tutte trascurabili per capire il funzionamento, tipo contesto 2048 invece di 1024, inizializzazione diversa, attention sparsa a bande alternate rispetto quella densa nei layer pari/dispari, forse quest'ultima è l'unica grande novità ma per semplificare l'implementazione ci limiteremo a implementare l'attention densa ovunque.
GPT-3 con solo layer densi è GPT-2 più grande, la sparsa la possiamo aggiungere dopo e vedere cosa cambia.

---

Concetti da affrontare in ordine.

1. Tokenizer, partiamo da un char-level tokenizer, BPE dopo, così non ci blocchiamo all'inizio.
2. Bigram model, embedding -> Logits, niente attention, serve per avere un loop di training e sampling funzionante prima di toccare effettivamente il transformer
3. Single head self-attention, qui c'è il 70% della comprensione
4. Multi-head + proiezione
5. Blocco completo, con residual, pre-LN, MLP
6. Stack + lm_head + waight tying
7. Training loop, AdamW, warmup + cosine decayu, grad clipping
