SwiGLU prende il posto dell'MLP con GELU nel Block

La MLP classica é `proj(σ(fc(x)))` una proiezione, una non linearità applicata elemento per elemento, poi una seconda propiezione, ogni neurone nascosto decide da solo quanto passa del segnale guardando solo il proprio valore

Le Gated Linear Untis (Dauphin et al., 2017) separano due ruoli, cosa passa e quanto passa

```
GLU(x) = (x*W_up) ⊙ σ(x·W_gate)
```

up è il contenuto, ua proiezione lineare, gate è una seconda proiezione indipendente ch eodpo la non linearità fa da valvola, mentre ⊙ è il prodotto elemento per elemento ovviamente, il neurone i sce sempore come up_i * valcola_i

Shazeer (2020, GLU Variants Improve Transformer) ha provato diverse funzioni al posto di σ, con Swish/SiLU si ottiene SwiGLU, che è risultata la migliore a parità di calcolo, il paper è quello della famosa frase "We offer no explaination as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence." da allora la usano tutti come LLaMA, Mistra, Qwen e PaLM

SiLU

```
SiLU(x) = x * sigmoid(x)
```

è parente stretta della GELU, che è `x * Φ(x)` stessa idea di "x pesato per quanto è positivo" con la sigmoide al posto della CDF gaussiana, le due cuirve sono quasi sovrapposte, per i valori negativi SiLU non taglia a zero, scende leggermente sotto (minimo circa -0.28 in x~-1.28) e il gradiente resta non nullo, il vantaggio di SwiGLU non viene quindi da SiLU contro GELU, viene dalla porta

ma perché la porta aiuta?

Interazioni moltiplicative, con la GELU ogni neurone nascosto è una funzione di una sola proiezione di x, con la porta è il prodotto di due proiezioni diverse, cioè un termine di secondo grado in x (ogni coppia x_a*x_b compare), la rete esprime facilmente condizioni del tipo "passa il contenuto A solo se c'è il contesto B" che una MLP con una sola non linearità deve approssimare con molti neuroni

Lettura chiave-valore più pulita, la MLP si comporta come una memoria chiave-valore, con SwiGLU i ruoli sono espliciti, gate riconosce il pattern (chiave), up prepara il contenuto, proj lo riscrive nel residual stream


il gradiente passa da due strade `∂(g·u)/∂u` e `∂/∂g = u` anche dove l aporta è quasi chiusa il flusso non si azzera di colpo, il training è più liscio

c'è un contro sui parametri però, perché 8/3?

ora ci sono tre matrici invece di due, con hidden = 4C costerebbe 12C^2 cioè il 50% in più ma è un confronto truccato

per restare a parità di parametri e FLOP

```
3 * C * H = 8C^2 -> H = 8C/3
```

con C=384 viene H = 1024 esatto, il blocco passa da 8C^2 + 5C a C^2 tondi e dal modello escono i 5C di bias per blocco, da 10660992 a 10649472 parametri

l'arrotondamento a multipli di 64 serve ai tensor core che lavorano a tile, una matrice 384*1000 costa quasi quanto una 384*1024 e sfrutta meglio la GPU, LLaMA fa lo stesso con multiple_of=256, su C piccoli l'arrotondamento pesa, con C=32, 85.3 diventa 128 cioè 4C


---

RMSNorm + SwiGLU sulla baseline da un piccolo guadagno, il margine è vicino al rumore però, su 5000 iterazioni


 | | val | bpb | parametri | velocità |
 |---|---|---|---|---|
 | LayerNorm + GELU (baseline) | 3.4006 @ 4750 | 1.338 | 13786368 | - |
 | RMSNorm + SwiGLU | 3.3878 @ 5000 | 1.333 | 13769.859 | 64.1 ms/it, 256k tok/s |

Il guadagno è piccolo: -0.13 nat, cioè -0.4% bpb con un seed per lato, due run identiche ocn seed diversi di solito differiscono già di qualch emillesimo, ci sono però due segnali coerenti a favore, nella run nuova ha il minimo all'ultimo step, quindi sarebbe sceso ancora mentre l abaseline si era fermata a 4750, lavora con meno parametri

Il risultato era atteso, queste due modifiche danno pochi punti percentuali anche su modalli grandi e a 13.8M parametri su 0.57 epoche siamo lontani dal regime in cui contano davvero, il vantaggio concreto è stessa qualità, stessa velocità, un blocco uguale a quello di LLaMA. Il MoE partirà da lì

Il testo generato si comporta come prima, struttura wikipedia riconoscibile, qualche tabella wiki sporca, ripetizioni come "sistemi di ricerca...", il sampling di train.py usa top-k 40 senza repetition penality quindi le ripetizioni sono attese



