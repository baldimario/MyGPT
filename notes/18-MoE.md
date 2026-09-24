Il MoE con 8 esperti e top-2 è meglio del dense a parità di step, a parità di tempo ogni step costa 1.8 volte tanto


| 5000 it, stesa config | val | bpb | parametri (attivi) | ms/it | gap train/val |
|---|---|---|---|---|---|
| dense RMSNorm + SwiGLU | 3.3878 | 1.333 | 13.8M | 64.1 | 0.13 |
| MoE 8 esperti, top-2 | 3.2496 | 1.279 | 63.3M (~21M) | 115.9 | 0.16 |

-0.138 nat, cioè -4.1% di bpb, è dieci volte il salto di RMSNorm + SwiGLU e ben sopra il rumore tra seed, il vantaggio cresce per tutta la run: -0.08 a 1000 step, -0.11 a 3000, -0.14 alla fine, gli esperti continuano a separarsi e speciallizarsi

il confronto non è ancora alla pari perché primo, per token il MoE usa circa 21M paraemtri attivi contro 13.8 del denso, la MLP gira due volte, secondo nel tempo di 5000 step di MoE il denso ne fa circa 9000, il dense a 20k step arriva a 3.131, quindi il dense da 9000 step finirebbe forse tra 3.25 e 3.30, è una stima, non una misura

il gap train/val sale da 0.13 a 0.16 i parametri in più cominciano a memorizzare come ci si aspetta con 63M parametri e i 145M token, con tuti gli shard dovrebbe calare

La velocità è stabile a 115.9 ms/it, i valori relativi all'uniforme 1.0 significa 1/8 dei token

```
L0 aux 1.033 | 0.71 0.72 1.91 1.22 1.05 0.99 0.97 0.42
L1 aux 0.999 | 1.20 1.03 1.17 0.80 0.97 1.10 0.85 0.87
L2 aux 1.004 | 0.90 0.88 1.06 1.12 1.15 1.08 0.90 0.92
L3 aux 1.038 | 0.73 0.79 1.19 1.30 1.34 0.75 0.90 1.00
L4 aux 1.035 | 1.23 0.92 0.68 0.88 0.88 0.95 1.21 1.26
L5 aux 1.090 | 0.39 0.95 0.88 1.49 1.00 1.53 0.78 0.99
```

non c'è nessun collaso, tutti gli esperti lavorano e la aux resta tra 1.00 e 1.09, lo squilibrio è stabile, non va alla deriva. Nel layer 0 l'esperto 2 prende quasi il doppio della sua quota e l'esperto 7 circa 0.4, dal passo 1000 fino alla fine, è il compromesso che trovano, la loss principale vuole specializzare, e la aux vuole uniformare


gli squilibri stanno ai bordi del modello, nel layer 0 il router vede quasi solo embedding del token, quindi instrada per tipo di token, alcuni tipo come la punteggiatura o i pezzi di parola frequenti sono più comuni di altri, è un'interpretazione plausibile che non abbiamo verificato, si potrebbe controllare quali token finiscono nell'esperto 2 del layer 0

| run | dati | step | val | bpb |
|---|---|---|---|---|
| denso LayerNorm + GELU | 1 shard, 145M token | 20k | 3.1310 | 1.232 |
| MoE 8 esperti top-2 | 1 shard | 5k | 3.2496 | 1.279 |
| MoE 8 esperti top-2 | 10 shard, 1.37B token | 60k | 2.7562 | 1.085 |

-12% di bpb rispetto al miglior dense, ci contribuiscono insieme il corpus 9 volte più grande, i 3 volte più step, il MoE e RMSNorm + SwiGLU

La val scendeva ancora fino all'ultimo step, a questa scala il modello non è saturo, con più step o un modello più grande c'è ancora margine

Il gap tra train e val è rimasto stabile intorno a 0.25 per tutta la run, compatibile con l'idea che il train sia più facile della val e non con l'overfitting

Il carico degli esperti è sano fino alla fine, aux tra 1.00 e 1.12, in alcuni layer rest un esperto molto richiesto, intorno a 1.7x

In generazione 170 token/s a batch 1, contro 283 del denso, il dispatch in python sugli 8 esperti pesa di più quiando c'è un solo token

Campione con prompt "Il procesore Pentium 4" temperatura 0.7 tok-k 40 e repetition penality 1.15

```
Il processore Pentium 4 MIPS era stato progettato per essere il primo processore di tipo 2x2. L'architettura della 430 venne migliorata, e fu introdotto un nuovo processore con architettura a 54 bit.

La versione per Macintosh del Pentium 4 MIPS fu realizzata nel 1969. Fu l'ultima versione per Macintosh ad essere prodotta negli Stati Uniti...
```

L'italiano è fludio e il registro enciclopedico è giusto, il lessico è quello del dominio (architettura, bit, Macintosh, Windows), ma i fatti sono inventati "MIPS", 1969, 54 bit, a 21M paraemtri attivi il modello ha imparato la forma della conoscenza, non ancora la conoscenza

Il prossimo passo sarebbe il modello più grande, eper esempio 512/8/8 con 8 esperti, circa 150M parametri totali e 30M attivi
