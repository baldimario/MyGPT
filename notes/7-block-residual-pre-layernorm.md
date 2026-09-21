Layer Block, residual block e pre-layernorm

La connessione residuale sembra banale, è la ragione per cui esistono le reti profonde

```
x = x + f(x)
```

l'argomento del gradiente deriva quell'espressione

```
d/dx [x+f(x)] = 1 + f'(x)
```

quell'1 è tutto il senso. In una rete senza residual il gradiente attraversa n layer moltiplicandosi: f'_n * f'_{n-1} * ... * f'_1, se i fattori valgono in media 0.8, dopo 12 layer avremo 0.8^12 = 0.069, il primo layer riceve il 7% del segnale, con 96 layer (GPT-3) avremo 1.5e10 zero, e se i fattori valgono 1.2 esplide nella direzione apposta

con i residual ogni layer contribuisce 1+f' e c'è sempre un percorso diretto lungo cui il gradiente arriva intatto all'input a qualunque layer indipendentemente da cosa facciano i blocchi in mezzo.

Il residual stream, possiamo vederlo con l'interpretabilità meccanicistica

```
                    il residual stream: un bus largo C
   ═══════════════════════════════════════════════════════════════>
      │           ▲         │           ▲        │          ▲
      │ legge     │ scrive  │ legge     │ scrive │ legge    │ scrive
      ▼           │         ▼           │        ▼          │
    ┌──────────────┐      ┌──────────────┐     ┌──────────────┐
    │  LN → Attn   │      │  LN → MLP    │     │  LN → Attn   │
    └──────────────┘      └──────────────┘     └──────────────┘
```

x non è l'attivazione del layer corrente, è un bus condiviso largo C che attraversa tutto il modello da cima a fondo. Ogni sotto blocco lo legge attraverso la sua LayerNorm, calcola qualcosa e somma il risultato, nessuno sovrascrive mai, solo alla fine lm_head legge la somma accumulata di tutti i contributi.

Ci sono due conseguenze che contano

i blocchi comunicano tra loro attraverso il bus, il blocco 3 può scrivere un'informazione in certe dimensioni che il blocco 9 legge e usa, non c'è un flusso rigido layer-per-layer, c'è una lavagna condivisa. è così che si formano i circuiti multi-layer (le induction heads sono composte da due teste in layer diversi che si coordinano proprio così)

all'inizializzazione, la rete interna è l'identità, se f(x) ~ 0 allora x + f(x) ~ x e i dati attraversano i 12 blocchi senza essere toccati, il training non parte da una funzione casuale, parte dall'identità e i blocchi si accendono gradualmente man man oche serve, questo è il vero motivo per cui impsotiamo l'init piccolo (0.02) ed è un fattore che conta.

Pre-LN vs Post-LN

Il Transformer originale (2017) metteva la LayerNorm dopo

```
x = LayerNorm(x + Attn(x)) # Post-LN originale
```

quindi la layernorm è finita sulk percorso residuale, l'autostrada del gradiente viene interrotta da una normalizzazione a ogni singolo layer, i transformer psot-LN profondi divergono se non li si tratta delicatamente con warmup del learning rate obbligatorio e sopra un certo numero di layer diventano comunque instabili, GPT-2 ha spostato la norma prima ed è lo standard da allora

```
x = x + Attn(LayerNorm(x)) # Pre-LN
```

Ora il percorso residuale è pulito, da input a output è una pura catena di somme, senza una sola operazione sopra. Il gradiente ci scorre perfettamente, la LayerNorm è finita sul ramo dove normalizza ciò che il blocco legge senza toccare ciò che il blocco riceve


il prezzo e il dettaglio che si dimentica sempre è che siccome il bus non viene mai normalizzato la sua magnitude cresce strato dopo strato (stiamo dommando 2*n_layer contribuiti) serve quindi una layernorm finale dopo l'utimo blocco e prima di lm_head, altrimenti il layer di input riceve arttivazioni di scala arbitraria, quel ln_f che vediamo in ogni implementazione di GPT non è decorativo, esiste perchè l'architettura è pre-LN , lo aggiungeremo anche noi dopo, per ora facciamo il layer Block

Nell'implementazione abbiamo due LayerNorm distinte, ln1 e ln2 che hanno parametri propri e leggono il bus in due momenti diversi (dopo la scrittura dell'attention il bus è cambiato) riusare lo stesso modulo sarebbe un bug, condivideremmo gamma/beta tra due letture che non hanno motivo di condivderli

E x = x + ..., mai x += ..., l'operazione in-place scrive sopra un tensore che l'autograd potrebbe aver salvato per il backward pass, torch a volte se ne accorge e lancia un runtime error, a volte è silente.

Lo scaling dell'inizializzazione (per ora concettuale)

C'è un raffinamento che GPT-2 introduce e che completa il ragionamento sull'identità.
Il bus riceve 2*n_layer scritture indipendenti (una per attention e una per MLP per blocco), se ognuna ha una variazione sigma^2 la varianza dle bus in fondo è circa 2*n_layer*sigma^2, cresce linearmente con la profondit, con 12 layer la scala si moltiplica per ~5, con 96 layer per ~14, il modello parte già sbilanciato e piu profondo è e peggio è

la correzione è inizializzare solo i pesi che scrivono sul bus, cioè attn.proj e mlp.proj, le ultime operazioni prima di ogni + e li inizializziamo a 

```
std = 0.02 / sqrt(2*n_layer)
```

così ogni contributo è 1/sqrt(2*n_layer) volte più piccolo, i 2*n_layer contributi si sommano e la varianza totale ritorna 0.02^2 qualunque sia la profondità, è il pezzo che rende l'inizializzazione indipendente da n_layer

non possiamo scriverlo qui perché il nostro layer Block non sa quanti layer ci sono nel modello, lo facciamo quando costruiremo la classe GPT e sfruttiamo il fatto che entrambe le proiezioni si chiamao proj

```
uv run src/mygpt/model.py
parametri = 4,356
logits = (4, 8, 66)
loss = 4.1913 (attesa ~ ln(66) = 4.1897)
Head: shape ok, causalita' ok
MHA: shape ok, causalita' ok, 4128 parametri (= 4C^2 + C)
LayerNorm: identica a F.layer_norm, media 0 / var 1 ok
MLP: shape ok, per-token ok (posizione 3 alterata, le altre invariate)
Block: shape ok, causalita' ok, modifica il residual stream del 1.41%
```

l'ultimo check è il più interessante, capiamo quanto ogni blocco perturba il bus, è la verifica sperimentale che la rete nasce come identità, se uscisse 80% l'init sarebbe sbagliato, il modello partirebbe con una funzione caotica invece dlla funzione identica, possiamo provare anche xh*100, la percentuale resta la stessa per merito della LayerNorm che rende il ramo invariante alla scala del bus.
