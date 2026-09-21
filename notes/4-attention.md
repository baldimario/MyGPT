Il trucco matematico dietro l'attention

prima di scrivere la self-attention capiamo prima il meccanismo senza parametri, l'attention è una cosa sola travestita, una media pesata sul passato, tutto il resto è come si calcolano i pesi.

Il problema

La posizione t deve raccogliere le informazioni dalle posizioni 0..t mai da t+1 in poi.

La cos apiù stupida che possiamo fare è la media, x[b,t] diventa la media di tutti i vettori da 0 a t, è un canale di comunicazione debolissimo, perde l'ordine, perde tutto ma è comunicaizone

```
import torch
import torch.nn.functional as F

torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)

# il loop esplicito, lento e ovvio
xbow = torch.zeros(B, T, C)
for b in range(B):
    for t in range(T):
        xbow[b, t] = x[b, : t + 1].mean(dim=0)  # media di tutto il passato incluso t

# la stessa cosa come moltiplicazione di matrici
wei = torch.tril(torch.ones(T, T))  # triangolare inferiore di 1
wei = wei / wei.sum(dim=1, keepdim=True)  # ogni riga somma a 1
xbow2 = wei @ x  # (T,T) @ (B,T,C) -> (B,T,C)

# la stessa cosa passando da una softmax
tril = torch.tril(torch.ones(T, T))
wei3 = torch.zeros(T, T)  # "affinita'" tutte uguali
wei3 = wei3.masked_fill(tril == 0, float("-inf"))  # il futuro e' proibito
wei3 = F.softmax(wei3, dim=-1)  # -> stessa matrice di v2
xbow3 = wei3 @ x

assert torch.allclose(xbow, xbow2, atol=1e-6)
assert torch.allclose(xbow, xbow3, atol=1e-6)

torch.set_printoptions(precision=3, sci_mode=False)
print(wei)
```

come output da
```
tensor([[1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.500, 0.500, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.333, 0.333, 0.333, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.250, 0.250, 0.250, 0.250, 0.000, 0.000, 0.000, 0.000],
        [0.200, 0.200, 0.200, 0.200, 0.200, 0.000, 0.000, 0.000],
        [0.167, 0.167, 0.167, 0.167, 0.167, 0.167, 0.000, 0.000],
        [0.143, 0.143, 0.143, 0.143, 0.143, 0.143, 0.143, 0.000],
        [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]])

```

questa matrice è l'oggetto più importante dello step, wei ha shape (T, T)

wei[i, j] = quanto la posizione i (che sta predicendo) pesa la posizione j (che sta fornendo informazione)

- riga i = la "ricetta" con cui la posizione i si ricostruisce dal passato. somma a 1.
- triangolare inferiore = causalità, wei[2, 5] = 0 significa che l aposizione 2 non sa nulla della 5, no è un'iottimizzazione, è il vincolo che rende leggittimi i B*T esempi di training
- qui i pesi sono uniformi sul passato, nell'attention vera saranno calcolati dai dati, la posizione i deciderà lei a chi dare personaggi

quindi ci sono delle cose da capire

- aggregazione pesata = matmul. questa equivalenza è tutto il trucco, il loop O(T^2) della prima implementazione, la seconda è un singolo matmul che usa la GPU con i tensor core, stessi bit, ordini di grandezza di differenza, ogni volta che vediamo "ogni elemento guarda tutti gli altri con dei pesi" sotto c'è un matmul
- il broadcasting del matmul. wei è (T, T), x è (B, T, C), torch tratta il prodotto tra matrici @ come batched matmul, usa le ultime due dimensioni come matrici e fa broadcast di tutto il resto, quindi (T, T) viene espanso a (B, T, T) se usiamo pesi diversi per ogni sequenza il broadcasting sparisce
- Perché passare per sofrmax se la terza (ultima) versione è come la seconda? perché nella seconda i pesi sono cablati 1/(t+1) sempre mentre nell'ultima implementazione la matrice di partenza è torch.zeros, affintà tutte uguali, ma potebbe essere qualunque cosa, la softmax prende una matrice di numeri reali arbitrari e la trasforma in una distribuzione di probabilità sul passato, la media uniforme è solo il caso particolare di "tutte le affinità identiche"

da qui in poi la domanda chiave è "da dove vengono quei numeri?", spoiler:  query * key, non c'è altro.

- Il -inf non è un valore grande, è il valore giusto perché le probabilità sono logaritmiche e exp(-inf) = 0 esattamente, quindi il futuro riceve peso zero ed è i l punto cruciale, non entra nemmeno nel denominatore della softmax, la normalizzazione coinvolge solo il passato, se mascherassimo con -1e9 funzionerebbe quasi ma in float16 -1e9 è -inf comunque e in bf16 rischiamo NaN quando una riga è tutta mascherata, usare -inf lascia questa cosa da gestire alla softmax matematicamente

E facciamo attenzione a dim=-1, normalizziamo lungo le colonne dentro ogni riga, cioè "la posizione i distribuisce la sua attenzione tra i vari j" con dim=0 normalizzeremmo le colonne, il modello non impara


ovviamente il limite di questa versione è che la media non ha un ordine, mean(x0, x1, x2) == mean(x2, x0, x1), il meccanismo di aggregazione è permutazione-equivalente sul passato sa cosa c'era prima ma non sa l'ordine, vale anche per la self-attention vera, con Q/K/V, è esattamente il motivo per cui serviranno embedding posizionali (wpe), l'informaizone sulla posizione va iniettata nei vettori prima dell'attention perché l'attention da sola non l apossiede

---

Il layer Linear

perché (out, in) e non (in, out)? sembra scomodo, costringe ad una matrice trasposta con .T nel forward pass ma c'è un motivo concreto oltre alla convenzione, allo step 6 legheremo i pesi dell'embedding e del layer finale wte.weight è (V, C) e lm_head.weight deve essere la stessa matrice, con la convenzione (out, in), Linear(C, V).weight è (V, C), combaciano senza trasposizioni, con l'altra convenzione dovremmo trasporre e perderemmo la condivisione del tensore

(E il .T nel forward non costa, è come una view, cambia solo gli stride, non c'è copia, il kernel di matmul gestisce nativamente grandi trasposti)

Il bias parte da zero, non da random. Il peso deve rompere la simmetria tra neuroni, se tutti i pesi fossero uguali tutti i neuroni calcolerebbero la stessa cosa e resterebbero identici per sempre, il bias no, l asimmetria è già rotta dai pesi e partire da zero significa nessuno spostamento a priori.

Query, Key, Value

finora wei era una costante, ora la calcoliamo dai dati, ogni token emette tre vettori, proiettando il suo embedding con tre Linear diversi

| | domanda a cui rispondere |
|---|---|
| query q | cosa sto cercando nel passato? |
| key k | cosa offro, per cosa sono trovabile |
| value v | cosa comunico se mi scegli? |

l'affinità tra la posizione i e la j è il prodotto scalare q_i*k_j, alto quando quello che i cerca assomiglia a quello che j offre.

Facciamo un esempio concreto, nella frase "il gatto nero dorme" il token "dorme" è un verbo e ha bisogno di sapere chi è il soggetto, la sua query codifica qualcosa come "cerco un nome singolare alla mia sinistra", il token "gatto" ha una key che dice "sono un nome singolare maschile", prodotto scalare alto -> "dorme" guarda "gatto"", e si porta via il suo value, che magari codifica "animale, terza persona".

Perché key e value separati? perché come ti si trova non è cosa comunichi, "gatto" è trovabile come "nome singolare" (key), ma quello che trasmette è "felino domestico" (value), disaccoppiare l'indirizzo dal contenuto è ciò che rende l'attention un meccanismo di ricerca invece di una semplice correlazione, è il dettaglio più importante

quindi ora possiamo lavorare sul layer Head, alcuni dettagli sull'implementazione

register_buffer, tril non è un parametro, nessun gradiente, non va addestrato, ma deve spostarsi su GPU con .to(device) e finire nello state_dict(), questo è  esattamente il caso d'uso dei buffer, se lo mettessimo come attributo normale (self.tril = torch.tril(...)) resterebbe su CPU mentre il resto va su GPU e avremo a runtime degli errori

self.tril[:T, :T] non self.tril, il buffer è (block_size, block_size) ma in genrazioni partiamo da un contesto di 1 token e crediamo, senza lo slicing crashiamo appena T < block_size, cioè sempre durante generate


transpose(-2, -1) non .T, su tensori a più di 2 dimensioni .T è deprecato, transpose(-2, -1) scambia le ultime due dimensioni e lascia stare il batch, che è praticamente ciò che vogliamo (B, T, hs) -> (B, hs, T)

bias=False, su q/k/v GPT-2 e GPT-3 hanno i bias, lo togliamo perché prima di un prodotto scalare aggiungere quei parametri danno un guadagno trascurabile ed infatti modelli come LLaMA, PaLM li tolgono

L' head_size * -0.5 è il dettaglio che fa la differenza, non è un valore cosmetico, con q e k a compoenti indipendenti di varianza ~1, il prodotto scalare su head_size dimensioni è una somma di head_size termini indipendenti, la sua varianza è head_size, quindi la scala cresce come sqrt(head_size)

con head_size=64 i logits partono con deviazione standard 8, cioè sparsi tra -24 e 24, la softmax su valori così è quasi un one-hot encoding, all'inizializzazione prima di imparare qualunque cosa ogni token guarda solo un tokenm scelto a caso e una softmax satura ha gradiente ~0, il modello nace bloccato nell'apprendimento

Ispezionare la matrice tril verifica che abbiamo scritto la maschera, questo verifica che funziona, riscriviamo da capo la seconda metà della sequenza e le uscite della prima metà devono essere identich ebit per bit, se sbagliamo il verso della maschera il dim della softmax o l'ordine del traspose lo capiamo, facciamo delgi assert per modifiche future all'attention così restiamo sicuri


```
uv run src/mygpt/model.py
parametri = 4,356
logits = (4, 8, 66)
loss = 4.1913 (attesa ~ ln(66) = 4.1897)
Head: shape ok, causalita' ok
```


