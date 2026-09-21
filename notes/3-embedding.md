Il layer di Embedding è il primo vero layer addestrabile.

Abbiamo deciso che useremo torch proprio a basso livello, quindi NON useremo nn.Embedding, nn.Linear e nn.LayerNorm
ma uyseremo solo nn.Module, nn.Parameter, autograd per il gradiente nel passo all'indietro per la propagazione dell'errore della funzione di loss e F.cross_entropy che la scriviamo per capirla ma poi usiamo quella di torch che sicuramente sarà più stabile.

L'embedding è un Linear layer su un one-hot encoding, questa equivalenza è il motivo per cui il weight tying (più layer che condividono gli stessi pesi), funzionerà poi, quindi vale la pena scriverla matemaicamente con V=4, C=3 e token idx=2

```
one_hot(2)  @        W          =      W[2]
                 ┌ w00 w01 w02 ┐
[0 0 1 0]   @    │ w10 w11 w12 │   =  [w20 w21 w22]
                 │ w20 w21 w22 │
                 └ w30 w31 w32 ┘

```

Il one-hot seleziona una riga, è letteralmente un Linear senza bias ma calcolarlo come matmul sarebbe demenziale, con V=50257 moltiplichiamo (B*T, 50257) @ (50257, C) dove il 99.998% degli input è zero, quindi prendiamo la scorciatoia dell'indicizzazione diretta della riga.
Per quanto riguarda il gradiente `dL/dW[i]` è non nullo solo per le righe effettivamente usate e si accumula se lo stesso token compare più volte nel batch, autograd di torch lo gestisce con uno index_add sotto il cofano, quindi noi non dobbiamo fare nulla

la funzione forward non impone nessuna shape a idx, se passiamo (B, T) otteniamo (B, T, C), passiamo (T,) e otteniamo (T, C), è l'advanced indexing spiegata prima

Dobbiamo verificare due cose che sono il cuore di nn.Module
- torch.randn(...) da solo non è un parametro, va avvolto in nn.Parameter e comparirà in model.parameters(), si sposterà con .to('cuda') sulla VRAM, finirà nello state_dict(), se non fosse wrappato in nn.Parameter l'ottimizzazione non addestra nulla
- super().__init__() prima di assegnare i parametri, nn.Module.__setattr__ intercetta le assegnazioni per registrarle, se il modulo non è inizializzato lancia un errore

Perché moltiplichiamo per 0.02? Questo valore è reso da GPT-2/GPT-3, non è arbitrario, nel bigram si vde in modo lampante, perché lì la tabella di embedding è già la matrice die logits
- std = 0.2 -> i logits partono in N(0, 0.02^2) cioè praticamente tutti uguali, la softmax è quasi uniforme, il modello parte dichiarando "non so nulla, tutti i 66 caratteri sono equiprobabili", è lo stato di ignoranza corretto da cui partire
- std = 1.0 -> i logits partono sparsi tra -3 e +3, softmax già confidenze su valori casuali. Il modello parte con convinzioni forti e sbagliate, le prime centinaia di step le spende a disfarle

Nel trasformer completo c'è una seconda ragione ancora più importante, la tenuta del residual steam attraverso n_layer somme, la veiamo dopo dove dovremo scalare per 0.2/sqrt(2*n_layer)


---

Il bigram model

Il trucco è che embedding_dim = vocab_size, la riga t-esima della tabella è il vettore di logits per "cosa viene dopo il token t". Non c'è nessun layer intermedio. W[i][j] è, dopo il training, il log-conteggio (non normalizzato) di quante volte j segue i nel corpus.
È una tabella di conteggi bigram, imparata per discesa del gradiente invece che contata. Inefficiente, ma didatticamente utile, non ha contesto. La predizione alla posizione t guarda idx[b, t] e basta, i token 0..t-1 sono lì nel tensore completamente ignorati.
Teniamo a mente questa cosa perché l'unica cosa che aggiungiamo successivamente è il meccanismo che fa arrivare l'informazione delle psoizioni precedenti a quella corrente, tutto il resto dello scheletro (embedding, loss, batch, loop, generate) è già questo


Il .view() prima della loss.
F.cross_entropy vuole (N, C) per i logits e (N,) per i target. Noi abbiamo (B, T, V) e (B, T). La riduzione è l'affermazione che abbiamo fadtto prima ma resa operativa, le B*T posizioni sono B*T esempi indipendenti quindi si appiattiscono in un'unica dimensione batch.

La .view() funziona senza copia perché logits esce contiguo dall'embedding, se dovesse lamentarsi con errore sul size not comaptible il rimedio è fare un .reshape() manuale che copia quandos erve. Nel transformer dopo le trasposizioni dell'atention vedremo sicuramente questo caso.

La cross cross_entropyPrima di delegare a F, scriviamo la cross_entropyper confrontarla per chiarire cosa succede davvero

```
def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = logits - logits.max(dim=-1, keepdim=True).values #stabilità numerica
    logprobs = logits - logits.exp().sum(dim=-1, keepdim=True).log() # log_softmax
    return -logsprobs[torch.arange(len(targets)), targets].mean()
```

alcuni dettagli importanti...

La sottrazione del massimo, exp(100) in float32 + inf. La softmax è invariante per traslazione -softmax(x) == softmax(x-c) per qualunque c, perché la costante si semplifica tra numeratore e denominatore, quindi sottrarre il massimo non cambia il risultato ma garantisce che l'argomento di exp sia sempre <= 0, cioè exp(...) <= 1 mai overflow, questo trucco si usa spesso, anche nell'attention.

log_softmax invece di log(softmax(x)), matematicamente sono la stessa cosa, numericamente no, se una probabilità vale 1e-45 in flaot32 è già denormale, e prenderne il log da -inf -> gradiente NaN nell'apprendimento. Calcolando x-logsumexp(x) non passiamo mai dallo spazio delle probabilità ma restiamo in precisione piena.

L'indicizzazione `logprobs[arange(N), targets]`, fancy indexing con due tensori di pari lunghezza, prende logprobs[p, targets[p]], logprobs[1, targets[1]], ccioè solo il log-prob della classe corretta che è tutta la cross entropy per target -log p(classe giusta). non c'è nessuna somma su tutte le classi, quella è già dentro logsumexp.

si può verificare
```
logits = torch.randn(32, 66)
targets = torch.randint(66, (32,))
assert torch.allclose(cross_entropy(logits, targets), F.cross_entropy(logits, targets))
```

quella di F è un kernel fuso, un solo passaggio sulla memoria e il backward è scirtto a mano dagli sviluppatori di torch invece di ricostruirlo con l'autograd, almeno ora sappiamo cosa fa.

Quello che dobbiamo vedere numericamente è
```
m = BigramLM(tok.vocab_size).to(device)
x, y = get_batch(train, 4, 8)
logits, loss = m(x, y)
print(logits.shape, loss.item())
```
dovremmo aspettarci `torch.Size([4, 8, 66])` e una losso di circa 4.19, non è un numero a caso ln(66) = 4.1897, è la cross entropy di una distribuzione perfettamente uniforme su 66 classi, la loss di un modello che non sa nulla perché i logits sono tutti identici std=0.02.

Questo è il sanity check più importante di tutti e dovremmo farlo ad ogni step

- loss iniziale ~ ln(V) -> init, reshape, e allineamento dei target sono corretti
- loss inziiale molto più alta (tipo 5.5) -> init troppo grande, o abbiamo scambiato le dimensioni nel view
- loss iniziale più bassa di ln(V) -> stiamo leakando il target, in un LM significa quasi sempre shift sbagliato, il modello vede la risposta

come riferimento su tinyshakespeare bigram addestrato è circa `~2.45`, su un transformer piccolo `~1.5`, il limite teorico dell'entropia dell'inglese a livello di carattere è `~1.1`

```
uv run src/mygpt/model.py
parametri = 4,356
logits = (4, 8, 66)
loss = 4.1913 (attesa ~ ln(66) = 4.1897)
```

qualche nota utile

Il seed di partenza non è 0, la riga torch.zeros((1,1)) è sbagliata perché per noi l'id 0 è UNK, invece tok.encode("\n") parte da newline, cioè linizio di ogni riga che è semanticamente quello che vogliamo

I tre passi del training in ordine sono
```
optimizer.zero_grad(set_to_none=True) # azzera i gradienti del passo precedente
loss.backward() # calcola dL/dp per ogni parametro, li ACCUMULA in p.grad
optimizer.step() # p -= lr * f(p.grad)
```
In torch i gradienti si accumulano non si sovrascrivono, è una scelta progettuale per gradient accumulatin su batch che non stanno in memoria, significa che se dimentichiamo zero_grad sommiamo i gradienti di tutti gli step recedenti e il training diverge.
set_to_none=True mette p.grad = None invece di riempirlo di zeri, salta un kernel di memset per ogni parametro, dal 2.0 è di default ma esplicitarlo è una sicurezza.

model.eval() / model.train() per ora non fanno nulla, non abbiamo dropout ne normalizzazioni con statisttiche, li mettiamo perché poi quando aggiungiamo il dropout non vanno dimenticati o valuteremo il modello con neuroni spenti a caso e non capiremo perché la loss è rumorosa.

lr=1e-2 è alto, per un modello da 4356 parametri va bene e converge subito, il transformer vero gira a 3e-4 o meno, più parametri e più profondità significa grandienti che si compongono e un lr così lo farebbe esplodere, poi lo cambieremo a implementazione finita insieme al warmup.

Ci aspettiamo val da 4.19 a ~2.45 e poi che il testo sarà illegibile ma non casuale, vediamo parole di lunghezza plausibile, spazi nei punti giusti, q seguite da u, i due punto dopo i nomi dei personaggi, è quello che ci aspettiamo guardando un solo carattere indietro ed è il muro contro cui ci serve l'attention.

L'inefficienza da notare in generate è ch ecalcoliamo i logits per tutte le T psoizioni e ne buttiamo via T-1, per il bigram è assurdo (conta solo l'ultimo token), per il transformer sarà ricalcolare da zero l'intero contesto a ogni carattere, è il problema che risolve la KV cache che implementeremo dopo.

```
uv run src/mygpt/train.py
step     0 | train 4.1865 | val 4.1864
step  5000 | train 2.4533 | val 2.4831

============================================================


NCoth d INouekimausmmy tharthfolabl fo, bud
FORGlevis ongave t,
He INANCarshavere, nend the rt forutheg HEal E:
ERELI ded ftr andsanmar I orupeate, mmory? s;
OMuth prs,
Mouprouino no u avithepl orengin f the lle.
EYOngh bef t
RDUn ween tho'silofo gmppove wh, hisownn arinee IINRDUENGor s w hiefomon yshy hthamive as teng
PO:
D mpesie
I le wat t hy tothurk'enkertows:


BENCAhand I bymmanger tu Bl him
We bat od ISThore, h isurearouticaueco-co t hd facath ssestat, sor w Yong t hapalllis s inge,
MEDU
```


