Dai token ai batch (B, T)

L'idea centrale e sottovalutata è costruire il batch da predire, facciamo un esmepio
x = "First Ci"
y = "irst Cit" è x shiftato di 1

Notiamo che non è un esempio di training, sono ben otto:

context         -> target
"F"             -> "i"
"Fi"            -> "r"
"Fir"           -> "s"
"Firs"          -> "t"
"First"         -> " "
"First "        -> "C"
"First C"       -> "i"
"First Ci"      -> "t"

Il modello emette (B, T, V) logits, una distribuzione di probabilità per ogni posizione, non una sola alla fine, la maschera casuale garantisce che la predizione viene fatta alla posizione t e che non vediamo nulla oltre t, ogni posizione è un esempio di training legittimo ed indipendente.

Un batch (64, 256) = 16384 esempi per forward pass, da 64 righre di test

Questo è il motivo  per cui il pretraining non supervisionato funziona su scala, i dati etichettano se stessi, T volte per sequenza, non esiste annotazione umana e ogni carattere del corpus è contemporaneamente input e label.

Un side effect utile da tenere a mente è che il modello impara a predire il testo dato un contesto di lunghezza 1, 2, 3, ..., T non solo T, è per questo che a inferenza si può partire da un prompt di un solo token e funziona lo stesso, quel caso è nel training set, il contarrio però non vale, oltre block_size non possiamo andare perché wpe ha solo T righe.

nel tokenizer quindi creiamo load_data(path, tokenizer, device) -> tuple[Tensor, Tensor] che legge il corpus, lo codifica, lo splitta e ritorna la coppia (train, val) e costruiamo anche get_batch(data: Tensor, batch_size: int, blcok_size: int) -> tuple[Tensor, Tensor] che ritorna (x, y) etrambi (batch_size, block_size)

Il corpus è 1.1M token, In int64 sono 8.8MB di VRAM, quindi mettiamo tutto il dataset in memoria una sola volta all'avvio senza swappare memoria e consumare banda utile, get_batch diventra un'indicizzazione di tensori già residienti, zero trasferimenti host->device, zero implementazioni di roba come DataLoader, zero num_workers per lavorare in aprallelo, zero pin_memory, zero collate, bastano poche righe invece di una pipeline, vista la scala ridotta teniamo l'implementazione semplice.

Ci sono dei punti da decidere però
- Lo split trian/val come lo si fa? ingenuamente is può dire mescolo tuto e prendo il 90% per il training set, ovviamente è scorretto. Conviene lo split contiguo e mai mescolato, nel nostro scenario block_size=256, finestra di val che parte a 1000, finestra di train che parte a 995. Condividendo 251 token su 256 il modello ha praticamente memorizzato la risposta di validazione durante il training, la val loss scende insieme alla train loss e non vediamo l'overfitting, lo split contiguo data[:n] / data[n:] elimina il problema per costruzione, nessuna finestra di train può sconfinare oltre n, nessuna di val può iniziare prima.
- L'off-by-one di get_batch, bisogna estrarre indici di partenza casuali i, quale è il massimo i valido? Possiamo derivalo usando data[i+blcok_size-1] (l'utimo token di x) o data[i+block_size] (ultimo token di y) bisogna stare attenti però a torch.randint(high, ...) ha high esclusivo, se sbagliamo avremo IndexError ogni ~100k batch. high = len(data)- block_size, ci serve che esista data[i + blcok_size] (l'ultimo token di y), quindi i + block_size <= len-1, cioè i <= len - blcok_size -1. Siccome randint(high) è esclusivo su high passando len-block_size si ottiene i_max = len - block_size -1, se si passa len - blcok_size -1 per sicrezza non crasha ma perdiamo l'ultima finestra
- dtype, non è una scelta libera, l'embedding lookup in torch richiede indici interi a 64 bit, torch.long (int64)
- Campionamento causale o epoche? ogni get_batch pesca B offset casuali indipendenti con reimmissione, alcuni token verrann ovisti più volte, altri mai, e va bene così, il vantaggio è sia l asemplicità ma con offset casuali il modello vede tutti gli allineamenti possibili dei chunk, mentre con chunk fissi vedrebbe sempre le stesse 4300 finestre, se gestiamo le epoche dovremmo gestire shuffle, stato, resume e complichiamo le cose.
- Il seed, dove lo mettiamo e cosa lo rende riproducibile? suprattuto per la validation  usiamo batch diversi a ogni valutaizone o gli stessi? le due scelte impattano sul rumore e sulla curva della loss. torch.manual_seed(1337) una volta sola all'inizio e siamo coperti su CPU e GPU. Il rumore sulla loss dovrebbe mediarsi da solo su ~200 batch ed evitiamo di portarci un generator a parte che su cuda è un casino (tortch.Generator(device='cuda')) incompatibile con quello della CPU.

checkx, y = get_batch(train, batch_size=4, block_size=8)
assert x.shape == y.shape == (4, 8)
assert x.dtype == torch.int64
assert (x[:, 1:]) = y[:, :-1]).all() questo è il vero test per cui sappiamo che stiamo apprendendo il prossimo token, è l'invarianza allo shift, y e x si spostano di uno, gli ultimi T-1 token di X devono coincidere coi primi T-1 di y, se si sbaglia questo stiamo apprendendo funzione sbagliata

vale la pena fare un appunto sulla riga 
```
idx = ix[:, None] + offset # (B, 1) + (T,) -> (B, T)
```
è la prima volta che vediamo il broadcasting, ma lo incontreremo spesso nell'attention e altrove. Torch allinea le shape da destra ed espande ogni dimensione di talgia 1
```
ix[:, None]     (B, 1)      off (T,)    -> (1, T)
                  | espande su T              | espande su B
                  V                           V
                (B, T)                      (B, T)
```
risultato idx[b, t] = ix[b] + t, la matrice di tutti gli indici assoluti del batch. Poi data[idx] è advanced indexing, torch prende un tensore di indici di shape qualsiasi e restituisce un tensore della stessa shape con i valori corrispondenti. Due kernel, nessun loop Python, niente lascia la GPU.

L'alternativa leggibile fa la stessa cosa con un loop python, a B=64 sono 128 slice per step, in pratica
```
x = torch.stack([data[i : i+block_size] for i in ix])
y = torch.stack([data[i+1 : i+block_size+1] for i in ix])

```

Guardiamo i token del train set e un esempio di batch
```
uv run src/mygpt/data.py
vocab_size = 66
"\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
train = 1,003,854 token | val = 111,540 token | device = cuda:0
'v'          -> 'e'
've'         -> 'r'
'ver'        -> 'y'
'very'       -> ' '
'very '      -> 'm'
'very m'     -> 'e'
'very me'    -> 'a'
'very mea'   -> 'n'
```
