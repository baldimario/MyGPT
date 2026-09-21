il nostro corpus tinyshakespeare è 1115394 byte, 294763 chunk, 15057 unici

Il concetto di BPE è già nel suo nome Byte Per Encoding

un modello di linguaggio deve scelgiere una unità atomica su cui lavorare, le scelte ovvie sono entrambe pessime

Carattere, è quello che avevamo con CharTokenizer, vocaboalrio minuscolo (66) nessuna parola sconosciuta possibile, ma ogni token prota pochissima informazione, il modello spende gran parte della su acapacità ad imparare l'ortografia, che q è seguita da u, che -ing eisste, invece che la sintassi e il suo senso, inoltre la finestra di contesto si riempie subito, 256 token sono 256 caratteri, tre righe di testo

Parola, ogni token è un'unità di senso, la finestra rende molto di più ma il vocabolario esplode (decine di migliaia di voci ognuna con la sua riga di embedding), la coda lunga è infinita, nomi propri, refusi, parole compsote, tutto ciò che non è nel dizionario diventa <UNK> cioè informazione buttata via e run, runs, running sono tre voci scorrelate, il modello deve imparare tre volte la stessa cosa


BPE sta nel mezzo, e ci sta in un modo che non è un compromesso ma una soluzione, le sequenze frequenti diventano un token unico, quelle rare si compongono in pezzi. CORIOLANUS è un token solo perché in questo corpus compare ovunque, una parola che il modello non ha mai visto si scrive comuqnue, lettera per lettera se serve, il vocaboalrio è finito e sceglibile, non esiste niente di inesprimibile

BPE di per sè nasce nel 1994 come alfgoritmo di compressione (Philip Gage) e viene  ripescato nel 2016 da Sennrich et al per la traduzione autoamtica, l'idea intera sta in quattro punti
1. partiamo con il vocabolario = i simboli elementari (epr noi i 256 byte)
2. contiamo tutte l ecoppie adiacenti nel corpus
3. la coppia più frequente diventa un simbolo nuovo con id nuovo
4. torniamo al punto 2 finché il vocabolario non ha la taglia voluta

non c'è alcuna euristica linguistica, nessun dizionario, nessuna regola morfologica è pura statistica e cooccorrenza, il fatto che non ne escano unità linguisticamente sensate è un risultato non un ingrediente

Osserviamo l eprime fusioni imparate sul corpus osservabili dal voraboalrio data/bpe.json

```
 256 =  ' ' + 't'   -> ' t'
 257 =  'h' + 'e'   -> 'he'
 258 =  ' ' + 'a'   -> ' a'
 259 =  'o' + 'u'   -> 'ou'
 260 =  ' ' + 's'   -> ' s'
 ...
 267 = ' t' + 'he'  -> ' the'   fusione di due fusioni
 271 = '\n' + '\n'  -> '\n\n'    il separatore di battuta
```

Il numero 267 è il cuore della cosa ' the' non nasce da una regola, nasce perché ' t' (256) e 'he' (257) erano già diventati simboli e la loro coppia è risultata la più frequente all'undicesimo giro, il vocabolario si costruisce a strati, ogni simbolo nuovo può fondersi con quelli nati prima, dopo 768 fuysiuoni abbiamo token lunghi fino a 10 byte

```
lunghezza:  1    2    3    4    5   6   7   8   9  10 byte
quanti:   256  204  238  147  109  36  18  10   3   3
```

i tre più lunghi sono CORIOLANUS, ' VINCENTIO', GLOUCESTEWR, il tokenizer ha imparato il cast di shakespeare semplicemente contando

si parte da byte non da caratteri, è la scelta di GPT-2 (Radford et al. 2019) e ha una conseguernza fondamntale, l'<UNK> non esiste più per costruzione, qualunque testo in qualunque lingua con qualunque emoji o carattere di controllo è una sequenza di byte e tutti e 256 i byte sono nel vocabolario dal primo istante, non c'è niente da gestire, nessun fallabck, nessun caso limite


```
vocab   512: train  3.5s | 571.095 token | 1.95 byte/token
vocab  1024: train  9.6s | 456.332 token | 2.44 byte/token
vocab  2048: train 20.6s | 385.329 token | 2.89 byte/token
```
quindi scegliamo 1024 per il nostro tokenizer, vocab 1024, 456332 token da 1115394 byte | 2.36 byte/token sulla val, abbiamo token più lunghi, i nomi come PETRUCHIO, ORIOLANS, ' daughter', GLOUCESTER, ' VINCENTIO', CORIOLANUS, sono un singolo token
'Tobe, or not to be' -> `[To][ be][,][ or][ not][ to][ be]`

abbiamo fatto tre scelte
- byte-level, quindil'UNK token non esiste, partiamo dai 256 byte possibili, qualunque testo è rappresentabile per costruzione. Il test bpe.decide(bpt.encoe("日本語")) == "日本語" passa, mentre il CharTokenizer avrebbe restituito [0, 0, 0], quindi un'intera classe di problemi è sparita e non può più presentarsi
- Pre-tokenizzazione con reges, r" ?\w+| ?[^\w\s]+|\s+" prima di fondere, impedisce che un token scavalchi il confine di parola (niente "e the") e attacca lo spazio della parola che segue, come GPT-2, per questo vediamo ' daughter' con lo spazio dentro
- si allena su 15057 chunk unici pesati per frequenza non su 294763 totali, da minuti a 10 secondi, e l'encode del corpus scende a 0.1s con una cache per chunk, è il motivo per cui non serve nessun file .npy di token particolari

I risultati non sono proprio quelli speciali in vari test

| | | val loss | bit/byte | ms/it |
|---|---|---|---|---|
| char + wpe | 2000 it | 1.4485 | 2.090 | 49.9 |
| char + Rope | 2000 it | 1.4581 | 2-104 | 56.6 |
| BPE + RoPE | 2000 it | 3.4279 | 2.099 | 126 minimo a iter 5000|
| BPE + Rope | 600 it | 3.3743 | 2.066 | 126 vince |
| BPE + RoPE | 1200 it d=.35 | 3.4117 | 2.089 | 127 |

La val loss di 3.37 non è peggiorata rispetto 1.45 sono unità diverse, un token BPE vale 2,36 byte, quindi la loss per token è per forza più alta, bit/byte è l'unica metrica onesta fra tokenizzatori diversi ed è `loss / (ln2 * byte_per_token)` la stampiamo ogni eval in train.party

il BPE ci ha regfalato l'1.1%, da 2-090 a 2-066, praticamente niente ma perché su 1MB di corpus la scelta del tokenizer non sposta la qualità, il motivo è che i due effetti si annullano, ogni token porta più informazione ma i token sono 456k invece di 1M, quindi il modello vede meno esempi e va in overfitting molto prima, il minimo è a iter 500 invece di 1750 come prima col CharTokenizer cioè 24 epoche invece di 32, ho provato anche dropout 0.35 e più iterazioni ma peggiora (2.089)

quello che abbiamo gudagnato davvero col BPE non è tra questi numeri
- finestra effettiva 2,36x, i miei 256 token di contesto coprono ~604 caratteri invece di 256, il modello può vedere una battuta intera invece di mezza
- velocità in byute al secondo sostanzialmente invariata 126 ms/it contro 56,6, l'iterazione costa il doppio (la lm_head proietta su 1025 uscite invece di 66 e la cross-entropy ci lavora sopra), ma macina 2,36x più testo: 306k byte/s contro 289k, non abbiamo perso nulla ma abbiamo spostato il lavoro
- in genrazione 424 token/s * 2,36 = ~1000 byte/s contro i ~700 di prima

il campione si commenta da se: Juliet e la Nurse nella stessa scena, endecasillabi, le parole sono qusi tutte vere, restano invenzioni tipo "tule", "caush", "smilke" perché i sotto-token si possono comunque combinare in parole strane, il BPE non è un dizionario.

Il muro è il corpus, 1MB è poco per 11M parametri, bisogna eliminar el'overfitting

La pre-tokenizzazione e perché serve, se lasciamo BPE libero sul flusso di byte fonde qualsiasi cosa sia frequente anche attraverso gli spazi, otterremo token come 'e th' o 'd the', che osno frequentissimi e linguisticamente osceni, mischiano la fine di una parola con l'inizio della successiva, il modello si ritrova a dover disimparare quei confini finti

La soluzione di GPT-2 è spezzare il testo priam di fondere con una regex e non fondere mai oltre quei confinim ci sono tre alternative in ordine, spazio opzionale + parola, spazio opzionale + punteggiatura, spazi, il testo si partiziona senza perdere un carattere, questo rende il round-trip esatto e nessuna fusione scavalca un pezzo

I token ' the' e 'the' sono due token diversi, sembra uno spreco invece è furbo, nel testo reale una parola è quasi sempre preceduta da uno spaziom quindi il token " " è quello comune e la parola nuda rest disponibile per i casi speciali come inizio riga, dopo una virgola, dentro una parola composota

l'ordine delle fusioni è il vocabolario, è una sottigliezza che bisogna cogliere, non un dettaglio implementativo ma il cuore della correttezza, per codificare "lower" non cerchiamo il token più lungo che combacia ma rigiochiamo la storia delle fusioni nello stesso ordine in cui è avvenuta nel training, la coppia 256 è nata prima della 256, ad ogni giro si fonde con la coppia col rank più basso presente nella stringa, cio èla più antica, quella che in training sarebbe stata fusa per prima, quando nessuna coppia rimasta è nel vocabolario l'algoritmo è finito

il motivo per cui non si prende il token più lungo anche se è più intuitivo è perché produrrebbe una tokenizzazione divesa da quell asu cui il modello è stato addrestrato, i modello ha visto ' the' come 267 perché è ciò che l'algoritmo di training produce, se in inferenza gliela passiamo spezzata diversamente equivale a dare una sequenza di id che statisticamente non ha mai incontrato, la codifica deve essere la funzione inversa esatta del training non una sua approssimazione ragionevole

decodificare è banale ma nasconde un'insidia

```
b"".join(self.itob[i] for i in ids).decode("utf-8", errors="replace")
```

itob mappa ogni id ai byte che rappresenta costruita ricorsivcamente in __init__(itob[nuovo] = itob[a] + itob[b]), decodificare è concatenare, non serve nessuna informazione oltre alla lsita delle fusioni,  per questo bpe.json contiene solo quella lista, il resto si ricostruisce

l'errors=replace non è pigrizia, genrando, il modello emette un token che è mezzo carttere multi-byte e potrebbe fermarsi lì, perché ha esaurito i token richiesti, uella sequenza nono è utf-8 valido e decode fallirebbe

Addestrare in fretta senza scrivere codice furbo

La versione ingenua conta le coppie sull'intero corpus a ogni fusione, 1,1 milioni di simboli × 768 fusioni, In python sono minuti,
il trucco è che le fusioni non attraversano i chunk, quindi non serve guardare il corpus, basta l'insieme dei chunk unici, ognuno pesato per quante volte compare.

294.763 chunk totali -> 15.057 unici

venti volte meno lavoro, stesso identico risultato e il training scende a 10 secondi, si può applicare la stessa idea alla codifica, un dizionario chink -> ids fa si che " the" si codifichi una volta solo in tutto il corpus, l'encode di 1.1MB costa 0.1 secondi ed è il motivo per cui non ci serve un file di token particolari sul disco, una complicazione in meno e un file che non può diventare vecchio rispetto al tokenizer che l'ha prodotto



```
 uv run src/mygpt/train.p
y                                                                         
vocab 1024 | 2.36 byte/token sulla val
11,033,856 parametri
  decay:     25 tensori, 11,010,048
  no decay:  44 tensori,     23,808
step     0 | train 6.9832 | val 6.9786 | bpb 4.273 | lr 9.90e-06 |    nan ms/it |   
  0k tok/s
step   100 | train 3.8320 | val 4.0707 | bpb 2.493 | lr 1.00e-03 |  119.1 ms/it |   
138k tok/s
step   200 | train 3.2383 | val 3.6567 | bpb 2.239 | lr 9.14e-04 |  118.0 ms/it |   
139k tok/s
step   300 | train 2.9290 | val 3.4849 | bpb 2.134 | lr 6.89e-04 |  117.9 ms/it |   
139k tok/s
step   400 | train 2.6943 | val 3.4169 | bpb 2.092 | lr 4.11e-04 |  118.0 ms/it |   
139k tok/s
step   500 | train 2.5285 | val 3.3787 | bpb 2.069 | lr 1.86e-04 |  118.0 ms/it |   
139k tok/s
step   600 | train 2.4417 | val 3.3743 | bpb 2.066 | lr 1.00e-04 |  117.8 ms/it |   
139k tok/s

miglior val loss: 3.3743  ->  out/ckpt.pt
============================================================

Indeed to be dance.

First Murderer:
See, go, 'tis good: and by my life, I hear
My master's good Petruchio.

First Musician:
Straney, my lord. How craves his wife!

Second Murderer:
No, sir, no more than I, I have said, 'tis not.

First Musician:
And, my lord, I cannot drink to-morrow.

First Murderer:
Ado, sir; I'll go bad.

Second Murderer:
Farewell, sir; but I dare: I will never be as
the accusation.

Second Murderer:
I have stirrong the prattle to believe you; I know
not, I am but that, fellow: I'll not be powed and
the purpose, I will not chop it to the gods,
and the world's readiness and not beloved.

Third Servingman:
It is a good fellow.

Second Murderer:
Ay, ay, sir.

First Murderer:
Not a gentleman that fellow's in his charge. IVirtue--

First Musician:
I have born his business in his wind.

Second Murderer:
I am an 'twill to be a worthy man.

First Musician:
How! what?

Second Musician:
I could not want no.

Second Musician:
Ay, not a petition, and the truth of the people.

Second Murderer:
Why, what will I am? Stand us to dance?

First Murderer:
Why, then we are found.

First Musician:
Hold, I cannot.

Second Murderer:
A grow, my good cousin, sir.

Second Murderer:
Honour Master Peter: I have done a while to bed
is morality.

Third Citizen:
You drink so, sir: you shall not show it off;
I'll be my county's punish.

Second Murderer:
No, but I are deliver'd a woman's name.

First Murderer:
How well, sir, what is such as I have? See, you see:
your lady, my heart is a cup of a chair,
thy, he is a parcel of a tale: 'tis true;
there's a woman's wife, he would read the woe;
and 'thwere he, and think him what is he?

Second Murderer:
I think he had round; but he had a very thief as a
contempture of his lamb.

First Murderer:
What damnable guests he modests to make his
house-hapes, he hath made me his creditation: he
was made a virtuously for a thing I detesty itself
his first. By thousand times in a peacriblous
she's sister in the feast; I had not worn it
hades for us, but it was a gentleman of it
knowledge: it was a creptive of an elect
a givision.

First Murderer:
Anon, it may be a thing, I hope he, if
she, it may lay a careless ballad.

Second Murderer:
How long?

First Murderer:
Sir, I have said, sir, he had been much more.

First Murderer:
No; for he will not be a man as the
```
