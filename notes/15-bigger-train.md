Addestriamo su un dataset più grande e risolviamo alcuni problemini

dataset wikimedia/wikipedia, config 20231101.it, 1,83 milioni di articoli, 10 file parquet, il primo da 529 MB

Un solo shard basta per ora, 2000 iterazioi * 64 * 256 = 32.7M token per run a ~2.4 byte/token fanno circa 80MB di testo per epoca, lo shard 0 decompresso sono circa 1.2GB cioè 15 secondi per epoca, l'overfitting smette di esserci

Però ci sono alcune cose importranti
- Il training BPE diventa improponibile, riconta tutte le coppie a ogni fusione, su 1MB sono 10 secodni, su 500 con bocab 8192 sono ore, si risolve usando un'indice coppia -> chuk o in modo lazy come si fa spesso si addestra il tokenizer su un campione di 50MB e si usa su tutto
- Vocab 10024 va alzato, era tarato su 1MB di un solo autore, per wikipedia italiana possiamo provare 8192 e con lm_head legata sono 3.1M parametri di solo embedding, dobbiamo ribilanciare il modello
- load_data carica tutto in RAM della GPU, 500MB di testo a int64 sono circa 1.6GB VRAM per i token, forse è il momento di passare ua uint16 (vocab < 65536) e magari usare np.memmap come fa nanoGPT

gli accenti italiai no sono un problema, siamo byte-level, à sono 2 byte e BPE li fonde da solo aprimo giro

Il tokenizer BPE con vocab 8192, 3.76 byte/token contro 2.36 dell'inglese char-level, l'italiano si comprime meglio, ci sono parole più lunghe eregolari, guardando i token lunghi, vediam,o ' contemporaneamente', ' rappresentazione', ' cinematografica', ' precedentemente', ' originariamente', ' Successivamente'

Possiamo azzerare il dropout per ora per fare prove su un'epoca anche perché e contro la memorizzazione e ora non dovremmo più overfittare, se capita lo alziamo, max_iters lo rimettiamo a 5000 sono 0,6 epoche, non ci sono dati visti du evolte, il limite ritorna ad esser ei modello e non il corpus, blcok_size lo lasciamo 26 facciamo un cambiamento per volta tanto con 3.76 byte/token quei 256 token sono già circa 960 caratteri di contesto, quattro volte quello che avevamo prima, p.s. anche nanoGPT addestra con dropout=0.0 e lo riacende solo nel finetuning

Il training su GPU occupa 7.5GB VRAM, i logit ora sono 64*256*8192, mezzo GB di tensore più 1.16 GB di token residenti

la loss iniziale attesa è circa 9.01 = ln(8192) contro, contro i 4.19 di ln(66) del char-level, non è un peggioramento è che indovinare a caso tra 8192 token è più difficile che fra 66

il confronto giusto tra tokenizer diversi resta bit/byte, il riferimento inglese era 2.066, quindi non è paragonabile in senso stretto, lingua diversa, dominio diverso, ma il valore assoluto dice comunque quanto bene il modello comprime l'italiano e con 0.6 epoche invece di 24 ci aspettiamo che la train loss e la val restinano appaiate per tutto il run, se divergono vuol dire che 145M token non bastano

guardando al training vediamo train e val accoppiate con un piccoo gap 0.13, per tutte le 5k iterazioni, su shakespear finivamo con train a 0.49 e val 4.47, l'overfitting no nè stato combattuto, è semplicemente sparito, con 145M token e 0,57 epoce non c'è niente da memorizzare, è il motivo per cui abbiamo messo dropout = 0.0 che a posteriori è la scelta giusta

```
step  4750 | train 3.2725 | val 3.4006 | bpb 1.338 | 115.2 ms/it | 142k tok/s
step  5000 | train 3.2623 | val 3.4077 | bpb 1.341
```

La val stava ancora scendendo a 4750, non siamo più data-bound, siamo limitati dal calcolo, da qui in poi più iterazioni epiù parametri si traducono in loss più bassa


| | | bpb | gap | train/val | byte/s |
|---|---|---|---|---|---|
| char+wpe | Shakespeare | 2.090 | 1.45 / 4.47 | 289k |
| BPE 1024 | Shakespeare | 2.066 | 0.49 / 4.47 | 305k |
| BPE 8192 | it.wikipedia | 1.338 | 3.27 / 3.40 | 519k |

1.338 bit/byte, il confronto con 2066 no nè stretto, lingua e dominio sono diversi, ma il numero assoluto dice che il modello comprime l'italiano di wikipedia a un terzo di un byte per carattere che è il run più veloce in byte/sec che abbiamo visto, 519k contro 289k del char-lvel a parità di GPU

Il testo è curiosamente corretto
```
= Plutone =

Plutone è un comune francese di 3.623 abitanti situato nel dipartimento
della Côte-d'Or nella regione della Borgogna-Franca Contea.

Società
Evoluzione demografica
Note
Altri progetti
Collegamenti esterni
```
sembra italiano corretto, concordanze giuste, `Côte-d'Or` con accento e apostrofo, numeri formattati all'italiana, a imparato la struttura del documento, le sezione di wikipedia nell'ordine canonico, titolo ripetuto in fondo come categoria

Poi si incanta e sforna venti comuni francesi di fila, no è un difetto del modello è la distribuzione del corpus, wikipedia italiana ha decine di migliaia di stub sui comuni francesi tutti generati dallo stesso template, identici, tranne nome  enumero di abitanti, il modello ha imparato benissimo la cosa più frequente che gli abbiamo dato, con un prompt esplicito su dante il problema si vede ancora di più, ripete "Divina Commedia" come un mantra e ci attacca sopra la biografia di un governatore ottocentesco perché quel template è molto più probabile di un'analisi letteraria

quindi abbiamo tre strade con diverso rapporto risultato/fatica

1. allunghiamo il run, la val scende ancora, 20k iterazioni sono circa 40 minuti e ci danno il salto gratis senza toccare una riga
2. ingradiamo il modello, 13.8M su 145M token sono pochi, il rapporto sano sarebbe 10-20 token per parametro quindi ci starebbe un 440-100M di parametri che entrerebbe nei miei 16GB di VRAM
3. filtraiamo il corpus, buttiamo gli stub sotto una certa lunghezza, toglie gran parte dei cloni, è la più laboriosa e manipolativa

Il criterio di Chinchilla è 20 token per parametro, quindi con 145M token il modello ottimale è 7M paraametri, al massimo 15M se scendiamo a 10 token/parametro, quindi il nostro modello dovrebbe essere nella fascia giusta, un modello 50-100M gestirebe 1-2 miliardi di token (7-14 shard di wikipedia)

Quindi per ora aumentiamo il run a 20k iterazioni, facciamo l'eval ogni 500 iterazioni, dropout = 0.0

```
 uv run src/mygpt/train
.py                                                              
vocab 8192 | 3.67 byte/token sulla val
13,786,368 parametri
  decay:     25 tensori, 13,762,560
  no decay:  44 tensori,     23,808
step     0 | train 9.1033 | val 9.1043 | bpb 3.582 | lr 9.90e-06 |    nan ms/it |   
  0k tok/s
step   500 | train 4.5227 | val 4.6304 | bpb 1.822 | lr 9.99e-04 |  118.3 ms/it |   
139k tok/s
step  1000 | train 3.9844 | val 4.1100 | bpb 1.617 | lr 9.95e-04 |  116.7 ms/it |   
140k tok/s
step  1500 | train 3.7779 | val 3.9004 | bpb 1.535 | lr 9.89e-04 |  117.2 ms/it |   
140k tok/s
step  2000 | train 3.6589 | val 3.7751 | bpb 1.485 | lr 9.80e-04 |  116.8 ms/it |   
140k tok/s
step  2500 | train 3.5703 | val 3.6993 | bpb 1.456 | lr 9.68e-04 |  115.4 ms/it |   
142k tok/s
step  3000 | train 3.5340 | val 3.6227 | bpb 1.425 | lr 9.54e-04 |  125.0 ms/it |   
131k tok/s
step  3500 | train 3.4594 | val 3.5909 | bpb 1.413 | lr 9.37e-04 |  124.4 ms/it |   
132k tok/s
step  4000 | train 3.4243 | val 3.5570 | bpb 1.400 | lr 9.17e-04 |  125.2 ms/it |   
131k tok/s
step  4500 | train 3.3972 | val 3.5240 | bpb 1.387 | lr 8.96e-04 |  125.2 ms/it |   
131k tok/s
step  5000 | train 3.3662 | val 3.4898 | bpb 1.373 | lr 8.72e-04 |  125.2 ms/it |   
131k tok/s
step  5500 | train 3.3436 | val 3.4775 | bpb 1.368 | lr 8.46e-04 |  124.4 ms/it |   
132k tok/s
step  6000 | train 3.3243 | val 3.4620 | bpb 1.362 | lr 8.19e-04 |  115.2 ms/it |   
142k tok/s
step  6500 | train 3.3011 | val 3.4325 | bpb 1.351 | lr 7.89e-04 |  115.2 ms/it |   
142k tok/s
step  7000 | train 3.2577 | val 3.4136 | bpb 1.343 | lr 7.58e-04 |  115.2 ms/it |   
142k tok/s
step  7500 | train 3.2404 | val 3.3893 | bpb 1.334 | lr 7.26e-04 |  125.4 ms/it |   
131k tok/s
step  8000 | train 3.2367 | val 3.3656 | bpb 1.324 | lr 6.93e-04 |  116.1 ms/it |   
141k tok/s
step  8500 | train 3.2144 | val 3.3568 | bpb 1.321 | lr 6.59e-04 |  115.2 ms/it |   
142k tok/s
step  9000 | train 3.2009 | val 3.3341 | bpb 1.312 | lr 6.24e-04 |  117.7 ms/it |   
139k tok/s
step  9500 | train 3.1622 | val 3.3176 | bpb 1.305 | lr 5.89e-04 |  116.3 ms/it |   
141k tok/s
step 10000 | train 3.1665 | val 3.3212 | bpb 1.307 | lr 5.54e-04 |  116.6 ms/it |   
140k tok/s
step 10500 | train 3.1466 | val 3.3062 | bpb 1.301 | lr 5.18e-04 |  116.0 ms/it |   
141k tok/s
step 11000 | train 3.1468 | val 3.2917 | bpb 1.295 | lr 4.83e-04 |  115.2 ms/it |   
142k tok/s
step 11500 | train 3.1168 | val 3.2669 | bpb 1.285 | lr 4.48e-04 |  115.5 ms/it |   
142k tok/s
step 12000 | train 3.1014 | val 3.2509 | bpb 1.279 | lr 4.14e-04 |  117.6 ms/it |   
139k tok/s
step 12500 | train 3.0866 | val 3.2397 | bpb 1.275 | lr 3.80e-04 |  124.6 ms/it |   
131k tok/s
step 13000 | train 3.0723 | val 3.2262 | bpb 1.269 | lr 3.48e-04 |  121.3 ms/it |   
135k tok/s
step 13500 | train 3.0566 | val 3.2162 | bpb 1.265 | lr 3.17e-04 |  120.0 ms/it |   
137k tok/s
step 14000 | train 3.0633 | val 3.2143 | bpb 1.265 | lr 2.87e-04 |  124.3 ms/it |   
132k tok/s
step 14500 | train 3.0317 | val 3.2008 | bpb 1.259 | lr 2.59e-04 |  123.1 ms/it |   
133k tok/s
step 15000 | train 3.0224 | val 3.1774 | bpb 1.250 | lr 2.33e-04 |  123.0 ms/it |   
133k tok/s
step 15500 | train 3.0055 | val 3.1696 | bpb 1.247 | lr 2.09e-04 |  120.5 ms/it |   
136k tok/s
step 16000 | train 3.0012 | val 3.1794 | bpb 1.251 | lr 1.87e-04 |  123.8 ms/it |   
132k tok/s
step 16500 | train 2.9820 | val 3.1603 | bpb 1.244 | lr 1.67e-04 |  124.9 ms/it |   
131k tok/s
step 17000 | train 2.9792 | val 3.1621 | bpb 1.244 | lr 1.50e-04 |  125.6 ms/it |   
130k tok/s
step 17500 | train 2.9763 | val 3.1390 | bpb 1.235 | lr 1.35e-04 |  125.5 ms/it |   
131k tok/s
step 18000 | train 2.9655 | val 3.1449 | bpb 1.237 | lr 1.22e-04 |  133.7 ms/it |   
123k tok/s
step 18500 | train 2.9544 | val 3.1342 | bpb 1.233 | lr 1.13e-04 |  124.1 ms/it |   
132k tok/s
step 19000 | train 2.9480 | val 3.1376 | bpb 1.235 | lr 1.06e-04 |  130.3 ms/it |   
126k tok/s
step 19500 | train 2.9505 | val 3.1310 | bpb 1.232 | lr 1.01e-04 |  127.7 ms/it |   
128k tok/s
step 20000 | train 2.9443 | val 3.1368 | bpb 1.234 | lr 1.00e-04 |  131.3 ms/it |   
125k tok/s

miglior val loss: 3.1310  ->  out/ckpt.pt
============================================================

Benvenutiaceae, regia di Jerzy Sakhirowski (1966)
Judd a Tisser, regia di John B. Cooper (1978)
Sintus (Dark Side), regia di Gus Van Sant (1979)
La ragazza che si chiama Niente (High Warrior), regia di Jim Ivory (1979)
The Movie, regia di Gus Van Sant (1984)
Topolino, regia di Jeffrey Becker (1989)
Che cosa è successo (Che cosa è successo), regia di David B. Schaast (1990)
Aftermath, regia di Nathan H. DeMille (1990)
Aftermath, regia di Adam Taylor (1991)
Derby, regia di Jerry West (1991)
Vedova che abiti (Vedova Chevalier), regia di Daniel Shiki (1995)
Derry, regia di Jerzy Nekhrowski (1995)
Vuoto (Dream), regia di John Donat (1996)
La signora in giallo, regia di Jerzy Nekhrowski (1997)
La regina e il boia, regia di Ethan Scott (1997)
Derby, regia di Tony Scott (1998) - cameo
Tark (Mr. Howard), regia di Ethan Scott (1997)
Primo giorno, regia di Renato Castellani (1998)
Don't Go, regia di Ethan Scott (1998)
R.T.D.P. di R.T.D.B. (1999)
 Dr. Strangl (Demolition Dr. Strangl), regia di Tony Scott (2000)

Riconoscimenti 
Premio Oscar
 1962 – Miglior attore protagonista in una serie TV per la TV
 1962 – Miglior attore protagonista in una serie TV per la TV
 1963 – Miglior attore protagonista in una serie TV per la TV
 1969 – Miglior attore protagonista in una serie TV per la TV
 1975 – Miglior attore protagonista in una miniserie o film cinematografico per la T
V
 1979 – Miglior attore protagonista in una miniserie o film cinematografico per la T
V
 1981 – Miglior attore protagonista in una serie TV per la TV
 1983 – Miglior attore protagonista in una serie TV per la TV
 1983 – Miglior attore protagonista in una serie TV, episodi speciali e un film tele
visivo per la TV
 1984 – Miglior attore protagonista in una serie TV per la TV
 1985 – Miglior attore protagonista in una serie TV per la TV
 1986 – Miglior attore protagonista in una serie TV per la TV
 1986 – Miglior attore protagonista in una serie TV per la TV
 1988 – Candidatura per la miglior regia per la TV
 1987 – Candidatura per la migliore sceneggiatura originale per la TV
 1988 – Candidatura per la migliore sceneggiatura originale per la televisione per l
a TV
 1989 – Candidatura per la migliore sceneggiatura originale per la televisione per l
a TV
 1988 – Candidatura per la migliore sceneggiatura originale per la televisione per l
a TV
 1989 – Candidatura per la migliore sceneggiatura originale per la televisione per l
a TV
 1995 – Candidatura per la migliore attrice non protagonista per la televisione per 
la TV
 1996 – Miglior attore protagonista per la televisione di culto per la TV
 2002 – Candidatura per la migliore sceneggiatura originale per la televisione per l
a TV
 2005 – Migliore attrice per la televisione per la televisione per la TV
 2005 – Miglior attore protagonista per la televisione per la televisione
 2009 – Candidatura per la migliore attrice per la televisione per la televisione pe
r la televisione per la televisione per la televisione
 2013 – Candidatura per la migliore attrice per la televisione per la televisione pe
r la TV
 2020 – Miglior sceneggiatura per la televisione per la televisione per la TV
 2022 – Candidatura per la migliore attrice per la televisione per la TV
 Premio internazionale per la migliore sceneggiatura per la televisione per la telev
isione per la TV
 2004 – Candidatura per la miglior serie per la televisione per la televisione per l
a TV
 2010 – Miglior serie per la televisione per la televisione per la TV
 2020 – Miglior attrice per la televisione per la televisione per la TV
 2020 – Candidatura per la migliore attrice per la televisione per la TV per la TV
 2023 – Miglior attrice per la televisione per la TV
 2020 – Miglior attrice per la televisione per la televisione per la TV
 2022 – Miglior attrice per la televisione per la TV
 2022 – Miglior attrice per l'interpretazione di una televisione per la televisione 
per la TV
 2023 – Miglior attrice per la televisione per la TV

Doppia
```

bpd 1.338 -> 1.232 best val 3.1310 a 19.500, ci siamo resi conti di un bug che ci faceva addestrare a metà velocità, il contatore sself.pos di RoPE si incrementava anche in training, dove non serve a niente (use_cache è False, t0 vale sempre 0) ogni nuovo valore di pos era un grafo nuovo da compilare

Guardiamo i numeri
| | bpb | train/val | gap |
|---|---|---|---|
| 5k it | 1338 | 3.27/3.40 | 0.13 |
| 20k it | 1.232 | 2.94/3.14 | 0.19 |

quattro volte il calcolo per un -8$, rendimenti decrescenti visibili, la val si appiattisce dopo 15k iterazioni e da lì oscilla, il gap si è aperto un po' da 0.13 a 0.19 a 2,3 epoche l'overfitting si affaccia ma non è ancora forte

Analizzando il testo, sicurament eil modello ha imparato la struttura di wikipedia in modo più complesso, filmografie con titolo, regia di <nome> (anno) la sezione riconoscimenti con i premi ordinati per anno , la formattazione delle candidature, poi si incarta

```
Miglior attrice per la televisione per la televisione per la televisione per la televisione
```

è il campionamento non l modello, con top_k 40 e u modello piccolo la distribuzione dopo "per la" è talmente alta che si entra in un attrattore e non se ne esce, si può risolvere in due modi, uno con nucleus sampling (top-p) che taglia la coda in base alla massa di probabilità invece che al numero fisso di candidati e una penalità di ripetizione sui token emessi, quindi un paio di righe in generate.py

Quindi implementiamo --repetition-penality su generate.py e top-p oltre che top-k

Il risultato è nettamente migliorato

```
uv run src/mygpt/generate.py --prompt "Dante" --tokens 160 --temperature 0.6         
# out/ckpt.pt: iter 19500, val loss 3.1310
# temp 0.6 | top-k 40 | top-p None | rep 1.15
# 160 token in 0.50s = 318 tok/s (kv cache, batch 1)
============================================================
Dante e Dante in un'altra scena, che si svolge a Roma, dove è presente una statua di
 S. Salvatore.

Sulle pareti della chiesa sono presenti alcuni affreschi del pittore genovese Giacom
o Mancini con la Madonna del Carmine, San Felice (San Francesco) e san Pietro da Ver
ona.

Il primo dipinto fu realizzato nel 1467 per volere dell'imperatore Carlo III d'Altav
illa; il secondo riprende l'iconografia dei santi patroni del paese. Il secondo espo
sto è rappresentato dall'altare maggiore della Vergine col Bambino, opera attribuita
 al Deposito di Paolo II, raffigurante il patrono del paese, Sant'Anna e San Bernard
ino.

L'opera è
```

la temperatura 0.6 piuù penalità, la temperatura bassa tiene la sintassi in riga, la penalità impedisce che quella stessa sicurezza diventi un loop, da soli uno da testo incantato e l'altra testo sconclusionato

Questo è un LLM modern oora, quello che può farci guadagnare qualche punto percentuale è
- RMSNorm invece di LayerNorm, niente media da sottrarre, nient ebias, un po' più veloce e funziona uguale
- SwiGLU invece di GELU nella MLP, una porta moltiplicativa, tre matrici invece di due
- GQA (grouped-query attention), più head per le query che per le chiavi, serve a far stae la KV cache in memoria su contesti lughi, la cache ora non cresce con il contesto

La scala non è una manopola, 255k token/s su un modello da13.8M sono 21 TOPS/s effettivi, un modello da 7B addestrato bene vuole 140 milairdi di token cioè 6* 7e9 * 1.4e11 = 5.9e21 FLOP, sulla mia gpu, una 5060 Ti ci vorrebbero 9 anni senza mai spegnerla

il modello non entra in GPu col dataset, serve parallelism con ZeR0/FSDP per spezzare in strati e ottimizzare i gradienti, tensor e pipeline parallelism e tutto ciò è codice di sistema/train non di modello

l'addestramento diventa instabile, a 13.8M la loss scende liscia, a miliarid di parametri arrivano degli spike, la lossa salta su e non torna giù, per questo nasce la QK-normaliztion, la z-loss, μP per trasferire gli iperparametri fra scale, sono rimedi a cose che ora non possiamo neanche osservare

Un run dura settimane, quindi servono checkpoint, restart e monitoraggio di spegnimenti macchian ecc

i dati diventano il lavoro principale, abbiamo 145M token, Llama 3 ne ha visto 15000 miliardi, più di 100k volte tanto, a quella scala il tempo si spende in deduplicaizone, filtri di qualità, decontaminazione dai benchmark, è la parte noiosa

avere qualcosa che risponde bene serve il post-training che è un argomento separato

- SFT, fine-tuning su decine di migliaia di conversazioni scritte da umani che insegna il formato "domanda -> risposta utile"
- Apprendimento dalle preferenze, RLHF, FPO e parenti, si raccolgono giudizi umani su coppie di risposte, si addestra un modello di reward o si ottimizza direttamente la preferenza, quin nascono l'utilità, il tono, il rifiuto di richieste dannose
- Poi uso di strumenti come contesto lungo, valutazione seria, inferenza in produzione che è un altro mondo ancora, perché dovremmo trattare batching continuo, quantizzazione, paged attention, decodifica speculativa

ma ci siamo riferiti a GPT-3 che era un modello del 2020 ed era un modello base
