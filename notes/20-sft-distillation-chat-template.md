Possiamo dare al modello al momento di generare il vero testo degli articoli pertinenti, presi dal dataset Wikipedia, dentro la finestra di contesto, così il modello on deve più ricordare i fatti ma leggerli e riscriverli, questo si chiama RAG, retriefval-augmented generation

```
prompt: "Il processore Pentium 4"
 ||
 \/
RICERCA nei 1.8M articoli -> i passaggi più pertinenti ("Pentium 4", "NetBurst", "Intel Prescott", ...)
 ||
 \/
CONTESTO [passaggio 1] [passaggio 2] ... + "Il processore Pentium 4"
 ||
 \/
MODELLO continua il testo e i fatti li trova davanti
```

Per la ricerca serve un indice sugli articoli, le due opzioni che si usano di solito sono:
- BM25, cioè ricercare per parole chiave, come un motore id ricerca classico, trova i passagi che contengono le parole rare del prompt come "Pentium", semplice, senza reti neurali, le entità con nome proprio funzionano molto bene, SQLite ce l'ha già (FTS5)
- Embedding, cioè ricerca per significato, ogni passaggio diventa un vettore e si cercano i vicini del vettore del prompt, trova anche i sinonimi e parafrasi, ma serve un modello di embedding

Gli articoli vanno divisi in passaggi da qualche centinaio di token perché un articolo interno non sta nel contesto


Riguardo al contesto, i pasasggi trovati si mettono prima del prompt, il vincolo per noi è la finestra di 256 token, sono troppo pochi, ci sta un parafrafo e poco altro, bsiogna portarla almeno a 1024 che con la RoPE è fattibile


Il modello li usa, se ha imparato a farlo, con l'incipi vero del Pentium 4 davanti il modello risponde con l'anno giusto, il 2000, ma ha sbagliato il mese, Legge ma male perché in training non ha mai visto la situazione "teto di riferimento, poi testo che lo usa", il suo contesto è stato sempre la parte precedente dello stresso articolo

Quindi deve cambiare qualcosa nel training, cambiano i dati, a ogni finestra di training si antepone un passaggio recuerato che contiene informazioni utili a quella finestra

Quando la finesra dice "presentato nel novembre 2000" e il passaggio davanti contiene "novembre 20000" copiare fa scendere la loss, il modello impara a guardare il contesto quando c'è un'informazione precisa, diventa un lettore invece di un modello che tira ad indovinre, è il principrio di RETRO (DeepMind, 2022) e di In-Context RALM (2023) nella versione più semplice


cosa ci da questo metodo?
- fatti corretti quando sono nel testo recuperato, anche con un modello piccolo, copiare è molto più facile che ricordare
- le fonti, so sempre da quale articolo vien el'informazione
- aggiornabile, se wikipedia cambia, si aggiunge l'indice senza riaddestrare

cosa non ci da?
se la ricerca sbaglia il modlelo sbaglia, come nel caso di Manzoni, il paragrafo non contiene luogo di nascita e il modlelo ha inventato, la qualità dipende molto dalla ricerca

Non ragiona meglio, combianre fatti di articoli diversi resta difficile pe run modello così piccolo

Quindi il piano attuabile è provare queste cose:
- il benchmark cloze, per avere il punto zero, frasi dalla val come "X nacque a ___ nel ___" con l'accuratezza senza retrieval
- l'indice BM25 in SQLite sugli articoli divisi in passaggi, più una prova senza riaddesrare per capire quanto migliora il cloze mettendo solo i passaggi davanti
- la finestra a 1024 e il raining con passaggi anteposti per poi rimusurare

Il benchmark cloze consist enel prendere le voci dalla val (testo che il modello non ha mai visto in training), troncarle prima di un fatto e controllare se il modello lo completa in modo giusto

le domande sono costurite da voci di opere (film, album, romanzi, ...) della val estraendo con uan regex anno e regista per esempio, le domande si tengono pulite con due filtri, si scartano le domande la cui risposta compare già nel prompt e si scartano quelle in cui il confine tra popt e risposta cade dentro un token perché la risposta non sarebbe separabile in modo pulito

per valutare cloze non si genra testo, si fa solo forward su contesto + prompt + risposta e si guarda a ogni posizione della rispsota se il token più probavile è quello giusto

Dai i primi due punti del metodo che abbiamo pensato il benchmark chiarisce che il collo di bottiglia è il modello che non sa leggere e non la ricerca
Per i film nella val abbiamo misurato anno e regista, una risposta è esatta se la generazione greedy scrive esattamente la risposta giusta

| condizione | anno (n=405) | regista (n=1259) |
|---|---|---|
| senza retieval | 0.5% | 1.3% |
| retieval, voce compresa | 65.4% | 35.3% |
| retrieval, voce esclusa | 12.6% | 14.1% |

poi abiamo separato i due mdi di fallire, la risposta non era nel contesto o c'era e il modello non l'ha usata

| | rispostra nel contesto | ... e il modello la usa |
|---|---|---|
| voce compresa, anno | 84.9% | 76.5% |
| voce compresa, regista | 90.5% | 38.7% |
| voce esclusa, anno | 57.8% | 19.7% |
| voce esclusa, regista | 82.0% | 16.7% |

questo cosa significa esattamente?

senza retrieval i fatti non ci sono, 1%, conferma quello che immaginavamo sulla capacità di memoria, le voci della val non sono mai state vise dal modello e i passi non bastano

con la voce giusta nel contesto l'anno sale a 65% senza nessun retraining, il modello copia anche se nessuno glielo ha insegnato
Il dato chiave è il regista, il nome è nel conesto nl 90% dei casi ma il modello lo copia solo el 39%, un anno è un token solo, un nome come "Eriprando Visconti" ne richiede 4 o 5 e il mdello invena a metà, la ricerca funziona ma la lettura no

senza la voce sessa le rispose stanno spesso in altre voci (il 58-82% delle volte) ma il modello le usa ancora meno (17-20%), lì il testo non ha la stessa forma, bisogna insegnare al modello a legere addesrando con i passaggi recuperati dal contesto, quindi aumentare il contesto a 1024 token, per ogni articolo del train vanno precalcoalti con BM25., nella metà dei casi c'è il suo incipit, nell'altra metà o altre voci
Training, un fine-une del checkpoint attuale senza partie da zero
Per misurare si rifà cloze

Qui entriamo nel vivo di un concetto, quello che abbiamo addestrato attualmente è il modello con il solo pretraining, senza cambiar el'architettura si fa un secondo training per condizionare il moello a rispondere in un certo modo, ad esempio conversazionale, il modello abse sa solo continuare il testo, per farlo rispondere lo si addestra su conversazioni, si chiama post-training

Si definisce un chat template con token speciali aggiunti al vocabolario che segnano i turni e ogni conversazione diventa un testo unico

```
<|user|>
Quando è stato presentato il Pentium 4?<|end|>
<|assistant|>
Il pentium 4 è stato presentato da Intel nel novembre 2000.<|end|> 
```

per il modello è ancora "prevedi il prossimo token" l'obiettivo, impara però che dopo `<|assitant|>` viene una risposta alla domanda e che `<|end|>` significa "fermati", il template si può salvare anche nel GGUF in tokenizer.chat_template così llama-cli e llama-server lo usano in modalità chat

Inoltre si fa SFT (supervised fine-tuning), si riprende il modello base e si continua l'addestramento su decine o centinaia di migliaia di conversaiozni, ci sono due dettagli impotanti però, la loss si calcola solo sui token della risposta, le domande dell'utente sono mascherate, il modello non deve imparare a scrivere domande ma a rispondere a queste, servono pochi step, di solito un'epoca o due sul dataset di conversazioni, la lingua  e le conoscenze vengono dal pretraining, SFT insegna il comprotamento, il formanto, quando fermarsi, il tono

Inoltre per ottimizzare le preferenze si usano DPO e simili, si danno al modello coppie di risposte alla stessa domanda, una migliore e una peggiore e lo si spinge verso l aprima, DPO lo fa direttamente con una loss sulle du eprobabiità, senza ddestrare un modello di reward separato come nel RLHF classico, serve per l'utilità, per la concisione e per rifiutare le richieste dannose, TinyLLama-Chat segue la ricetta Zephyr, SFT su UltraChat e poi DPO su UltraFeedback

Il segreto dei modelli piccoli è la disrtillazione, le conversazioni le scrivono qusi sempre modelli grandi, dataset come UltraChat, OpenHermes o smoltalk (usato per SmolLM2) sono generati o filtrati da LLM grossi. Il modello piccolo impara ad imitare un modello bravo e questo funziona meglio che ocn pochi dati scritti a mano, è la stessa idea dei modelli Phi di Microsoft addestrati su testi da manuale sintetici


Ci sono due lmiti per il nostro caso, i modelli piccoli veri hanno visto molti più dati, TinyLlama (1.1B) è stato preaddestrato su circa 3000 milairdi di token, SmolLM2-135M su qualche migliaio di miliardi, noi su 1, l'SFT da il formato da asssitente, ma le risposte valgono quanto il modello base nel contenuto, MyGPT risponderebbe come un assistente in italiano fluente ma con fatti inventati, poi 256 token di contesto bastano a malapena per una domanda e una rispsota breve, serve arrivare a 1024

Quindi quelo che possiamo mettere in pratica è un modo per unire la conversazione e il retrival, risolve il problema dei fatti, un SFT in cui ogni esempio è "passagio recuperato + domanda -> risposta presa dal passaggio":

```
<|context|>
= Pentium 4 =
Pentium 4 è una serie di processori Intel … prodotti dal novembre del 2000 …<|end|>
<|user|>
Quando è uscito il Pentium 4?<|end|>
<|assistant|>
Nel novembre del 2000.<|end|>
```

domande e risposte si possono generare con un LLM grande a partire dagli incipit di Wikipedia che abbiamo indicizzato, è distillazione ma con la risposta ancorata al testo, il modello impara due cose insieme, il formato da assistente il fatto di leggere dal conesto ivece di inventare, il 3% del regista dovrebbe salir emolto, a runtime la domanda passa per BM25, i passaggi vanno nel contesto e il modello risponde, è un piccolo assistente su wikipedia italiana con le fonti

Solo che il nostro obiettivo non è un modello con rag ma un modello che abbia di fatto assorbito la conoscenza, si può fare ma entro i limiti della capacità del modello

Il problema è che facciamo 0.72 epoche quidi i dati il modello li vede ma una sola volta, un solo pasaggio basta a imaprare la lingua ed il registro ma non a memorizzare "Malamore, 1982, Eriprando Visconti"

Quindi vediamo cosa serve per fare in modo che la conoscenza entri nei pesi


Allen-Zhu & Li (Phisics of Language Models) misurano in modo controllato questa cosa e per farlo servono 3 cose:
1. Molte esposizioni per fatto, con circa 100 esposizioni si arriva a circa 1 bit di conoscenza per parametro, con 1000 a circa 2, noi siamo a meno di 1
2. Diveristà, se un fatto compare sempre nella stessa frase, il modello lo impara come stringa ma non lo sa estrarre quado la domanda è posta in un altro modo, serve vederlo in pià forme
Capacità, cioè parametri, il MoE è la scelt agiusta, per memorizzare contano i parametri totali, non quelli attivi, Jelassi et al (2025, Mixture of Parrots), mostrano che aggiungendo esperti la memoria cresce mentre il ragionamento no, più esperti significa più fatti allo stesso costo per token

Facciamo due conti per capire cosa è realistico...
tutta Wikipedia non entra, solo gli incipit osno 1.15 milairdi di caratteri, circa 320M token, 100 esposizioni farebbero 32 milairdi di token, con l amia GPU sono settimane di training, un dominio intero invece sì, i film sono solo 58748 voci, i loro incipit fanno circa 10M token, 100 esposizioni sono circa 1B token  cioè una run come quelle fatte fino ad ora, come capacità ci sta largamente, sono circa 60k film per 4-5 fatti, richiedono pochi milioni di parametri a 2 bit per parametro e noi abbiamo 163M

Possiamo fare un esperimento controllato sulla conoscenza cinematografica in questo modo:
- estraiamo i fatti dagli incipit dei film con le regex (anno, regista), più paese e genere
- generiamo riformulazioni con template, senza LLM: 4-5 frasi diverse per film più delle coppie domanda-rispsota
- Mix di training, circa 50% wikipedia normale così non dimenica la lingua e circa 50% incipit dei film con le riformulazioni, ripetuti molte volte, è un fine.tune dal checkpoint attuale non da zero
- misurare durante il training col benchmark cloze sui film del train, per vedere la curva "esposizioni -> fatti ricordati" e sui film della val come controllo, questi ultimi devono restare bassi perché i lmodello non li ha mai visti, se salgono stiamo misurando altro


Il bpb sulla val peggiorerà un po' perché dedichiamo capacità ai film è un compromesso, ottimizziamo la memoria a discapito della genralizzazione, se la curva sale come in Allen-Zhu sappiamo che funziona, a quel punto estendere a più domini richiede solamente più capacità, ovvero più eserti o memory layer

---

Abbiamo tre obiettivi separati

| obiettivo | come | quando |
|---|---|---|
| Conoscenza nei pesi | fatti ruoetytu nikte vikte ub nikte firne | pretraining continuato nel checkpoint attuale |
| Comportamento da assistente | conversazioni con il chat template | SFT, poi eventualmente DPO |
| Contesto a 1024 | cambiare block_size, la RoPE regge | durante la prima fase |

C'è un motivo per cui i fatti non si mettono nell'SFT, sembra natrale insegnare i fatti con le conversazioni ma ci sono due effetti contrari, Allen-Ahu dice che se un fatto viene visto dal modello solo nel fine-tuning impara il formato ma non estrae bene la conoscenza, funziona invece quando i fatti e domande-risposte stanno già nel pretraining, in forme diverse, mentre Gekhman et al (2024) dice che fare fine-tuning su fatti nuovi è lento, aumenta le allucinazioni, il modello impara "rispondi con sicurezza anche quando non sai"

La regola che ne viene insegna cosa il modello sa, il post-training insegna come lo dice, l'SFT deve usare fatti che il modello conoce già dandogli la forma desiderata

Dunque la pipeline corretta sarebbe

Prima fase: pretraining continuato dal checkpoint attuale a 1024 token di contesto
- Metà Wikipedia normale per non di dimenticare la lingua
- Metà fatti dal dominio per esempio i 58k film, incipit originali più riformulazioni e domande-risposte in testo semplice ("Chi ha diretto Malamore? Eniprando Visconti") tutto ripetuto molte volte
- Qui entra in gioco l'LLM grande, possiamo usare qwen3.8 27b localmente, ma solo per riscrivere i fatti che vengono da Wikipedia, l'LLM produce gli stessi fatti in salse diverse rifrasando, va verificato che ogni riscrittura contengfa ancora la risposta giusta perché anche i grandi LLM sbagliano, in alternativa si usano dei template meno vari ma senza rischio di errori
- Rimisuriamo col banchmark cloze sui film del train, dovrà salire molto e su quelli della val deve restare basso

Seconda fase: SFT con chat template
- Conversazioni generiche per insegnare il formato, più domande sui film che il modello ha imparato così collega "domanda in chat" con "conoscenza nei pesi"
- Loss solo sulle risposte dell'assistente

Terza fase: DPO (facoltativa), se si vuole rifinire il tono e la qualità delle risposte

Su UltraChat e UltraFeedback c'è un dettaglio, sono in inglese e il nostro modello non lo conosce, dobbiamo usare qualche fork di quei dataset italiani o generarle da un LLM grande direttamente in italiano, è la stessa distillazione ma su misura

In questo senso "impariamo per distillazione" vale due volte, nella fase 1 per la forma diversa dei fatti e nella 2 distilla il comportamento da assistente, i fatti invece restano quelli di Wikipedia

