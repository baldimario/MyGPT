Un tokenizer è una funzione (biezione, bijection) str <-> list[int], niente id più, non ha parametir, non si addestra col gradient descent, non fa parte della rete, è solo un processore.

La scelta del tokenizer impatta due imperparametri per tutto il progetto:

- V (vocab size), determina la dimensione della tabella di embedding (V, C) e il layer finale (C, V), con weight tying è la stessa matrice, ma il costo della softmax finale diventa O(B*T*V)
- T (quanti token servono per dire una cosa), l'attention costa O(T^2), raddoppiare T significa quadruplicare il costo dell'attention.

Quindi sono in trade-off diretto, vocabolario più piccolo -> sequenze più lunghe.

| schema | V | "il gatto dorme" | pro | contro |
|---|---|---|---|---|
| char-level | ~65-200 | 15 token | banale, mai OOV | sequenze lunghissime, il modello spreca capacità ad imparare l'ortografia |
| word-level | 50k-500k+ | 3 token | sequenze corte | OOV, vocab enorme, morfologia ignorata (gatto/gatti non correlati) |
| BPE (subword) | 50257 (GPT-2/3) | ~4 token | compromesso, mai OOV, morfologia parziale | va addestrato, implementazione più complessa |

Per sentire il peso parliamo di numeri concreti, GPT-3 Small ha V=50247, C=768, la sola tabella di embedding è 50257*768=38.M parametri, sui 125M totali. Il 31% del modello è il tokenizer. Nei modelli grandi la frazione crolla (in GPT-3 175B è lo 0.1%) ma nel nostro modellino da 5M sarà dominante, motivo per cui converebbe partire con un tokenizer char-level.

Implementare un tokenizer char-level ci toglie anche una variabile di mezzo, con V=65 la tabella di embedding è irrilevante e possiamo concentrarci sul transformer, BPE lo implementiamo dopo con la stessa API così da poterlo swappare senza cambiare il resto del codice.

Il tokenizer sarà una classe in cui avreom due dizionari per il lookup str -> int e quello inverso int -> str con due metodi encode(s: str) -> list[int] e decode(ids: list[int]) -> str, niente torch, python puro. Possiamo conservare due dizionari stoi: dict[str, int] e itos: dict[int, str] (o list[str]) per un lookup veloce

Alcuni punti di controllo su cui possiamo prendere decisioni
- Come ordiniamo il vocabolario, set(text) non ha ordinamento, se cambia il vocabolario invalidiamo tutto il training fatto su un vocaboalrio con ordinamento diverso, serve un ordinamento deterministico
- itos è dict o list? le chiavi sono 0..V-1 contingue, usare un dict o una lista può darci vantaggi e svantaggi, forse list è meglio sopravvive meglio alla serializzazione json
- Caratteri mai visti in encode, il vocabolario viene dal training set, se poi gli diamo un caratter emai visto come una emoji o una lettera accentata strana e non c'è in stoi[c] avremo un KeyError nel tokenizer, possiamo lasciare che fallisca, possiamo avere un token <UNK> o normalizzare il testo a monte, magari <UNK> id 0 indipendente dal corpus
- Il vocab va salvato? il modello impara id -> significato, se ricostruiamo il tokenizer da un testo diverso gli id si spostano e il checkpoint è da buttare, serve serializzarlo, sarebbe utile serializzarlo come json


Il corpus
Lo standard di fatto per questo tipo di toy llm è un dataset di ~1-2MB da 65 caratteri unici chiamato tinyshakespear.
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

Ovviamente va bene qualunque txt da peso >= 1MB, un testo italiano dal progetto Gutenberg anche funziona bene, avrò solo più caratteri a causa degli accenti magari

Il check
una sola asserzione ed è l'unica invariante che definisce un tokenizer assert tok.decode(tok.encode(text)) == text, questa identità deve essere sempre rispettata o non è una funzione biunivoca.

CharTokenizer is implemented in src/mygpt/data.py on tinyshakespeare it shows vocab_size of 66 = 65 + UNK
uv run src/mygpt/data.py
vocab_size = 66
"\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

