Perchè una testa non basta

Una head produce una matrice (T, T), ogni posizione ottiene una distribuzione di probabilità sul passato, ma un token ha bisogno di relazioni simultanee e di tipo diverso


torniamo all'esempio "il gatto nero dorme" il token "dorme" deve contempoeaneamente trovare il soggetto (gatto) per l'accordo verbale, sapere che nero è u nmodificatore di gattom non suo, tenere traccia del tema del discorso, molte parole indietro

con una sola softmax queste richieste collassano in una media unica, se "dorme" mette a 0.5 su gatto e 0.5 su nero, il value che raccoglie è una poltiglia di due, e l'informazione "chi è il soggetto" è persa.

Con più teste, ognina calcola la sua matrice di affinità, indipendentemente. La testa 1 può fare accordo soggetto-verbo, la testa 2 legame aggettivo-nome, testa 3 tema globale, i risultati si concatenan oinvece di mediarsi.

Non è teoria, l'interpetabilità meccanicistica ha identificato teste specializzate in modelli reali, previous-token heads (guardano sempre t-1), induction heads (trovano la precedente occorrenza del token corrente e copian ocosa la seguiva, il meccanismo alla base dell'in-context learning), teste sintatiche, ecc... emergono da sole, nessuno le progetta.

le teste sono gratis
head_size = n_embd / n_head
con n_embd=384 e n_head=6 6 teste da 64 dimensioni ciascuna, concatenate 6* 64 * 384 = n_embd, la larghezza del residual stream non cambia.

Il conto dei parametro è il punto, per ogni testa 3*(n_embd*head_size) per n_head teste si ha 3*n_head*n_embd*(n_embd/n_head) = 3*n_embd^2

n_head si semplifica, una testa da 384 e sei teste da 64 hanno esattamente gli stessi parametri e quasi lo stesso costo computazionale, non stiamo comprando capacità, stiamo partizionando lo stesso budget in sottospazi indipendenti, è una ristrutturazione gratuita che fa funzionare meglio tutto, raramente in deep learning si trova qualcosa di così forte, gratis e netto

ora che sappiamo di cosa si tratta possiamo implementare il layer MultiHeadAttention.

nn.ModuleList, non è una lista python ovviamente, come nn.Parameter, vanno su cuda, sono registrati e l'ottimizzatore li tocca.

A cosa serve elf.proj? non è di certo decorativa, dopo il cat le dimensioni :65 sono solo output della testa 0, le 64:128 sono della testa 1 e così via, sono settori separati, senza la proiezione ogni testa scriverebbe sempre sulla sua fetta riservata del residual stream senza poter cambiare i risultati ne scelgiere dive depositarli, la proiezione mescola, è il layer che permette al modello di dire "quello che ha trovato la testa 2 va sommato a quello che ha trovato la testa 5, e scritto in queste dimensioni"

ha anche un secondo ruolo, ma lo vedremo dopo, è l'ultima operazione prima della somma residuale, funziona da valvola con cui il blocco decide quanto scrivere sul residual bus, va inizializzata piccola, all'inizio il blocco è quasi un'identità, per questo i transformer profondi si addestrano

```
uv run src/mygpt/model.py
parametri = 4,356
logits = (4, 8, 66)
loss = 4.1913 (attesa ~ ln(66) = 4.1897)
Head: shape ok, causalita' ok
MHA: shape ok, causalita' ok, 4128 parametri (= 4C^2 + C)
```

L'ultimo assert aggiunto è importante, se cambiamo n_head a 1, 2, 8, 16 e rilanciamo il numero rimane lo stesso, è la dimostrazione operativa che le teste non costano niente

Questa versione lancia 3*n_head matmul piccoli per layer (uno per proiezione per testa), più il cat. le implementazioni vere fanno solo un Linear(C, 3C) che produce q, k, v in tutte le teste insieme, poi un view + transpose riorganizza (B, T, 3C) in (B, n_head, T, hs) matematicamente è identico ma su gpu la differenza è sostanziale, per matmul piccoli il costo è lanciar eil kernel non il calcolo, è una piccola ottimizzazione che possiamo fare dopo
