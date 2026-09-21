LayerNorm e MLP sono due layer indipendenti, entrambi per-token e agiscono su ogni vettore di C dimensioni isolatamente, senza far comunicare le posizioni tra loro. Dopo l'attention che è l'unico posto dove i token si parlano, questa distinsione è netta.

alcuni dettagli sull'implementazione che posso essere trappole


dim=-1, normalizziamo lungo le C feature non lungo B ne T, il vettore del torken in psoizione 3 viene normalizzato guardando solo se stesso, le sue 384 componenti diventano media 0 e varianza 1, il token in psoizione 7 fa lo stesso per conto suo, nessuna statistica condivisa

unbiased=False, divide per N, non per N-1, è la correzione di Bessel, sere a stimare la varianza di una popolazione da un campione, quindi non stiamo stimando niente, stimiamo normalizzando dati che abbiamo tutti, F.layer_norm usa N, se lasciamo il default unbiased=True non crasha, ottengo numeri leggermente diversi e sbagliamo silenziosamente

sqrt(var + eps) non sqrt(var) + eps, l'epsilon va dentro la radice, serve per evitare la divisione per zero quando un vettore ha tutte componenti uguali (succede ad esempio per un embedding non ancora addestrato), dentro la radice il comportamento per var che tende a 0 non da errori, fuori il gradiente di sqrt esplode comunque

weight parte da 1, non da 0.02, è un fattore di scala, non un peso da simmetria da rompere, a 1 e 0, la trasformazione affine finale è l'identità e il layer fa esattamente e solo la normalizzazione, inizializzarlo a 0.02 schiaccerebbe il segnale 50 volte ogni blocco

a cosa servono gamma e beta
forzare l amedia a 0 e varianza 1 è un vincolo, stiamo togliendo alla rete la possibilità di usare la scala e l'offset come informazione, l'affine imparabile gliela restituisce ma sotto controllo, la rete può reimparare una scala diversa per ogni feature, se le serve partendo però da uno stato stimando
perché layernorm e non batchnorm

batchnorm normalizza lungo la dimensione del batch, per ogni feature, media e varianza vengono calcolate su tutti gli esempi, pe rla sequenza è indadatto per diversi motivi
- accoppia gli esempi, l'output per la sequenza 3 dipende da cosa c'era nella sequenza 0, 1, 2, concettualmente assurdo in un modello autoregressivo
- serve uno stato, in inferenza non abbiamo un batch, quindi BN mantiene media mobili aggiornate durante il training e il training ed eval si comportano diversamente, sarebbe un problema
- batch size 1 lo rompe, che è il caso normale di generazione
- il padding lo inquina, sequenze di lunghezza diversa contaminano le statistiche con token fittizi

layernorm non ha senso in questi problemi: nessuna dipendenza dal batch, nessuno stato, comportamento identico in train ed eval, è anche il motivo per cui il nostro model.eval() finora non fa nulla e continuerà finché non aggiungeremo il dropout

c'è una ragione più profonda per cui serve in un transformer, il residual stream accumula somme lungo n_layer blocchi, quindi la sua magnitude cresce strato dopo strato, la layernorm rinormalizza prima che ogni blocco lo legga mantenendo la scala sotto controllo per tutta la profondità, ma lo vedremo dopo

MLP

Il fattore 4 per espansione e ricompressione viene dal paper originale "Attention Is All You Need" ed è rimasto uno standard praticamente universale, è anche dove vivono i parametri 8C^2 per il MLP contro 4C^2 pe rl'attention, circa due terzi di un transformer è MLP, quando sentiamo "modello da 175 miliardi di parametri" la maggioranza sta in questi due Linear nell'MLP

ma cosa fa davvero? la divisione del lavoro nel blocco è netta
- l'attention sposta informazione tra psoizione, è comunicazione
- l'MLP elabora quello che è arrivato, è computazione

senza MLP impilare attention darebbe solo medie pesate di medie pesate, resta un'operazione essenzialmente lineare sui value, e la pofondità non aggiunge potere espressivo, la non linearità è ciò che rende utile impilare.

L'intepretabilità ha una lettura più precisa, l'MLP si comporta come una memoria chiave-valore, le righe di fc sono rivelatori di pattern, si attivano quando il vettore del token somiglia a qualcosa di specifico, come un ricevitore a massima verosimiglianza) le colonne di proj sono i contenuti che vengono riscritti nel residual stream in risposta, gran parte della conoscenza fattuale di un LLM risiede lì dentro, non nell'attention.

GELU invece di ReLU, GELU(x) = x * phi(x) dove phi è la CDF della normale standard cioè `0.5*x*(1 + erf(x*sqrt(2)))` è una relu ammorbidita, invece di tagliare bruscamente a zero pesa x per la probabilità che sia positivo, il vantaggio pratico è il gradiente, la relu non ha gradiente esattamente zero per ogni input negativo, quindi un neruone spinto in territorio negativo può non toranre più indietro (dying ReLI), la GELU mantiene un gradiente piccolo ma non nullo e i neuroni possono recuperare, GPT-2 e GPT-3 la usano

GPT-2 usa un'approssimazione con tanh perché erf all'epoca era lenta (F.gelu(x, approximate='tanh') la riproduce), su hardware moderno la versione esatta va bene, la differenza numerica è dell'ordine 1e-3, noi lasciamo il default

```
uv run src/mygpt/model.py
parametri = 4,356
logits = (4, 8, 66)
loss = 4.1913 (attesa ~ ln(66) = 4.1897)
Head: shape ok, causalita' ok
MHA: shape ok, causalita' ok, 4128 parametri (= 4C^2 + C)
LayerNorm: identica a F.layer_norm, media 0 / var 1 ok
MLP: shape ok, per-token ok (posizione 3 alterata, le altre invariate)
```

facciamo i soliti check, con C=32 LayerNorm 64 parametri, MLP 8*1024 + 160 = 8352
