Abbiamo tutti i pezzi che ci servono, Embedding, Linear, LayerNorm, Head, MultiHeadAttention, MLP, Block, dobbiamo solo assemblare, impilare n_layer blocchi, aggiungere wpe, ln_f, lm_head, il weight tying e quello scaling dell'init e possiamo fare il primo training vero e proprio

wpe, la posizione va iniettata a mano
l'attention è permutazione-equivalente sul passato, l'avevamo anticipato, rimescolare l'ordine dei token e le affinità q*k sono identiche, il meccanismo sa cosa c'era prima ma non in che ordine, da solo il tasforme rlegge il testo come un sacchetto di parole

la soluzione di GPT è la più diretta possibile, una sola tabella di lookiup, block_size*n_embd, che mappa l'indice di posizione in un vettore.
Posizionale imparato, non sinusoidale come nel paper del 2017, GPT-1/2/3 usano tutti questro, unziona uguale ed è più semplice.

Il prezzo è un tetto rigido, wpe ha esattamente block_size righe e quindi il modello non può vedee una posizione mai incontrata. Niente estrapolazione, è il motivo per cui i modelli moderni (LLaMA, Mistral) usano RoPE, che codifica la posizione come una rotazione dipendente dall'offset relativo e quindi estrapola oltre la lunghezza di training

perché si somma invece di concatenare? domanda lecuta, sommare "sei il token 42" e "sei in psoizione 7" sembra distruggere informazione, ma il residual strtream ha larghezza fissa C, e concatenare la raddoppierebbe, in pratica il modello impara ad allocare sottospazi diversi ai due segnali, in C=384 dimensioni c'è spazio abbondante per tenerli separabili, è la stessa logica per cui un blocco può scrivere sul bus senza cancellare quello che c'era, sommare in alta dimensione è soprendentemente non distruttivo.

Riguardo il weight tying

```
self.lm_head.weight = self.wte.weight
```

non è una copia, è proprio lo stesso tensore, per referenza, questo funziona perché le shape combaciano ed è esattamente questo per cui abbiamo insistito sulal convenzione (out, in) in Linear, wte.weight è (V, C) e Linear(C, V).weight è (V, C)

il motivo per cui è semanticamente giusto e non è solo un trucco per risparmiare è che la riga i di quell amatrice è "il vettore che rappresenta il token i" e in ingresso la usiamo per chiedere quale è il vettore del token i e la leggiamo, in uscita la usiamo per chiedere quanto la posizione assomiglia al token i, ci facciamo il prodotto scalare

sono la stessa domanda vista da due lati diversi. se due token hanno significato simile devonmo avere embedding simili e devono ricevere logit simili, legarli lo impone per costituzione e il gradiente arriva sulla setessa matrice da entrambe le direzioni

il risparmio è reale perché V*C parametri, per GPT-2 Small sono 38.6M su 125M, il 31% del modello,m nel nostro caso 66*384 = 25k, poco ma la ragione resta concettuale.


Invece riguardo lo scaling dell'init recuperiamo con named_parameters() le matrici di proiezione che finsicono con "proj.weight" che resituisce ualcosa come "blcoks.3.attn.proj.weight" e "blcoks.3.mlp.proj.weight" cioè le uniche due proiezioni che scrivono sul residual stream, il filtro sul suffisso ne becca esattamente 2*n_layer, le inizializziamo a 0.02/math.sqrt(2*n_layers), in questo modo non passiamo n_layer a ogni Block


Riguardo il crop in generate

```
idx_cond = idx[:, -self.block_size:]
```

senza questo appena generiamo il token numero block_size+1 chiamiamo wpe con un indice fuori range e abbiamo IndexError, con il crop, il modello genera testo di lunghezza illimitata tenendo però una finestra scorrevole, al token 1000 ha completamente dimenticato i primi 744, non c'è memoria oltre block_size. Quando si sente parlare di context window di un LLM è letteralmente questo numero.


Infine riguardo temperatura e top_k, sono due manopole sulla distribuzione finale,m entrambe a costo zero, nessuna delle due tocca il training:
- temperature divide i logits prima della softmax, < 1 amplifica le differenze modello conservativo, ripetitivo e grammaticalmente solido, > 1 appiattisce e il modello è più creativo e incoerente, a 0.01 è praticamente un argmax determinsitico
- top_k azzera tutto tranne i k logit più alti prima di normalizzare, elimina la coda lunga, senza questo a ogni passo c'è una probabilità piccola ma non n ulla di pescare un carattere assurdo e su 500 passi succede quasi sicuramente

Guardiamo i check

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
GPT: 10,764,288 parametri (formula ok)
loss init = 4.3500   (ln(66) = 4.1897)
GPT: generate ok
```

12*C^2 per layer perché 4C^2 di attention più 8C^2 di MLP, con 6 Layer da 384 sono 10.6M dei 10.76 totali, il 98%. Embedding e posizionali sono briciole a questa scala, in GPT-2 Small con V=5257 il rapporto si ribalta.

La loss iniziale è ~4.28, non 4.19, sopra ln(V) e non è un bug, è un effetto del weight tying, i logits valgono ln_f(x) @ wte.weight.T e dopo la LayerNorm finale x ha componenti di varianza 1 quindi i logit hanno deviazione standard `~ 0.02*sqrt(384) ~ 0.39` invece di ~0, non sono perfettamente uniformi e ogni deviazione dall'uniforme costa loss.
C'è anche un secondo effetto, il residual stream contiene ancora traccia di wte[idx] e dotarlo con al stessa matrice alza il logit del token corrente, i lmodello all'init tende a predire "si ripete lo stesso carattere" che è sbagliato, da qui la tolleranza a 0.4 invece di 0.1
