Adesso a ogni token rifacciamo il forward su tutto il contesto e ricalcoliamo k, v per token che non cambieranno mai più, 256 volte lo stesso lavoro, possiamo evitarlo implementando la KV cache

Ci sono tre trappole
1. is_causal=True in decode è il classico bug, la maschera di SDPA è ancora in alto a sinistra non in basso a destra, con la q lunga 1 e k lunga 200, l'unica cella permessa è la prima e il modello attende solo al token 0, non crasha ma genera garbage, per questo l ariga is_causal=q.size(2) == k.size(2) che è True nel prefill (qadrata, serve), False nel decode (rettangolare, nonm serve niente perché il token nuovo è l'ultimo e può vedere tutto)
2. Le posizioni, senza cache passiamo tutta la finestra e arange(T) va bene, con la cache passiamo un token solo arange(1) gli darebbe la posizione 0, cioè "sono l'inizio del testo" ad ognis ingolo passo, il t0 letto dalla cache è quelo che glielo impedisce
3. Oltre block_size la cache è un'approssimazione, non è più esatta, tagliamo i token vecchi ma quelli che restano si portano dietro la wpe della posizione che avevnao quando sono entrati mentre la versione senza cache ricalcola tutto traslato. Il testo resta buono, i due patch divergono. Non è pigrizia, l'embedding posizionale appreso assoluto non è traslabile, non c'è niente da fare. è il motivo per cui i modelli moderni usano RoPE o AliBi dove l aposizione è relativa e la cache si può scorrere senza mentire, è uno dei motivi per cui GPT-2 è invecchiato.

```
uv run src/mygpt/model.p
y
parametri = 4,356
logits = (4, 8, 66)
loss = 4.1913 (attesa ~ ln(66) = 4.1897)
Head: shape ok, causalita' ok
MHA: shape ok, causalita' ok, 4128 parametri (= 4C^2 + C)
LayerNorm: identica a F.layer_norm, media 0 / var 1 ok
MLP: shape ok, per-token ok (posizione 3 alterata, le altre invariate)
Block: shape ok, causalita' ok, modifica il residual stream del 1.45%
GPT: 10,764,288 parametri (formula ok)
loss init = 4.2446   (ln(66) = 4.1897)
GPT: generate ok
GPT: KV cache ok
Dropout: identita' in eval, ~20% azzerato in train, media 1.0 ok
``` 

i check sono ok sulla kv cache, verifichiamo quanto guadagnamo in velocità considerando che è single batch

Senza kv cache 426 tok/s

```
uv run src/mygpt/generate.py --prompt "EDWARD" --tokens 160 --temperature 0.6 --seed 1 --no-cache
# out/ckpt.pt: iter 2000, val loss 1.4485
# 160 token in 0.38s = 426 tok/s (no cache, batch 1)
============================================================
EDWARD IV:
The benefit hath been banish'd times made me from the friend.

GLOUCESTER:
But what man can scarce this last fast?

CLARENCE:
How now! the did I not speak 
```

Con kv-cache 3647 tok/s
```
uv run src/mygpt/generat
e.py --prompt "EDWARD" --tokens 160 --temperature 0.6 --seed 1 --samples 8
# out/ckpt.pt: iter 2000, val loss 1.4485
# 1280 token in 0.35s = 3647 tok/s (kv cache, batch 8)
============================================================
EDWARD IV:
The benefit hath been banish'd times made me from the friend.

GLOUCESTER:
But what man can scarce this last fast?

CLARENCE:
How now! the did I not speak 
============================================================
EDWARD:
They say, so 'twas straight a while.

GLOUCESTER:
Why, what was the way with the world?

HASTINGS:
More than that which I will live.

GLOUCESTER:
My lord, the
============================================================
EDWARD IV:
For the devil that we have stay'd to London.

PRINCE EDWARD:
The statute that we may see the bottom,
And therefore it were, that the end, I would do not be
============================================================
EDWARD IV:
Now, by the loss of Lancaster, they are but false
As I may be sent to their beds,
As the more that stops them say their and where
Were the plots of present
============================================================
EDWARD IV:
Why, this is the great king, and so the troop
That then he was a score foul match from her scoler.

JOHN OF GAUNT:
With all grief that thou hast fought to 
============================================================
EDWARD IV:
I do beseech you, what will you stay?

BUCKINGHAM:
I beseech you, sir, my lord.

KING RICHARD III:
Have you to my heart that he is dead?

TYRREL:
I thought
============================================================
EDWARD IV:
What art thou?

PRINCE EDWARD:
My lord, the king of York the law that made him.

YORK:
And I hear thee, thou shalt be here to me,
But what far thou wast an
============================================================
EDWARD IV:
Richard, as I was the king of Clarence,
But shall I do assout the blood of Hereford,
That would thou might her with his fortune and the king
Who hath perfo
 localghost@acquarium  ~/work/iworkonit/mygpt   master 
```
