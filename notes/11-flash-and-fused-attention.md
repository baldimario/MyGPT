Dopo il trainig possiamo vedere che genera, mancano un paio di cose, fused attention e flash attention
Un solo Linear(C, 3C) al posto di 3*n_head inear separate e F.scaled_dot_product_attention al posto del softmax materializzato

3C^2 + C^2 + C = 4C^2 + C, stessi identici parametir, la formula dei 10764288 non cambia, il guadagno è tutto mnl kernel lauch e memoria

Ci sono però dei punti a cui fare attenzione
- l'ordine dello split è q, k, v quindi nel test le righe 0:C sono q, non k, se invertiamo allclose fallisce
- dropout_p va a 0 in eval a mano, non guarda self.training è una funzione pura, il modello campiona dropout attivo e il testo peggio altrimenti
- Le chiavi dello state_dict cambiano (sa.heads.key.weight -> sa.c_attn.weight), il nostro checkpoint id training a 1.4679 non si caricherà più

Bisogna quindi rilanciare il training e possiamo confrontare ms/it staibili (dopo la prima eval, la prima iterazione è tutta compilazione)

```
uv run src/mygpt/train.py
10,764,288 parametri
  decay:     26 tensori, 10,740,480
  no decay:  44 tensori,     23,808
W0921 02:52:54.719000 12255 torch/_inductor/utils.py:1953] [0/0] Not enough SMs to u
se max_autotune_gemm mode
step     0 | train 4.2448 | val 4.2473 | lr 9.90e-06 |    nan ms/it |     0k tok/s
step   250 | train 1.9421 | val 2.0600 | lr 9.98e-04 |   76.1 ms/it |   215k tok/s
step   500 | train 1.5232 | val 1.7118 | lr 9.85e-04 |   49.9 ms/it |   328k tok/s
step   750 | train 1.3573 | val 1.5768 | lr 9.61e-04 |   49.9 ms/it |   329k tok/s
step  1000 | train 1.2637 | val 1.5135 | lr 9.27e-04 |   49.8 ms/it |   329k tok/s
step  1250 | train 1.1942 | val 1.4844 | lr 8.83e-04 |   49.8 ms/it |   329k tok/s
step  1500 | train 1.1414 | val 1.4745 | lr 8.31e-04 |   49.8 ms/it |   329k tok/s
step  1750 | train 1.0890 | val 1.4658 | lr 7.71e-04 |   49.9 ms/it |   329k tok/s
step  2000 | train 1.0403 | val 1.4710 | lr 7.05e-04 |   48.2 ms/it |   340k tok/s
step  2250 | train 0.9883 | val 1.4740 | lr 6.36e-04 |   49.9 ms/it |   328k tok/s
step  2500 | train 0.9424 | val 1.5016 | lr 5.64e-04 |   49.8 ms/it |   329k tok/s
step  2750 | train 0.8908 | val 1.5215 | lr 4.92e-04 |   49.8 ms/it |   329k tok/s
step  3000 | train 0.8432 | val 1.5437 | lr 4.22e-04 |   47.4 ms/it |   345k tok/s
step  3250 | train 0.7967 | val 1.5733 | lr 3.55e-04 |   48.4 ms/it |   339k tok/s
step  3500 | train 0.7514 | val 1.5919 | lr 2.93e-04 |   53.3 ms/it |   307k tok/s
step  3750 | train 0.7148 | val 1.6101 | lr 2.37e-04 |   54.1 ms/it |   303k tok/s
step  4000 | train 0.6794 | val 1.6460 | lr 1.89e-04 |   53.7 ms/it |   305k tok/s
step  4250 | train 0.6499 | val 1.6692 | lr 1.51e-04 |   53.7 ms/it |   305k tok/s
step  4500 | train 0.6236 | val 1.6921 | lr 1.23e-04 |   53.7 ms/it |   305k tok/s
step  4750 | train 0.6065 | val 1.7040 | lr 1.06e-04 |   53.7 ms/it |   305k tok/s
step  5000 | train 0.5901 | val 1.7199 | lr 1.00e-04 |   53.6 ms/it |   306k tok/s

miglior val loss: 1.4658  ->  out/ckpt.pt
============================================================

The noble Capulet,
See this music o' the city; supering me
That our speeches which was privy
And he shall for his wit.

Second Lord:
So did you see that time will shame to God,
To obsequiously lawful maidenhood done.

Lord Mayor:
Good my lord, the grand to be entreated well.

BUCKINGHAM:
And let this give learn, when I will purchase them.

PRINCE EDWARD:
Repent is for wilt, and betide to fight.

BUCKINGHAM:
I do beseech your grace to speak with them,
For I will hope you depart to with content
That let this less deign before I let me do.

GLOUCESTER:
I did; but yet I will say what I am.

YORK:
I will, my lord.

GLOUCESTER:
Come to me, Lord William: to-morrow night:
I thought my soul to help your condition.

KING EDWARD IV:
How will my cousin Buckingham I speak?

GLOUCESTER:
But I in tell you, what do you all me,
That I may sleep a mighty life and mine,
That I am the disgrace that enter'd youth
And what you will serve. But can you refuse it?
It is not your will: but your grace gods
From 
```

Confrondtando l aversione con al HeadAttention era a 67.13 ms/it 244 tok/s peak 2.54 GB, la fuse attention + SDPA 50.51 ms/it 324k tok/s peak 1.77 GB, quindi 1.33x performance -30% memoria di guadagno

Per 1.33x non è FlashAttention che fa il miracolo, a T=256 la matrice TxT è ancora piccola, il grosso è che facciamo una sola matmul da (C, 3C) invece di 18 matmul da (C, 64) più una cat, meno kernel launch, GEMM più grosse, GPU più occupata, il -30% di memoria occupata  tutto SDPA, la (B, nh, T, T) non viene mai materializzata, è quello che ci fa scalare block_size senza esaurire la VRAM

Sul training overfittiamo praticamente
```
val   1.4658 @ 1750  <- minimo
      1.4710 @ 2000
      1.5437 @ 3000
      1.7199 @ 5000  (train 0.5901)
```

il minimo è 1750 su 5000, dopo la train scende di un 45% ulteriore e la val risale monotona, il modello smette di imparare l'inglese e inizia a memorizzare il dataset tinyshakespeare, considerando che salviamo i checkpoint solo se hanno val migliore del precedente, non salviamo il checkpoint quando val risale e ha overfittato.

Questo significa che 3500 iterazioni dopo il minimo è GPU sprecata, producono un modello peggiore, il droppout 0.2 ha rallentato la cosa, senza avevamo ~1.65 e saremmo arrivati molto prima al valore minimo, non l'ha impedito però, 10.7M parametri su 1M token sono troppi

Il fix più economico è un numero max_iters = 2000, non è fermarsi prima, è allineare il cosine della finestra utile, così il lr annealing fa il suo lavoro proprio dove la val è al minimo invece di scendere a 1e-4 mentre memorizziamo, ci aspettiamo circa ~1.44 in un terzo del tempo, oltre quello siamo data-bound, il dropout 0.3 o un modello più piccolo mi danno decimali, non salti di qualitù, il salto vero è più corpus

 ```
uv run src/mygpt/train.py
10,764,288 parametri
  decay:     26 tensori, 10,740,480
  no decay:  44 tensori,     23,808
step     0 | train 4.2448 | val 4.2473 | lr 9.90e-06 |    nan ms/it |     0k tok/s
step   250 | train 1.9491 | val 2.0663 | lr 9.86e-04 |   51.7 ms/it |   317k tok/s
step   500 | train 1.5217 | val 1.7088 | lr 9.05e-04 |   49.8 ms/it |   329k tok/s
step   750 | train 1.3511 | val 1.5586 | lr 7.64e-04 |   49.9 ms/it |   329k tok/s
step  1000 | train 1.2569 | val 1.5099 | lr 5.87e-04 |   49.9 ms/it |   328k tok/s
step  1250 | train 1.1818 | val 1.4733 | lr 4.04e-04 |   49.9 ms/it |   328k tok/s
step  1500 | train 1.1284 | val 1.4523 | lr 2.45e-04 |   49.9 ms/it |   329k tok/s
step  1750 | train 1.0880 | val 1.4568 | lr 1.38e-04 |   49.8 ms/it |   329k tok/s
step  2000 | train 1.0620 | val 1.4485 | lr 1.00e-04 |   49.7 ms/it |   329k tok/s

miglior val loss: 1.4485  ->  out/ckpt.pt
============================================================

The liege of this execution, and as I can
but the cold first that will come my throne.

LUCIO:
Friar, if he be my brother be here in her hand;
but when he bids the point of Perdita,
who had made you sure to desire his daughter kindness
died; for, he shall be more here scarce to his speech.

DUKE VINCENTIO:
He is a herd with his character, he hath his power
son as it: may you are prize our eyes a lappier.

Provost:
This is it for you are very fit
of any faith, and he cannot please you to do.

DUKE VINCENTIO:
He hath them?

Provost:
No, you are gound in a happy brother.

BRUTUS:
You are come to the people's one about: if they can
say, it were he, and that I cannot be avoided.

DUKE VINCENTIO:
Come, poor maid, and I do not say it well.

MARCIUS:
I told you the Master Henry is the sea-meet; yet, sir, and
The pattern of Hermione, or it, one that are recorded
In exile,
When did some worse that worst to stay by the lists
I should all the renowned darter. But, I will be
Marry to prove her but 
 ```
