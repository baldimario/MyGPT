Prima del training vero e proprio c'è un problema aritmetico da affrontare, 10.7M parametri su 1M token, dieci parametri per token di dataset, questo modello può memorizzare shakespeare a memoria e lo farà, overfitting, vedremo la train loss continuare a scendere mentre la val los risale, è l'oppoto del bigram che non poteva overfittare nemmeno volendo.

Note su llayer di Dropout
self.trainign è l'attributo che nn.Module.train() e .eval() ribaltano, ricorsivamente su tutti i sottomoduli, fino a ora nessun layer lo leggeva e model.eval() in train.py non faceva nulla, ora sì.

Il /(1-p) è la parte che di solito si sbaglia, se azzeriamo il 20% delle ativazioni, la somma che arriva al layer successivo è mediamente 20% più piccola, in inferenza non azzeriamo niente, quindi il modello riceve input sistemicamente più grandi di quelli su cui è stato addestrato, e tutte le sue calibrazioni saltano, dividendo per 1-p il valore atteso resta invariato e le due fasi combaciano, si chiama inverted dropout (inverted perché la correzione sta in training invece che in inferenza) ed è il default dal 2014 ovunque

.to(x.dtype) e non .float() perché useremobf15, se forzo le maschere a float32 la moltiplicazione promuove tutto a float32 e perdo il vantaggio della precisione ridotta

Nel paper GPT-2/3 ci sono 3 punti, aggiungono un parametro dropout: float = 0.0 alle firme di Head, MultiHeadAttention, MLP, Block, GPT e lo passiamo giù, default 0.0 è voluto per tuti gli assert esistenti che così continuano a funzionare

GPT-3 usa 0.1 come valore di dropout, ma su 300 miliardi di token con 175 miliardi di parametir, qui 0.2 va bene, i lrapporto è rovesciato rispetto al nostro, la regolarizzazione gli serve molto meno, a 0.2 su questa configurazione la val loss si ferma intorno a 1.48, a 0.0 succede a ~1.65 poi risale


facciamo il check per verificare sperimentalmente che /(1-p)  faccia il suo lavoro, i test di causalità e per token rimangono con dropout=0.0 (default)

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
Dropout: identita' in eval, ~20% azzerato in train, media 1.0 ok
```
