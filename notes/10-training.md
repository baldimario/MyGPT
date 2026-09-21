Finalmente possiamo passare al training vero e proprio


Weight decay selettico

```
decay = [p for p in raw_model.parameters() if p.dim() >= 2]
no_decay = [p for p in raw_model.parameters() if p.dim() < 2]
```

Il weight decay è una penalità che tira i pesi verso zero, ha senso sulle matrici dei matmul, limita la norma dei pesi, che è una forma di regolarizzazione, non ha senso ed è dannoso sui tensori 1D
- i bias sono pochi e non causano overfitting, decaderli toglie solo gradi di libertà utili
- i gamma della LayerNorm partono da 1 e rappresentano una scala, tirarli verso 0 significa spegnere progressivamente il segnale che attraversa il layer ed è attivamente sbagliato

Il criterio p.dim() => 2 è una euristica che funziona per i lnostro modello perché i tensori 2D sono esattamente i pesi dei matmul e degli embedding, i tensori 1d sono esatamente bias e gain, circa 10.74M in decay cotnro 9k in no-decay

Perché AdamW non Adam, Adam applicherebbe l apenalità L2 sommandola al gradiente, dopo quella somma passa attraverso l anormalizzazione adattiva g/sqrt(v) il risultato sarebbe che i parametri con gradienti grandi hanno v grande, quindi l aloro penalit viene divisa di più, proprio i pesi che vorremmo regolarizzare con decay

AdamW disaccoppia, sottrae lr*wd*p direttamente dal peso, fuori dalla divisione adattiva, il decay diventa uniforme e prevedibile, è il contenuto del paper di Loshchilov & Hutter (2017), ed è il motivo per cui oggi nessuno usa più adam per addestrare Transformers

betas = (0.9, 0.99), beta1 è la media mobile del gradiente (momento, beta2 quella del gradiente al quadrato, GPT-3 usa 0.95 per beta2, qui 0.99 va meglio peché con batch piccoli le stime di secondo momento sono rumorose e una finestra più lunga le stabilizza

Warmup + cosine


```
lr
1e-3 ┤      ╭──────╮
     │     ╱        ╰──╮
     │    ╱             ╰───╮
1e-4 ┤   ╱                   ╰────────
     └───┴──────────────────────────────> iter
         100                        5000
        warmup        cosine no-decay
```

il warmup non è superstizione, Adam normalizza il gradiente per sqrt(v) dove v è una media mobile esponenziale del gradiente al quadrato inizializzata a zero, al passo 1 quella stima è costruita su un singolo campione, il rapporto g/sqrt(v) può essere enorme e assurdo, con un learning rate pieno, i primi passi possono distrugger el'inizializzazione accurata costruita e non si recupera più, la rampa lineare da alle statistiche il tempo di stabilizzarsi

è il motivo per cui il warmup era obbligatorio nei transformers post-LN, e solo fortemente consigliato nei pre-LN, il residual steeam pulito li rende più tolleranti

Il cosine decay vuole lr alto all'inizio (esplorare il passaggio della loss a grandi passi) e basso alla fine (assestarsi nel minimo invece di rimbalzarci intorno), l aforma a coseno rispetto a una rampa lineare passa più tempo agli estremi e meno nel mezzo, empiricamente funziona meglio ed è quello che usano in GPT-3, Chinchilla e LLaMA

Gradient Clipping

```
torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
```

Calcola la norma L2 globale di tutti i gradienti concatenati, se supera 1.0 li scala tutti dello stesso fattore, direzione preservata ma magnitude limitata, serve contro batch patologici, su 1M di token prima o poi pescheremo una finestra strana, il gradiente schizza e uin singolo optimizer.step() con un gradiente 100x ci sposta i pesi a una regione da cui non tornano, la loss salta a 8 e non scende più, con il clipping quel batch fa un passo normale e il trainign continua

bf16 è il motivo per cui non serve GradScaler

```
with torch.autocast(device_type=device, dtype=torch.bfloat16):
_, loss = model(x, y)
```

autocast esegue le operazioni pesanti (matmul, conv) in bf16 e tiene in fp32 quelle sensibili (softmax, layernorm, la loss), i pesi restano fp32, si casta al volo non si converte il modello

bfloat16 ha 8 bit di esponenbte, esattamente come fp32, stesso range dinamico solo mantisse da 8 vs 24 bit, float16 ne ha 5, su valori piccoli i gradienti vanno in u nderflow, quindi fp16 richiede GradScaler che moltiplica la loss per un fattore grande prima del backward e lo divide dopo con la logica di ripiego quando produce inf


Con bf16 tutta quella impalcatura sparisce, e su architettura Blackwell è nativo, la scelta di default pe rl'addestrament moderno, e la semplicazione è reale non teorica


TF32 e non torch.compile
torch.set_float32_matmul_precision("high") autorizza i matmul che restano in fp32 a usare internamente TF32 (10 bit di mantissa, esponente fp32) sui tensor core. Gratis, e con i gradienti non si nota la differenza

torch.compile traccia il modello in un grafo e genera kernel Triton fusi. Il guadagno principale non è sui matmul (già ottimali) ma sulle catene di operazioni elementwise: GELU, dropout, le somme residuali, la LayerNorm. Senza fusione ognuna fa un viaggio andata e ritorno in memoria; fuse, i dati restano nei registri

La prima iterazione sarà lentissima (compilazione, da mezzo minuto a qualche minuto): ignora il primo ms/it stampato. Se dovesse fallire — è l'unico pezzo di questo script che può non funzionare al primo colpo — metti compile_model = False e vai avanti: perdi velocità, non correttezza


Cosa ci apsettiamo dal trainign

La val loss dovrebbe fermarsi intorno a 1.48, contro il 2.48 del bigram. E guarda il gap: la train scenderà sotto 1.2 mentre la val si pianta a ~1.48. Quello è l'overfitting che il dropout sta tenendo a bada — prova a rilanciare con dropout = 0.0 e vedrai la val loss toccare ~1.65 e poi risalire mentre la train continua a scendere. È la dimostrazione sperimentale dello step 7a.

Il testo a 1.48 non sarà inglese vero, ma sarà riconoscibilmente Shakespeare: nomi di personaggi coerenti, struttura in versi, dialoghi con NOME: e a capo, parole quasi tutte pronunciabili, punteggiatura nei posti giusti.

Segnati il ms/it e i k tok/s stabili (dopo la prima eval). Sono la baseline contro cui misureremo lo step 7c: l'attention fusa in un solo Linear(C, 3C) più F.scaled_dot_product_attention, cioè FlashAttention. Te l'ho promessa allo step 4 e la misuriamo, non la diamo per buona.

```
uv run src/mygpt/generate.py --prompt "EDWARD" --tokens 160 --temperature 0.6
# out/ckpt.pt: iter 2000, val loss 1.4679
============================================================
EDWARD:
My name is in a word that same fellow cannot love.

GLOUCESTER:
But strike my son, what now is it not?

CLARENCE:
And why shall I be that shall be loss to see
```
