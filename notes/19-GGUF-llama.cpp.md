Non dovrebbe essere complicato creare i pesi in GGUF per girare su llama.cpp, la nostra architettura è molto simile a Mixtral ormai e llama.cpp la supporta nativamente, il punto delicato è il tokenizer, non i pesi

Quello che combacia per ora è

| MyGPT | |llama.cpp (architettura llama con esperti tipo Mixtral) |
|---|---|---|
| RMSNorm, eps 1e-5 | si |
| SwiGLU senza bias | si, ffn_gate, ffn_up, ffn_down |
| MoE: softmax -> top-k -> rinormalizazione | si, è esattamente il gating di Mixtral |
| RoPE, theta 10000 | si |
| embedding legati a lm_head | si, se manca output.weight usa token_embd |
| head dim 64, hidden 1408, vocab 8192 | si |

Cosa va adattato nei pesi come conversione meccanica?

- c_attn fuso va diviso in attn_q, attn_k e attn_v, tre fette di una matrice
- RoPE a metà contro RoPE interleaved, MyGPT ruota le coppie (i, i+hs/2) come rorate_half di HF, llama.cpp si aspetta coppie adiacenti (2i, 2i+1), si risolve permutando le righe di Wq e Wk, lo script ufficiale di llama.cpp fa la stessa cosa per i modelli Llama di HF
- il bias di proj nell'attention, Llama non lo ha, llama.cpp accetta attn_output.bias opzionale, se non dovesse accettarlo bsiogna toglierlo e riaddestrare o forkare llama.cpp per supportarlo
- il router, router.weight diventa ffn_gate_inp, gli 8 esperti vanno impilati in un unico tensore per layer (ffn_gate_exps ecc)

Il punto difficile resta il tokenizere

Il vocabolario e le merge si esportano facilmente, il problema è la pre-tokenizzaione, llama.cpp non accetta una regex arbitraria, ha un elenco fisso di pre-tokenizer che prende dal nome (gpt2, llama3, qwen2 e così via), la nostra regex è r" ?\w+| ?[^\w\s]+|\s+", ci sono due strade

- usare gp2 che ha una regex simile ma non identica, separa i numeri, gestisce le contrazioni inglesi e tratta diversamente gli spazi finali, funziona ma alcune stringhe potrebbero risultare spezzate in modo diverso da come le ha viste il modello nel training e la qualità calerebbe un po'
- aggiungere il nostro pre-tokenizer a llama.cpp con un fork

Analizzando il sorgente di llama.cpp possiamo confermare che RoPE è confermato, LLAMA usa ROPE_TYPE_NORM con coppie adiacenti, quindi la permutazione di Wq e Wk serve, attn_output.bias è supportato come opzionale quindi non si deve trainare nuovamente per questo, il gating MoE softmax con norm_w=true corrisponde al nostro caso e la regex del pre-tokenizer gpt2 sembra simile alla nostra

Il layout coincide con quello di HF (rotate_halp con coppie i, i+hs/2 per le head) quindi si può riusare la stessa permutazione, vanno solo recuperati i nomi nello state dict e capire llama.cpp come gestisce expert_weights_scale, ma sembra esserci tutto quello che serve

la conversione die pesi è semplice e riesce, la differenza PPL Pytorch/llama.cpp è solo dello 0.01% entre il pre-tokenizer gpt2 ci costa +8% in bpb rispetto al nostro, il 93% delle differenze deriva da un'unica causa, la regola `\s+(?!\S)` di GPT-2 spezza "\n\n" in due "\n" separati mentre nel nostro vocabolario lo tratta come otken unico, purtroppo nessun pre-tokenizer di llama.cpp riproduc  quello di MyGPT sena toccare il codice C++

la quantizzazione Q8_0 e Q4_0 è il motivo pirncipale per cui vogliamo usare GGUF e la conversione riesce, dobbiamo consciamente sapere che il pre-tokenizer gpt2 ci fa perdere l'8% di qualità

| | PPL |
|---|---|
| llama.cpp F32 | 15.0640 |
| Pytorch su stessi token | 15.0653 |

lo scarto è dello 0.01% ed è solo dovuto al rounding dei float, una permutazione sbagliata avrebbe dato numeri di ordine di grandezza diversi

| stesso moello e stesso testo | PPL | bpb |
|---|---|---|
| token del tokenizer MyGPT | 12.56 | 0.996 |
| token di llama.cpp (gpt2) | 15.06 | 1.076 (+8%) |


Il 93% delle differenze viene dalla regola di GPT-2 di spezzare "\n" in "\n" + "\n", wikipedia ha un doppio a capo ogni paragrafo e il modello non ha mai visto quella sequenza, il resto è marginale: "5°" diventa 5 + ° e compaiono le contraddizioni inglesi come "'s", tra i pre-tokenizer di llama.cpp nessuno si comporta come il nostro

Riguardo la quantizzazione

| formato | dimensione | PPL |
|---|---|---|
| F32 | 604 MB | 15.064 |
| Q8_0 | 145MB | 15.067 (identico) |
| Q4_0 | 85 MB | 16.195 (+0.9) |

I K-quant (Q4_K_M e simili) non sono disponibili, la dimensione interna degli esperti non è un multiplo di 256, noi abbiamo 1408, in generazione Q8_0 va circa a 2150 tok/s contro i 116 del codice Pytorch in Python

Per recuperare l'8% perso ci sono due strade quindi, o patchiamo llama.cpp o riaddestriamo il tokenizer con la regex di GPT-2 e ritokenizziamo il corpus e riaddestriamo il modello, sarebbe la strada più pulita e ci da la compatibilità completa anche se costa quasi 5 or edi training


```
~/llm/llama.cpp/build/bin/llama-completion -m out/mygpt-moe-Q8_0.gguf -p "Il processore Pentium 4" -n 150 --temp 0.7 --repeat-penalty 1.15 -no-cnv
0.00.105.884 I llama_completion: llama backend init
0.00.105.899 I llama_completion: load the model and apply lora adapter, if any
0.00.105.912 I common_init_result: fitting params to device memory ...
0.00.105.913 I common_init_result: (for bugs during this step try to reproduce them with -fit off, or provide --verbose logs if the bug only occurs with -fit on)
0.00.216.198 W load: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
0.00.239.018 I common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)
0.00.262.860 I llama_completion: llama threadpool init, n_threads = 8
0.00.262.873 I
0.00.262.910 I system_info: n_threads = 8 (n_threads_batch = 8) / 32 | CUDA : ARCHS = 1200 | USE_GRAPHS = 1 | PEER_MAX_BATCH_SIZE = 128 | BLACKWELL_NATIVE_FP4 = 1 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 |
0.00.262.911 I
0.00.262.954 I sampler seed: 4048025062
0.00.262.960 I sampler params:
        repeat_last_n = 64, repeat_penalty = 1.150, frequency_penalty = 0.000, presence_penalty = 0.000
        dry_multiplier = 0.000, dry_base = 1.750, dry_allowed_length = 2, dry_penalty_last_n = -1
        top_k = 40, top_p = 0.950, min_p = 0.050, xtc_probability = 0.000, xtc_threshold = 0.100, typical_p = 1.000, top_n_sigma = -1.000, temp = 0.700
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000, adaptive_target = -1.000, adaptive_decay = 0.900
0.00.262.964 I sampler chain: logits -> penalties -> ?dry -> ?top-n-sigma -> top-k -> ?typical -> top-p -> min-p -> ?xtc -> temp-ext -> dist
0.00.262.965 I generate: n_ctx = 256, n_batch = 2048, n_predict = 150, n_keep = 0
0.00.262.965 I
Il processore Pentium 4, sviluppato da Apple per iPhone e iOS, è stato presentato il 9 settembre 2009.

Il processore Prescott è stato presentato il 13 settembre 2008. È composto da due processori: un core con 64 MB di cache L3 da 8 MB ed un core HTC 3.1 MB che funziona a sessioni singole, multi-threading e multi core.

Il processore Prescott è stato presentato il 10 settembre 2008. È stato realizzato per l'aggiornamento del sistema operativo iOS 8.0 e Android 7.5.1 con una interfaccia grafica simile a quella di Android 7,

0.00.396.859 I common_perf_print:    sampling time =       6.08 ms
0.00.396.860 I common_perf_print:    samplers time =       5.23 ms /   157 tokens
0.00.396.863 I common_perf_print:        load time =      48.86 ms
0.00.396.864 I common_perf_print: prompt eval time =      41.24 ms /     7 tokens (    5.89 ms per token,   169.75 tokens per second)
0.00.396.864 I common_perf_print:        eval time =      86.23 ms /   149 runs   (    0.58 ms per token,  1727.96 tokens per second)
0.00.396.865 I common_perf_print:       total time =     134.00 ms /   156 tokens
0.00.396.865 I common_perf_print: unaccounted time =       0.46 ms /   0.3 %      (total - sampling - prompt eval - eval) / (total)
```

In teoria i K-quant funzionano già col modello attuale, llama-quantize non rifiuta il modello, solo che ffn_down_exps è l'unico tensore 1408 come dimensione interna, ricade su un altro tipo q5_0 o q8_0 invece di q4_K_*, tutti gli altri tensori usano i K-quant normalmente

| formato | dimensione | PPL |
|---|---|---|
| Q8_0 | 145 MB | 15.067 |
| Q4_K_M (con ripiego su ffn_down) | 99 MB | 15.120 (+0.37) |
| Q4_0 | 85 MB | 15.195 (+0.9%) |

Q4_K_M è la scelta migliore, perde meno della metà di Q4_0 in cambio di 14MB in più

Visto che riaddestriamo possiamo settare la costante multiple_of=256 in SwiGLU così portiamo 1408 a multiplo di 256 che per C=512 da 1536 (arrotondando per eccesso come fa LLaMA), però richiede retrain perché cambia la forma dei pesi degli esperti, i parametir crescono del 9% (da 151M a 165M) il calcolo degli esperti cresce della stessa misura, il guadagno sui k-want è piccolo, solo ffn_down passerebbe da q5_0 a q4_K e il file resterebbe intorno ai 100MB per via dei parametri in più, da solo no vale la pena però siccome dobbiamo riaddestrare il pre-tokenizer con la regex di GPT-2 sì


