# fase 2: SFT su conversazioni (Capybara-ITA), loss solo sulle risposte dell'assistente
init_from = "out/ctx1024-chat-init.pt"  # ctx1024 + token speciali (scratch/add_chat_tokens.py)
tokenizer_path = "data/bpe-it-gpt2-chat.json"
train_bin, val_bin = "data/sft-train.bin", "data/sft-val.bin"
sft = True
block_size = 1024
batch_size = 16
max_iters = 2100  # 11,377 righe / 16 = ~710 step per epoca -> 3 epoche
learning_rate = 5e-5  # basso: si insegna un comportamento, non si riscrive cio' che il modello sa
min_lr = 5e-6
warmup_iters = 50
weight_decay = 0.0
eval_interval = 150
eval_iters = 21  # la val sono 343 righe: 21 batch da 16 la coprono circa una volta
out_dir = Path("out/sft")
