# fase 1: estende il contesto da 256 a 1024 continuando il pretraining su Wikipedia.
# la RoPE non ha parametri, ma il modello ha visto solo distanze < 256: oltre, il bpb passa da ~1.0 a ~2.3
init_from = "out/ckpt-it-gpt2-moe8k2-512-58k.pt"
block_size = 1024
batch_size = 16  # 16 x 1024 = stessi token per step dei 64 x 256 del pretraining
max_iters = 4000  # ~65M token
learning_rate = 3e-4  # meta' del picco originale: il modello sa gia' la lingua, va solo adattato
min_lr = 3e-5
warmup_iters = 200
eval_interval = 500
eval_iters = 50
out_dir = Path("out/ctx1024")
