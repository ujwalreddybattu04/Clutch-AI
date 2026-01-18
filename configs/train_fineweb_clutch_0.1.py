# config/train_fineweb_clutch_0_1.py

out_dir = 'out-clutch-0.1'
eval_interval = 500
log_interval = 10
eval_iters = 200
wandb_log = False
wandb_project = 'clutch-0.1'
wandb_run_name = 'clutch-0.1-gpt2-124M-10B'

dataset = 'fineweb'
init_from = 'scratch'   # use 'resume' only if you already have a checkpoint in out_dir

batch_size = 4
block_size = 512
gradient_accumulation_steps = 48

max_iters = 20000
lr_decay_iters = 20000

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0

learning_rate = 6e-4
min_lr = 6e-5
warmup_iters = 2000

device = 'cuda'
compile = False
dtype = 'bfloat16'
