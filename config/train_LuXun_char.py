out_dir = 'out-luxun-char'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'luxun-char'
wandb_run_name = 'mini-gpt-luxun'

dataset = 'LuXun_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.3

learning_rate = 1e-3
max_iters = 1000
lr_decay_iters = 1000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100