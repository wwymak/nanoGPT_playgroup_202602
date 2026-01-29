import time

out_dir = 'out-shakespeare'
eval_interval = 5
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = out_dir #'shakespeare'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'shakespeare'
#init_from = 'gpt2-xl' # this is the largest GPT-2 model (Karpathy original), 6GB, via https://huggingface.co/openai-community/gpt2-xl/tree/main
#init_from - 'gpt2-large' # 3GB version https://huggingface.co/openai-community/gpt2-large/tree/main
#init_from = 'gpt2-medium' # 1.5GB version via https://huggingface.co/openai-community/gpt2-medium/tree/main
init_from = 'gpt2' # 0.5GB version via https://huggingface.co/openai-community/gpt2/tree/main

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 20

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
