This repo is a clone of the excellent https://github.com/karpathy/nanoGPT (MIT license) and I give huge props for the amazing code there.

# Before the playgroup day you should do some setup

Make a fresh Python env (e.g. `python -m venv .venv` `. .venv/bin/activate`), maybe using Conda or equivalent.

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

OPTIONAL but recommended - setup a https://wandb.ai/ account (Weights and Bias) and get an API key and then

```
export WANDB_API_KEY=wandb_v1_...
# this enables online tracking of the training runs
```

## Run the prepare script, this fetches local data

```
python data/shakespeare/prepare.py # prepares some input data
python train.py config/train_shakespeare_char.py 
# note if you have CUDA setup, this 'just works'
# if you're on a mac, add the mps argument
python train.py config/train_shakespeare_char.py --device=mps
# if you don't have CUDA and you don't have MPS (modern Macs) just use CPU with simplified settings
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0 
```

## You can test this

```
python sample.py --out_dir=out-shakespeare-char
```

IF YOU HAVE PROBLEMS WITH THE ABOVE, MESSAGE IN THE PLAYGROUP SLACK BEFOREHAND.

# Bonus points - get at least the small GPT2 model cached locally if you want to finetune

The GPT2 model is a large binary, it'll take an age to download at the venue, you can force an earlier download at home of the circa 0.5GB model. This is not required but we'll do it in the room, so I hope some of you have it.

```
python train.py config/finetune_shakespeare.py
```

/.cache/huggingface/hub/models--gpt2/blobs
