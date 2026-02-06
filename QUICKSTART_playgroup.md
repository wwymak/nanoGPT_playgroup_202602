This repo is a clone of the excellent https://github.com/karpathy/nanoGPT (MIT license) and I give huge props for the amazing code there.

# Before the playgroup day you should do some setup

Make a fresh Python env (e.g. `python -m venv .venv` `. .venv/bin/activate`), maybe using Conda or equivalent.

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

OPTIONAL but recommended - setup a https://wandb.ai/ account (Weights and Bias) and get an API key, choose 'Models' not Weave when signing up, then:

```
export WANDB_API_KEY=wandb_v1_...
# this enables online tracking of the training runs
```

## Run the prepare script, this fetches local data

Note if the following feels weird, read the original `README.md` file, it walks through this with more detail, I've compressed the critical bits that 'should just work'.

```
python data/shakespeare_char/prepare.py # prepares some input data
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
python data/shakespeare/prepare.py  # I think this uses the GPT BPE tokeniser, not the unigram tokeniser...
python train.py config/finetune_shakespeare.py
```

Just so you know, on my machine `pytorch` caches this in `/.cache/huggingface/hub/models--gpt2/blobs` (at 0.5GB), so you only need to do the download once.

## Lambda.ai

```
# Ian needs your ssh key to add it to the account, it is tied to a particular machine
# https://cloud.lambda.ai/instances A10/H100/A100 Lambda Stack 22.04 No Filesystem No Rules
# after ssh:
git clone https://github.com/ianozsvald/nanoGPT_playgroup_202602.git
python -m venv .venv # needed! else we inherit an out of date ubuntu python library set
. .venv/bin/activate
cd nanoGPT_playgroup_202602/
pip install torch numpy transformers datasets tiktoken wandb tqdm
[Note GH200 doesn't seem to support pytorch on python 3.10 due to Arm
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
https://discuss.pytorch.org/t/installing-pytorch-on-a-grace-hopper-gh200-node-with-gpu-support/216836
via google query, seems to solve it]
python data/shakespeare/prepare.py
python train.py config/finetune_shakespeare.py
# 10 seconds later...
python sample.py --out_dir="out-shakespeare" # maybe add the --start="" param too?
```


## Additional data

* https://raw.githubusercontent.com/tansaku/comedy_playgroup/refs/heads/main/data/expunations_annotated_full.json
* Ian's phrack download (via dropbox perhaps? 57mb) - note you need to download the zip, see `prepare.py
  * `python sample.py --out_dir=out-phrack-char --start="main( "`
* python - start with 'def lev' or 'hello'
* math_data_char - simple multiplication
  * `python sample.py --out_dir=out-math-data-char --start="3 * 5 "` (note can't add equals symbol, it borks!)
