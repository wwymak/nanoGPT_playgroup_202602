# How train.py Works

## Setup

Configuration variables are defined as module-level globals (model size, learning rate, batch size, etc.) and then overridden via `configurator.py` from the command line or a config file.

If launched via `torchrun`, DDP (Distributed Data Parallel) is detected through the `RANK` environment variable. The distributed backend is initialized with `torch.distributed.init_process_group`, and gradient accumulation steps are divided across workers. The model is later wrapped in `torch.nn.parallel.DistributedDataParallel` (DDP) so gradients are synchronized across GPUs.

Mixed-precision training is handled by `torch.amp.autocast`, which wraps the forward pass in a context manager to run in bfloat16 or float16. When using float16, a `torch.cuda.amp.GradScaler` scales the loss to prevent underflow during the backward pass.

If `compile=True`, the model is passed through `torch.compile()` for graph-level optimization.

## Data Loading

There is no `torch.utils.data.DataLoader`. Instead, a `get_batch` function memory-maps the pre-tokenized `train.bin`/`val.bin` files with `np.memmap` and draws random slices of length `block_size`. Input `x` is the token sequence; target `y` is the same sequence shifted by one. On CUDA, tensors are pinned to host memory (`.pin_memory()`) and transferred asynchronously with `non_blocking=True`.

## Model Initialization

Three modes controlled by `init_from`:

- `'scratch'`: constructs a new `GPT` from `GPTConfig` with the configured hyperparameters.
- `'resume'`: loads a checkpoint via `torch.load`, restores model weights with `model.load_state_dict`, and restores the optimizer state.
- `'gpt2*'`: loads pretrained OpenAI GPT-2 weights via `GPT.from_pretrained`.

The optimizer is an `AdamW` instance (created inside `model.configure_optimizers`) with separate parameter groups for weight-decayed and non-decayed parameters.

## Training Loop

Each iteration:

1. **Learning rate**: a cosine-with-warmup schedule is computed manually and written into `optimizer.param_groups`.
2. **Evaluation**: every `eval_interval` steps, `estimate_loss` runs the model in `model.eval()` mode over `eval_iters` batches (under `torch.no_grad()`) and averages the loss. Checkpoints are saved with `torch.save` when validation loss improves.
3. **Forward/backward**: the inner loop runs `gradient_accumulation_steps` micro-steps. Each micro-step computes the forward pass under `torch.amp.autocast`, scales the loss, and calls `scaler.scale(loss).backward()`. In DDP mode, gradient sync is suppressed on all but the last micro-step by toggling `model.require_backward_grad_sync`.
4. **Optimizer step**: gradients are unscaled, clipped with `torch.nn.utils.clip_grad_norm_`, and then the optimizer steps via `scaler.step(optimizer)`. Gradients are zeroed with `optimizer.zero_grad(set_to_none=True)` to free memory immediately.

## Key PyTorch Classes

| Class | Role |
|---|---|
| `torch.nn.parallel.DistributedDataParallel` | Multi-GPU gradient synchronization |
| `torch.amp.autocast` | Mixed-precision forward pass (bf16/fp16) |
| `torch.cuda.amp.GradScaler` | Loss scaling for fp16 training |
| `torch.optim.AdamW` | Optimizer (via `model.configure_optimizers`) |
| `torch.compile` | Graph-level JIT compilation (PyTorch 2.0) |
| `torch.nn.utils.clip_grad_norm_` | Gradient clipping |
