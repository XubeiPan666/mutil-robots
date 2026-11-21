# V1: shared single-arm normalization

This step prepares statistics only; it does not implement/train the intent model.
The source is the existing six-task dataset (600 episodes). The default command
is a read-only metadata/config preflight. `--apply` runs the full scan.

## Semantics

- Reuse `aevla_pi05_skill_lora_two_arms_multitask`'s existing LeRobot reader,
  horizon 50, end-of-episode padding, repack and data transforms. Do not invoke
  tokenizer/model transforms, load model weights or start W&B.
- Disable augmentation. Seven joints per arm use `future_absolute - current_state`;
  the gripper remains absolute. Every position in a chunk subtracts the same
  current observation, not the preceding future action.
- After that transform, convert `(32,)` state / `(50,32)` actions into paired
  `(2,8)` / `(2,50,8)` numeric arrays. Pool both arms into the same eight slots.
  The reusable `openpi.shared.single_arm.split_delta_pair` does **not** apply a
  second delta transform. Future training can reuse it after the same transform.
- Count every frame exactly once, including the final partial batch. State
  statistics count `2 * frames` vectors; action statistics count
  `2 * frames * 50` vectors, including LeRobot's padded future actions. This is
  the action-chunk distribution, not a once-per-physical-action distribution.
- Use OpenPI `RunningStats` (approximate histogram quantiles), with float64
  accumulation. Save 8 meaningful dimensions followed by 24 exact-zero padding
  dimensions, in OpenPI's 32D `norm_stats.json` format.
- Input metadata and previous statistics are read-only. Output must be a new
  directory outside the dataset. Never overwrite an old run. A failed attempt
  is retained with its manifest; use a new output directory for another attempt.
- The full reader still decodes images before dropping them. This intentionally
  reuses the trusted training transforms; allow several hours of CPU time.
  Image/schema errors fail the job instead of silently replacing frames.

Unlike the older generic stats script, this scan does not drop the last partial
batch. Tiny numeric differences from old statistics are therefore expected;
the primary change is pooling the arms. The source dataset remains unchanged.

## Server preflight (no full scan)

```bash
source "$(ws_find mutil-robots)/load_env.sh"
export PYTHONNOUSERSITE=1
export JAX_PLATFORMS=cpu
cd "$PROJECT_WS/code/mutil-robots"

uv run --frozen python scripts/intent_ar/compute_norm_stats.py \
  --dataset "$PROJECT_WS/datasets/paper6/multi_task_combined" \
  --output "$PROJECT_WS/assets/intent_ar_single_arm/preflight_only"
```

The preflight reuses the paper-six revision/metadata/file/old-stats audit and
checks the reference data transform. It does not decode images or guarantee
the later full scan will succeed. No output directory is created.

## Submit a CPU batch job

```bash
mkdir -p "$PROJECT_WS/logs"
bash -n scripts/intent_ar/prepare_norm.sh
sbatch --export=ALL \
  --output="$PROJECT_WS/logs/singlearm_norm_%j.out" \
  --error="$PROJECT_WS/logs/singlearm_norm_%j.err" \
  scripts/intent_ar/prepare_norm.sh
```

The existing cluster convention uses `a100mig` for CPU-only work. No GPU is
requested. The six-hour allocation begins when scheduled; queued time is extra.
Each submission writes its own directory:
`$PROJECT_WS/assets/intent_ar_single_arm/paper6_<JOB_ID>/`.
Disconnecting the terminal after successful submission does not stop the batch job.

```bash
# Replace 1234567 with the submitted job ID.
sacct -X -j 1234567 --format=JobID,State,ExitCode,Elapsed
tail -n 10 "$PROJECT_WS/logs/singlearm_norm_1234567.err"
tail -n 10 "$PROJECT_WS/logs/singlearm_norm_1234567.out"
cat "$PROJECT_WS/assets/intent_ar_single_arm/paper6_1234567/stats_manifest.json"
```

Completion requires Slurm `COMPLETED / 0:0`, manifest `status: complete`, expected
frame/vector counts and `norm_stats.json`. For 290,416 frames, expected counts
are 580,832 state vectors and 29,041,600 action vectors. The manifest records
metadata/source hashes, implementation hashes, Git commit and stats hash.
It does not hash every Parquet data file; use the existing dataset revision and
paper-six manifest as provenance, and do not modify source files during the job.

## Integration boundary

R1/R2/R3 may share this file only with the same slot mapping, delta semantics,
horizon, training dataset and padding convention. It is not automatically wired
into any existing model config. New training must point to this separate asset
directory; existing AE-VLA / Pi0 / Pi0.5 runs keep their original assets.

Local tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_single_arm_norm.py'
```

## V1 model implementation (local development; not a training result)

The architecture and its config live together in
`src/openpi/models/pi05_intent_ar.py`. It subclasses Pi0.5 to preserve all base
weight paths/shapes; only BOS and a message projection are new.
`Pi05IntentARWeightLoader` requires all base weights and only permits these new
message parameters to retain initialization. The implementation is JAX-only.

Registered ablations:

| Config | Cross-arm messages | Random training order |
| --- | --- | --- |
| `pi05_intent_ar_r1_no_comm_fixed_order` | No; base-only context, no coordination slots | No |
| `pi05_intent_ar_r2_comm_fixed_order` | Yes | No; A then B |
| `pi05_intent_ar_r3_comm_random_order` | Yes | Yes; independent A/B permutation per scene |

R2/R3 use one group of four continuous coordination slots: BOS for the first
arm, then the preceding arm's output. There are no additional query tokens.
One bidirectional LLM prefix pass over `[base, incoming]` produces both the
updated message (projected final hidden states of the incoming slots) and KV
for the action expert. Base and incoming slots can attend to one another.
The final message is sent to the next arm, never reinserted into its own prefix.
Training and inference both reuse this differentiable KV; inference reuses it
across all flow steps. R1 encodes only base and bypasses message generation.
No skill labels/loss/router, explicit arm-ID embeddings or action-token
autoregression is added. Messages/KV never receive noisy action targets, and
gradients are retained through KV and across the chain. The outgoing projection
is supervised through downstream action loss; the last arm's message is unused.
The last projection may be eliminated by JAX when its output is unused.
There is no mutable cross-request message memory. Both action chunks are
returned together; the existing simulator/controller decides how many steps
to execute synchronously. The model itself never steps the simulator.

This replaces the earlier two-pass query design. Its trained checkpoints are
not interchangeable with this architecture; start a new experiment from Pi0.5
base. Dataset transforms and single-arm statistics are unchanged and reusable.

### Data and inference contract

1. Existing Droid transform assembles the two arms and applies joint deltas.
2. `PairInputs` creates state `(2,32)` and actions `(H,2,32)` with eight meaningful
   slots per arm. The shared 32D statistics broadcast over the arm axis.
3. `TokenizePair` tokenizes each arm's private instruction with its own normalized
   state. It requires exactly `instruction A | instruction B` and zeroes padding.
4. For compatibility with the existing Observation and training APIs, only the
   transport representation is flattened: state `(64,)`, actions `(H,64)`,
   token IDs `(2 * max_token_len,)`. This is **not** a 64D action expert: each call
   receives one 32D arm and a canonical global/own-wrist image pair.
5. The model augments the global image once for the scene, selects all private
   inputs together, generates the message then the action chunk, and restores
   outputs to physical A/B order. Action noise is indexed by physical arm.
   The global image is visually encoded once per training/inference call, after
   augmentation. Both arms reuse those differentiable tokens, while each keeps
   its own wrist encoding and LLM KV. Tokens are never cached across calls.
6. Inference output transforms unflatten, unnormalize with shared stats, join the
   valid slots, add the current state back to joint deltas, and return the
   original 16D two-arm action interface (absolute grippers unchanged).

Inference order is a static model config, default `(0,1)`. To evaluate `(1,0)`,
replace `train_config.model.inference_order` before loading a policy. No checkpoint
weight changes are required. It is not a dynamic `sample_kwargs` parameter.
Do not alter a running shared policy's order; construct a separate policy/run.

Training loss averages only eight meaningful action dimensions and both arms.
Noisy/predicted padding is kept zero. All ablations use the same convention.
Current V1 intentionally supports exactly two arms. Three-arm generalization
and the related clients/transforms are future work, not an implemented feature.

### After statistics complete

Set `FUTURE_INTENT_STATS_DIR` to the exact completed job output. A dedicated guard
rejects missing/incomplete manifests, metadata/hash drift, a different dataset,
incorrect counts/horizon or old joint-arm statistics. It does not hash Parquet
contents. This variable takes precedence over `FUTURE_ASSETS_DIR` for these three
new configs only; existing configurations retain their original behavior.

```bash
source "$(ws_find mutil-robots)/load_env.sh"
export PYTHONNOUSERSITE=1
cd "$PROJECT_WS/code/mutil-robots"
export FUTURE_INTENT_STATS_DIR="$PROJECT_WS/assets/intent_ar_single_arm/paper6_5015188"
JAX_PLATFORMS=cpu uv run --frozen python scripts/intent_ar/check_model.py \
  --config pi05_intent_ar_r3_comm_random_order \
  --dataset "$PROJECT_WS/datasets/paper6/multi_task_combined" \
  --stats-dir "$FUTURE_INTENT_STATS_DIR"
```

This does not allocate a model or start training. Before a full training job,
validate real data loading, base-checkpoint restore, a GPU forward/backward and
checkpoint/policy roundtrip. Serial full-size forwards may exceed the old
baseline's memory/time budget; two A100s and batch 32 have not been validated for
this architecture. Do not reuse the old batch budget as proof that it fits.

CPU tests use a tiny image encoder, reduced Gemma depth/vocabulary and real
JAX/Flax message/action computation; they do not assess full-size GPU memory or
task success. Tests open randomly initialized adaRMS residual gates only inside
the tiny fixture (fresh Pi0.5 initializes them to zero; real runs load base weights).
Production initialization is unchanged.

### Slurm training launcher

`scripts/intent_ar/train.sh` accepts `r1`, `r2`, `r3`, or the corresponding full
configuration name above. The configuration must be selected explicitly.
It requests two A100 80GB GPUs, 32 CPUs and 24 hours, then uses the shared
`scripts/train.py` entrypoint. The two GPUs shard the shared model; they are not
assigned one per arm. Slurm options may be overridden before the script path.

Full training uses seed 42, batch 32, 30000 steps, one data worker, FSDP over two
GPUs, and save/keep intervals of 5000. `--smoke` uses batch 1 and 20 steps with
per-step logging; the final checkpoint is step 19. A successful smoke test does
not establish that full batch 32 fits in GPU memory.

Both modes require session `WANDB_API_KEY`, `WANDB_ENTITY` and `WANDB_PROJECT`.
Do not put credentials in this script or commit them. W&B uploads training
configuration, metrics and up to five first-batch camera samples. Code upload
is disabled. Select/authenticate your own W&B destination before submission.

After syncing the code to the server, load the environment and select the
completed statistics (adjust the path if normalization was recomputed):

```bash
source "$(ws_find mutil-robots)/load_env.sh"
cd "$PROJECT_WS/code/mutil-robots"
export FUTURE_INTENT_STATS_DIR="$PROJECT_WS/assets/intent_ar_single_arm/paper6_5015188"
mkdir -p "$PROJECT_WS/logs"
bash -n scripts/intent_ar/train.sh
bash scripts/intent_ar/train.sh --help
```

Submission examples below actually enqueue jobs; run them only when ready,
after setting the W&B variables. Replace `r3` with `r1` or `r2` as needed.

```bash
# Short validation of real data, base weights, forward/backward and checkpoint saving.
sbatch --export=ALL --time=01:00:00 \
  --output="$PROJECT_WS/logs/train_intent_ar_%j.out" \
  --error="$PROJECT_WS/logs/train_intent_ar_%j.err" \
  scripts/intent_ar/train.sh r3 --smoke

# Full training (GPU memory at batch 32 still needs validation).
sbatch --export=ALL \
  --output="$PROJECT_WS/logs/train_intent_ar_%j.out" \
  --error="$PROJECT_WS/logs/train_intent_ar_%j.err" \
  scripts/intent_ar/train.sh r3
```

The launcher fixes the dataset to `$PROJECT_WS/datasets/paper6/multi_task_combined`
and starts from the cached Pi0.5 base weights. It validates the selected shared
statistics and requires two visible JAX GPUs before starting training. It does
not resume or overwrite earlier checkpoints. Experiments are named
`<full-config-name>_paper6_<full-or-smoke>_<job-id>`; checkpoints live under
`$PROJECT_WS/checkpoints/<full-config-name>/<experiment>/`.
`$PROJECT_WS/runs/<experiment>/` records the Git revision/status, submitted
launcher, dataset/statistics manifests, statistics copy/hash, preflight output
and training command. The launcher does not perform a restored-policy inference
test; that still needs verification after the smoke checkpoint is written.

```bash
JAX_PLATFORMS=cpu uv run --frozen python -m pytest -q \
  tests/test_pi05_intent_ar.py tests/test_intent_ar_policy.py \
  tests/test_intent_ar_assets.py tests/test_single_arm_norm.py
```
