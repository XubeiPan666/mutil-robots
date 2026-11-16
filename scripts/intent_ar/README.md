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
