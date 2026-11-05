# Six-source-family AE-VLA data preparation

This fork's helper excludes bottle handover from the audited 700-episode public
release. It aligns the number of source families/episodes with AE-VLA paper
section 3.2: burger/fries, can/basket, object/cabinet, push cubes, stack bowls,
stack cubes, 100 each, 600 total. It does not prove that these public trajectories
are byte-for-byte the authors' original 600 demonstrations, or guarantee paper
metrics. Source revision: `447112d1c5b2ec10f7ad34ed438532c062bc51bc`.

## Script directory rename

The user may rename `scripts/paper6` to `scripts/paper6_AE-VLA` with `git mv`.
The launchers and tests support both names. From the repository root, set this
variable before copying the commands below:

```bash
AEVLA_SCRIPTS=scripts/paper6
if [ -d scripts/paper6_AE-VLA ]; then AEVLA_SCRIPTS=scripts/paper6_AE-VLA; fi
```

Only the source directory is renamed. Keep `datasets/paper6`, existing experiment
names, logs and checkpoints unchanged. The plain Pi0.5 baseline lives separately
in `scripts/paper6_pi05`; see its README for submission instructions.

## Audited split and conversion

Source bottle handover is episodes 0–99. The six families occupy 100–699.
The helper identifies families using instruction anchors, requires 100 episodes
in each of the seven original groups, excludes handover, and fails on ambiguity.
It never infers a family solely from its episode number.

The source is read only. A new dataset is materialized, with Parquet images and
actions retained, episode IDs reindexed to 0–599, global frame indices reindexed,
and retained task IDs reindexed along with tasks.jsonl / skill_labels.jsonl.
Skill adapter IDs and atom strings are not remapped. Episode statistics for
changed ID columns are updated. Old norm_stats.json is not copied.
The output manifest records source metadata hashes and old-to-new ID mappings.

This requires reading and rewriting the selected embedded-image Parquet files,
so allow additional disk space comparable to the six-task data size. It is not
a symlink view and must run within a CPU allocation. Existing output directories
are rejected, including partial outputs from failed runs; investigate first and
choose a fresh output for retries. There is no automatic overwrite or resume.

## Metadata dry run on the login node

After loading the existing training environment, from the training repo root:

```bash
uv run --frozen python "$AEVLA_SCRIPTS"/prepare_dataset.py \
  --source dataset/multi_task_combined \
  --output "$PROJECT_WS/datasets/paper6/multi_task_combined" \
  --revision "$(cat "$PROJECT_WS/macg_data_revision.txt")"
```

Without --apply this reads only metadata, prints the selection and writes nothing.

## CPU-only conversion and fresh normalization

The NHR-specific shell wrapper uses the existing load_env.sh and workspace.
It requests a100mig CPUs with no GPU GRES, then computes stats on the six-task
data only. It does not train or start W&B. Example:

```bash
mkdir -p "$PROJECT_WS/logs"
bash -n "$AEVLA_SCRIPTS"/prepare_norm.sh
sbatch --export=ALL \
  --output="$PROJECT_WS/logs/paper6_prepare_%j.out" \
  --error="$PROJECT_WS/logs/paper6_prepare_%j.err" \
  "$AEVLA_SCRIPTS"/prepare_norm.sh
```

Default time limit is six hours, not a runtime guarantee. The prior seven-task
stats job took about 5h24m; conversion time and filesystem load also matter.
The user confirmed server job 4903606 completed successfully (0:0), taking
04:16:05, with 600 episodes and 290416 frames. Recheck the files before training.

Output data and six-task normalization are kept together at:

```text
$PROJECT_WS/datasets/paper6/multi_task_combined/
  data/chunk-000/episode_*.parquet
  meta/info.json, episodes.jsonl, episodes_stats.jsonl, tasks.jsonl, skill_labels.jsonl
  paper6_manifest.json
  norm_stats.json              # created by successful stats computation
```

## New six-task training with Slurm and W&B

The NHR wrapper `train.sh` requests two A100 **80 GB** GPUs, 32 CPU cores and
24 hours. It runs the existing full AE-VLA config for 30000 steps, global batch
32, FSDP 2, logging every 50 steps, saving/retaining every 5000 steps. The final
checkpoint is step 29999. It initializes from cached Pi0.5 base, with a new
experiment `aevla_paper6_JOBID`; it never resumes the seven-task checkpoint.
Do not run the wrapper with bash on a login node. No job has been submitted by
adding these files. Synchronize this local fork through your GitHub repository
before executing the following commands on the server.

Before importing training configs, set the following for this experiment only:

```bash
export FUTURE_DATA_ROOT="$PROJECT_WS/datasets/paper6"
export FUTURE_DATASET_DIR="$FUTURE_DATA_ROOT/multi_task_combined"
export FUTURE_ASSETS_DIR="$FUTURE_DATA_ROOT"
export FUTURE_PI05_PARAMS="$PROJECT_WS/cache/openpi/openpi-assets/checkpoints/pi05_base/params"

JAX_PLATFORMS=cpu uv run --frozen python "$AEVLA_SCRIPTS"/check_training.py \
  --dataset "$FUTURE_DATASET_DIR" --check-config
```

Run that from the training repo after `source "$(ws_find mutil-robots)/load_env.sh"`
and `export PYTHONNOUSERSITE=1`. The read-only preflight checks six families,
600 contiguous episode IDs, file presence, metadata totals, task/skill-label IDs,
finite 32-dimensional state/action stats and the resolved config's data, label,
normalization and Pi0.5 paths. It does not decode the full dataset or prove the
statistics' provenance from their numerical values. The successful six-task
statistics job is the provenance evidence. It starts no model or W&B run.
Keep the same dataset overrides for later serving of this experiment.

### Choose and authenticate your W&B destination

The existing trainer uploads training configuration, scalar metrics and up to
five first-batch camera samples. This wrapper disables code upload; it adds no
checkpoint artifact upload. Choose your own entity (username/team slug) and
project before submission. On this shared server, use a session API key rather
than changing shared `~/.netrc`; do not put the key in Git, command arguments,
scripts, logs or chat. Obtain your key privately at https://wandb.ai/settings.

```bash
read -rp 'W&B entity (username or team): ' WANDB_ENTITY
read -rp 'W&B project [mutil-robots]: ' WANDB_PROJECT
export WANDB_ENTITY
export WANDB_PROJECT="${WANDB_PROJECT:-mutil-robots}"
read -rsp 'W&B API key (hidden input): ' WANDB_API_KEY
echo
export WANDB_API_KEY
```

No `wandb login` or W&B run is performed by the preflight. Submit from this same
shell: Slurm inherits these variables. The wrapper requires all three values,
sets online mode, and explicitly enables W&B in the training CLI. A missing or
invalid key/project permission must be resolved before a successful online run.

### Submit and monitor

```bash
mkdir -p "$PROJECT_WS/logs"
bash -n "$AEVLA_SCRIPTS"/train.sh
sbatch --export=ALL \
  --output="$PROJECT_WS/logs/train_paper6_%j.out" \
  --error="$PROJECT_WS/logs/train_paper6_%j.err" \
  "$AEVLA_SCRIPTS"/train.sh
```

Record the returned job ID. Once allocated, validation and training start
automatically, independently of your SSH terminal. Inspect `squeue -j JOBID`,
`sacct -j JOBID --format=JobID,State,ExitCode,Elapsed` and
`$PROJECT_WS/logs/train_paper6_JOBID.{out,err}`. W&B prints the run link in the
training logs. The batch records Git commit/status, the selection manifest,
the normalization file SHA256 and the preflight report under
`$PROJECT_WS/runs/aevla_paper6_JOBID/` without printing credentials.
Checkpoints go to
`$PROJECT_WS/checkpoints/aevla_pi05_skill_lora_two_arms_multitask/aevla_paper6_JOBID/`.
The 24-hour allocation is a limit, not a runtime guarantee; use observed
throughput after compilation to estimate whether it will finish in time.

Paper appendix B.1 specifies 30,000 steps, global batch 32, two GPUs in the paper
experimental setup, no arm shuffle/image masking/wait augmentation, and a 30,000
checkpoint interval. The user previously preferred 5,000-step saves; explicitly
choose and record this operational difference when preparing the new job.
Using the available A100 hardware instead of paper A800 is also an environment
difference. Confirm W&B entity/project and its sample-image upload scope before
enabling it, as requested for the next training. Never put API keys in this repo.

For paper metrics, separately select the paper's 23 evaluation conditions;
the public evaluator's default all suite has 26. Dataset task-count alignment
alone is not a claim of complete paper reproduction.

## Local checks

```bash
python -m unittest discover -s tests -p 'test_paper6*.py' -v
```

The Parquet integration test requires numpy/pyarrow and otherwise skips. It uses
tiny synthetic 700-episode data to check exclusion, image preservation, ID/label
alignment, source preservation, and rejection of existing/ambiguous outputs.

Local verification: five training-preflight tests passed, covering family
contamination, labels, invalid stats and missing episodes. The earlier dataset
preparation change passed three tests, including the PyArrow integration.
Bash syntax and repository release checks passed (Git Bash explicitly selected).
The full existing portable suite did not pass in this Windows environment, with
12 failures and one error, including shell-launch/decoding failures. It is not
claimed as a fully passing repository test run. Real conversion, LeRobot loading
and normalization have since completed on the server according to the user's
job output. The new training wrapper and config preflight still need execution
in that environment; no GPU training or W&B authentication was run locally.
