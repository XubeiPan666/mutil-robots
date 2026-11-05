# Six-task standard Pi0 baseline

Uses existing `pi0_two_arms_multi_task`, initialized from **Pi0 base**, not Pi0.5
or an AE-VLA checkpoint. No SkillLoRA, skill router or arm-wise attention.
Retains the dataset's phase-derived instructions and unified dual-arm actions.

Matched to our six-task runs: 600 episodes / 290416 frames, seed 42, global
batch 32, 30000 steps, one worker, FSDP 2, two A100 80 GB GPUs, 32 CPUs,
24-hour limit, logging every 50 steps, saving and retaining every 5000 steps,
online W&B. Optimizer, learning-rate schedule, EMA and dimensions are checked
against the AE-VLA config. No resume or overwrite is used.

Backbone-specific differences are intentional: Pi0 uses continuous state input,
its own weights and mean/std normalization. Pi0.5 uses tokenized state and
quantile normalization. Both read the existing six-task statistics file, which
contains mean/std and q01/q99. The state/action transforms used to compute that
file are equivalent, so another statistics pass is unnecessary here. This is
a comparison of model families, not an experiment isolating one architectural
module while holding the pretrained weights and normalization formula fixed.

## Prepare after syncing through GitHub

In the server training repository, run:

```bash
source "$(ws_find mutil-robots)/load_env.sh"
cd "$PROJECT_WS/code/mutil-robots"
git pull --ff-only origin main
bash -n scripts/paper6_pi0/train.sh
bash scripts/paper6_pi0/prepare.sh
```

Preparation downloads/reuses `gs://openpi-assets/checkpoints/pi0_base/params`
under `$PROJECT_WS/cache/openpi`. No GPU, W&B run or training job is started.
Keep the terminal connected while this foreground preparation runs. Its final
success message is `Six-task plain Pi0 preflight passed; no model or W&B run started.`
It uses the existing training environment and shared six-task metadata checks.
The earlier Pi0.5 base must also remain present for the reference check.
Directory checks are not a full checkpoint checksum audit or GPU load test.

## W&B and submission

Existing trainer uploads configuration, metrics and up to five first-batch camera
samples. Code upload is disabled; no checkpoint artifact upload is added. Use
your own session key, never shared `~/.netrc`, Git, logs or chat.

```bash
export WANDB_ENTITY="xubei-sun-yat-sen-university"
export WANDB_PROJECT="mutil-robots"
if [ -z "${WANDB_API_KEY:-}" ]; then
    read -rsp 'W&B API Key (hidden input): ' WANDB_API_KEY
    echo
fi
export WANDB_API_KEY

mkdir -p "$PROJECT_WS/logs"
if [ -n "${WANDB_API_KEY:-}" ]; then
    sbatch --export=ALL \
      --output="$PROJECT_WS/logs/train_pi0_paper6_%j.out" \
      --error="$PROJECT_WS/logs/train_pi0_paper6_%j.err" \
      scripts/paper6_pi0/train.sh
else
    echo 'No W&B API Key; no job submitted.'
fi
```

Submit yourself after preparation succeeds. Closing the terminal after submission
does not cancel the batch job. Resource availability and runtime are not guaranteed.
The batch repeats preflight before allocating the model and rejects extra script
arguments rather than silently ignoring them.

- Experiment: `pi0_paper6_JOBID`
- Checkpoints: `$PROJECT_WS/checkpoints/pi0_two_arms_multi_task/pi0_paper6_JOBID/`
- Logs: `$PROJECT_WS/logs/train_pi0_paper6_JOBID.{out,err}`
- Provenance: `$PROJECT_WS/runs/pi0_paper6_JOBID/`
- Evaluation must select `pi0_two_arms_multi_task` too; an AE-VLA-fixed launcher
  cannot be reused by changing only the checkpoint directory.

## Verification

Run `python -m unittest discover -s tests -p 'test_paper6*.py' -v` (Python 3.11+)
and `bash -n` on both new scripts. Real config imports and checkpoint loading
require the server environment. No training job is submitted by creating files.
