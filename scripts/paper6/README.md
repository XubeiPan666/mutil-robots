# Six-source-family AE-VLA data preparation

This fork's helper excludes bottle handover from the audited 700-episode public
release. It aligns the number of source families/episodes with AE-VLA paper
section 3.2: burger/fries, can/basket, object/cabinet, push cubes, stack bowls,
stack cubes, 100 each, 600 total. It does not prove that these public trajectories
are byte-for-byte the authors' original 600 demonstrations, or guarantee paper
metrics. Source revision: `447112d1c5b2ec10f7ad34ed438532c062bc51bc`.

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
uv run --frozen python scripts/paper6/prepare_dataset.py \
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
bash -n scripts/paper6/prepare_norm.sh
sbatch --export=ALL \
  --output="$PROJECT_WS/logs/paper6_prepare_%j.out" \
  --error="$PROJECT_WS/logs/paper6_prepare_%j.err" \
  scripts/paper6/prepare_norm.sh
```

Default time limit is six hours, not a runtime guarantee. The prior seven-task
stats job took about 5h24m; conversion time and filesystem load also matter.
Only metadata checks have run against the real public release. The new conversion
and normalization batch must still be validated on the server.

Output data and six-task normalization are kept together at:

```text
$PROJECT_WS/datasets/paper6/multi_task_combined/
  data/chunk-000/episode_*.parquet
  meta/info.json, episodes.jsonl, episodes_stats.jsonl, tasks.jsonl, skill_labels.jsonl
  paper6_manifest.json
  norm_stats.json              # created by successful stats computation
```

## Subsequent training (not launched by these helpers)

Before importing training configs, set the following for this experiment only:

```bash
export FUTURE_DATA_ROOT="$PROJECT_WS/datasets/paper6"
export FUTURE_DATASET_DIR="$FUTURE_DATA_ROOT/multi_task_combined"
export FUTURE_ASSETS_DIR="$FUTURE_DATA_ROOT"
```

This directs the existing full AE-VLA config to the new dataset, remapped labels
and separate normalization assets. Use a new experiment name, initialize from
Pi0.5 base, and do not resume the seven-task checkpoint. Recheck resolved config
paths and input loading before training. Keep those paths available for later
serving, because the config resolves skill labels and asset IDs through them.

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
python -m unittest discover -s tests -p test_paper6_dataset.py -v
```

The Parquet integration test requires numpy/pyarrow and otherwise skips. It uses
tiny synthetic 700-episode data to check exclusion, image preservation, ID/label
alignment, source preservation, and rejection of existing/ambiguous outputs.

Local verification: all three new tests passed, including the PyArrow integration;
Bash syntax and repository release checks passed (Git Bash explicitly selected).
The full existing portable suite did not pass in this Windows environment, with
12 failures and one error, including shell-launch/decoding failures. It is not
claimed as a fully passing repository test run. Real conversion, LeRobot loading
and normalization still need server validation before training submission.
