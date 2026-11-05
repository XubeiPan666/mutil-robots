"""Build an independent six-source-family dataset, never modifying the source.

Default is a metadata-only dry run. --apply rewrites Parquet index columns and
corresponding metadata into a new directory; old normalization is never copied.
"""

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path


ANCHORS = {
    "handover": "Handover the black bottle.",
    "burger_fries": "Pick up the burger.",
    "can_basket": "Pick up the can.",
    "object_cabinet": "Grab the drawer handle.",
    "push_cubes": "Push the blue cube to the target.",
    "stack_bowls": "Pick up the bowl.",
    "stack_cubes": "Pick up the blue cube.",
}


def load_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_lines(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def plan(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output directories must not overlap")
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a fresh directory: {output}")
    info = json.loads((source / "meta/info.json").read_text())
    if info["total_episodes"] != 700 or info.get("total_chunks", 1) != 1:
        raise ValueError("Expected the audited single-chunk 700-episode public release")
    if any(f.get("dtype") == "video" for f in info["features"].values()):
        raise ValueError("Only embedded-image Parquet data is supported")
    episodes = load_lines(source / "meta/episodes.jsonl")
    if sorted(e["episode_index"] for e in episodes) != list(range(700)):
        raise ValueError("Source episode indices must be unique and cover 0..699")
    selected, families = [], Counter()
    for episode in sorted(episodes, key=lambda e: e["episode_index"]):
        text = "\n".join(episode["tasks"])
        matches = [name for name, anchor in ANCHORS.items() if anchor in text]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous/unknown family: {episode['episode_index']}: {matches}")
        family = matches[0]
        families[family] += 1
        if family != "handover":
            selected.append((family, episode))
    if families != Counter({name: 100 for name in ANCHORS}):
        raise ValueError(f"Unexpected family counts: {families}")
    return source, output, info, selected


def build(source, output, *, revision, apply=False):
    source, output, info, selected = plan(source, output)
    report = {
        "source": str(source), "source_revision": revision,
        "output": str(output), "selected_episodes": len(selected),
        "families": dict(Counter(family for family, _ in selected)),
        "episode_map": {str(e["episode_index"]): i for i, (_, e) in enumerate(selected)},
        "source_metadata_sha256": {
            name: hashlib.sha256((source / "meta" / name).read_bytes()).hexdigest()
            for name in ("info.json", "episodes.jsonl", "tasks.jsonl", "skill_labels.jsonl")
        },
    }
    if not apply:
        print(json.dumps({key: value for key, value in report.items() if key != "episode_map"}, indent=2))
        print("DRY RUN: no output was created. Use --apply in a CPU allocation.")
        return report

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    tasks = load_lines(source / "meta/tasks.jsonl")
    labels = load_lines(source / "meta/skill_labels.jsonl")
    task_lookup = {row["task_index"]: row for row in tasks}
    label_lookup = {row["task_index"]: row for row in labels}
    if len(task_lookup) != len(tasks) or len(label_lookup) != len(labels):
        raise ValueError("Duplicate task or skill-label IDs")
    stats_lookup = {row["episode_index"]: row for row in load_lines(source / "meta/episodes_stats.jsonl")}
    used = set()
    for _, episode in selected:
        path = source / "data/chunk-000" / f"episode_{episode['episode_index']:06d}.parquet"
        used.update(pq.read_table(path, columns=["task_index"])["task_index"].to_pylist())
    mapping = {old: new for new, old in enumerate(sorted(used))}
    for old in mapping:
        if old not in task_lookup or old not in label_lookup:
            raise ValueError(f"Task {old} has no instruction or skill labels")
        ids = label_lookup[old]["skill_ids"]
        if len(ids) != 2 or any(not isinstance(i, int) or i < 0 or i >= 10 for i in ids):
            raise ValueError(f"Invalid two-arm skill labels for task {old}: {ids}")

    (output / "meta").mkdir(parents=True)
    (output / "data/chunk-000").mkdir(parents=True)
    new_episodes, new_stats, offset = [], [], 0
    for new_id, (family, episode) in enumerate(selected):
        old_id = episode["episode_index"]
        table = pq.read_table(source / "data/chunk-000" / f"episode_{old_id:06d}.parquet")
        count = len(table)
        if count != episode["length"] or count == 0:
            raise ValueError(f"Bad episode length: {old_id}")
        if set(table["episode_index"].to_pylist()) != {old_id}:
            raise ValueError(f"Wrong episode IDs in Parquet {old_id}")
        if table["frame_index"].to_pylist() != list(range(count)):
            raise ValueError(f"Unexpected frame indices in episode {old_id}")
        new_tasks = np.array([mapping[i] for i in table["task_index"].to_pylist()], dtype=np.int64)
        replacements = {
            "episode_index": np.full(count, new_id, dtype=np.int64),
            "index": np.arange(offset, offset + count, dtype=np.int64),
            "task_index": new_tasks,
        }
        stats = copy.deepcopy(stats_lookup[old_id])
        stats["episode_index"] = new_id
        for key, values in replacements.items():
            field = table.schema.field(key)
            table = table.set_column(table.schema.get_field_index(key), field, pa.array(values, type=field.type))
            if key in stats.get("stats", {}):
                stats["stats"][key].update(
                    min=[int(values.min())], max=[int(values.max())],
                    mean=[float(values.mean())], std=[float(values.std())],
                )
        pq.write_table(table, output / "data/chunk-000" / f"episode_{new_id:06d}.parquet")
        new_episodes.append(dict(episode, episode_index=new_id))
        new_stats.append(stats)
        offset += count
        if (new_id + 1) % 25 == 0:
            print(f"Rewrote {new_id + 1}/600 episodes", flush=True)

    new_info = dict(info, total_episodes=600, total_frames=offset, total_tasks=len(mapping),
                    total_chunks=1, splits={"train": "0:600"})
    (output / "meta/info.json").write_text(json.dumps(new_info, indent=2) + "\n")
    write_lines(output / "meta/episodes.jsonl", new_episodes)
    write_lines(output / "meta/episodes_stats.jsonl", new_stats)
    write_lines(output / "meta/tasks.jsonl", [dict(task_lookup[old], task_index=new) for old, new in mapping.items()])
    write_lines(output / "meta/skill_labels.jsonl", [dict(label_lookup[old], task_index=new) for old, new in mapping.items()])
    report["task_map"] = mapping
    report["total_frames"] = offset
    report["normalization"] = "Not copied; compute using this dataset only"
    (output / "paper6_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Dataset complete: {output}; 600 episodes; {offset} frames; normalization still required", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    build(args.source, args.output, revision=args.revision, apply=args.apply)
