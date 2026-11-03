"""Reject incorrect six-task inputs before allocating a training model."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/paper6"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("paper6_check", SCRIPTS / "check_training.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TrainingPreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "meta").mkdir()
        (self.root / "data/chunk-000").mkdir(parents=True)
        families = [name for name in MODULE.ANCHORS if name != "handover"]
        self.episodes = [
            {"episode_index": i, "length": 2, "tasks": [MODULE.ANCHORS[families[i // 100]]]}
            for i in range(600)
        ]
        self.write("meta/info.json", {"total_episodes": 600, "total_frames": 1200, "total_tasks": 6})
        self.write("paper6_manifest.json", {
            "source_revision": MODULE.REVISION, "selected_episodes": 600, "total_frames": 1200,
            "families": {name: 100 for name in families},
            "episode_map": {str(i + 100): i for i in range(600)},
        })
        self.lines("meta/episodes.jsonl", self.episodes)
        self.lines("meta/tasks.jsonl", [{"task_index": i, "task": name} for i, name in enumerate(families)])
        self.labels = [{"task_index": i, "skill_ids": [i, i + 1]} for i in range(6)]
        self.lines("meta/skill_labels.jsonl", self.labels)
        self.stats = {group: {"mean": [0.] * 32, "std": [1.] * 32,
                              "q01": [-1.] * 32, "q99": [1.] * 32} for group in ("state", "actions")}
        self.write("norm_stats.json", {"norm_stats": self.stats})
        # File-existence checks only: the preflight deliberately does not decode Parquet.
        for i in range(600):
            (self.root / f"data/chunk-000/episode_{i:06d}.parquet").write_bytes(b"fixture")

    def write(self, relative, value):
        (self.root / relative).write_text(json.dumps(value))

    def lines(self, relative, rows):
        (self.root / relative).write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_valid_six_family_report(self):
        report = MODULE.check_dataset(self.root)
        self.assertEqual((report["episodes"], report["frames"]), (600, 1200))
        self.assertEqual(len(report["families"]), 6)

    def test_handover_contamination_rejected(self):
        self.episodes[0]["tasks"] = [MODULE.ANCHORS["handover"]]
        self.lines("meta/episodes.jsonl", self.episodes)
        with self.assertRaisesRegex(ValueError, "family"):
            MODULE.check_dataset(self.root)

    def test_wrong_labels_rejected(self):
        self.labels[-1]["skill_ids"] = [99, 1]
        self.lines("meta/skill_labels.jsonl", self.labels)
        with self.assertRaisesRegex(ValueError, "skill labels"):
            MODULE.check_dataset(self.root)

    def test_invalid_stats_rejected(self):
        for field, values in [("mean", [0.] * 16), ("std", [-1.] * 32), ("q99", [-2.] * 32)]:
            with self.subTest(field=field):
                original = self.stats["state"][field]
                self.stats["state"][field] = values
                self.write("norm_stats.json", {"norm_stats": self.stats})
                with self.assertRaises(ValueError):
                    MODULE.check_dataset(self.root)
                self.stats["state"][field] = original

    def test_missing_episode_rejected(self):
        (self.root / "data/chunk-000/episode_000599.parquet").unlink()
        with self.assertRaisesRegex(ValueError, "Missing/empty"):
            MODULE.check_dataset(self.root)


if __name__ == "__main__":
    unittest.main()
