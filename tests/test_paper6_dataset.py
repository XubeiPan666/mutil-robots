"""Check split boundaries, label alignment and source preservation on tiny data."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "paper6_prepare", Path(__file__).resolve().parents[1] / "scripts/paper6/prepare_dataset.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = pq = None


class PaperSixTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / "source"
        self.output = Path(self.tmp.name) / "six"
        meta = self.source / "meta"
        meta.mkdir(parents=True)
        self.episodes = [
            {"episode_index": i, "length": 2,
             "tasks": [list(MODULE.ANCHORS.values())[i // 100]]}
            for i in range(700)
        ]
        (meta / "info.json").write_text(json.dumps({
            "total_episodes": 700, "total_frames": 1400, "total_tasks": 14,
            "total_chunks": 1, "splits": {"train": "0:700"},
            "features": {"image": {"dtype": "image"}},
        }))
        MODULE.write_lines(meta / "episodes.jsonl", self.episodes)
        MODULE.write_lines(meta / "tasks.jsonl", [{"task_index": i, "task": f"task_{i}"} for i in range(14)])
        MODULE.write_lines(meta / "skill_labels.jsonl", [
            {"task_index": i, "skill_ids": [i % 10, (i + 1) % 10], "atoms": ["a", "b"]}
            for i in range(14)
        ])

    def test_metadata_plan_and_guards(self):
        report = MODULE.build(self.source, self.output, revision="fixture")
        self.assertFalse(self.output.exists())
        self.assertEqual(report["selected_episodes"], 600)
        self.assertEqual(report["episode_map"]["100"], 0)
        self.assertEqual(report["episode_map"]["699"], 599)
        self.assertNotIn("99", report["episode_map"])
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            MODULE.plan(self.source, self.output)
        with self.assertRaises(ValueError):
            MODULE.plan(self.source, self.source / "nested")

    def test_changed_family_fails_instead_of_guessing(self):
        self.episodes[100]["tasks"] = ["Unknown task"]
        MODULE.write_lines(self.source / "meta/episodes.jsonl", self.episodes)
        with self.assertRaises(ValueError):
            MODULE.plan(self.source, self.output)
        self.assertFalse(self.output.exists())

    @unittest.skipIf(pa is None, "PyArrow required for Parquet integration")
    def test_conversion_preserves_images_and_aligns_ids_and_labels(self):
        data = self.source / "data/chunk-000"
        data.mkdir(parents=True)
        stats = []
        for old in range(700):
            task = (old // 100) * 2
            table = pa.table({
                "episode_index": [old, old], "frame_index": [0, 1],
                "index": [old * 2, old * 2 + 1], "task_index": [task, task + 1],
                "image": [{"bytes": str(old).encode(), "path": None}] * 2,
                "actions": [[0.1, 0.2], [0.3, 0.4]],
            })
            pq.write_table(table, data / f"episode_{old:06d}.parquet")
            stats.append({"episode_index": old, "stats": {
                k: {"min": [-1], "max": [-1], "mean": [-1], "std": [-1], "count": [2]}
                for k in ("episode_index", "index", "task_index")
            }})
        MODULE.write_lines(self.source / "meta/episodes_stats.jsonl", stats)
        original = (data / "episode_000100.parquet").read_bytes()
        (self.source / "norm_stats.json").write_text('"seven-task stats"')
        MODULE.build(self.source, self.output, revision="fixture", apply=True)
        first = pq.read_table(self.output / "data/chunk-000/episode_000000.parquet").to_pydict()
        last = pq.read_table(self.output / "data/chunk-000/episode_000599.parquet").to_pydict()
        self.assertEqual(first["image"][0]["bytes"], b"100")
        self.assertEqual(first["episode_index"], [0, 0])
        self.assertEqual(first["task_index"], [0, 1])
        self.assertEqual(last["index"], [1198, 1199])
        self.assertEqual(last["episode_index"], [599, 599])
        labels = MODULE.load_lines(self.output / "meta/skill_labels.jsonl")
        self.assertEqual(labels[0]["skill_ids"], [2, 3])
        info = json.loads((self.output / "meta/info.json").read_text())
        self.assertEqual((info["total_episodes"], info["total_frames"], info["total_tasks"]), (600, 1200, 12))
        self.assertEqual(len(MODULE.load_lines(self.output / "meta/episodes_stats.jsonl")), 600)
        self.assertFalse((self.output / "norm_stats.json").exists())
        self.assertEqual((data / "episode_000100.parquet").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
