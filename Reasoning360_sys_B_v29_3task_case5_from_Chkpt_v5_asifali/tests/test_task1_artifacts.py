import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


def load_file(name, relative_path):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


workflow = load_file("workflow", "verl/utils/puzzle_workflow.py")
reward = load_file("pa", "verl/utils/reward_score/reward_PA_solution.py")


class Task1ArtifactsTest(unittest.TestCase):
    def test_scored_candidates(self):
        truth = {"header": ["House", "Color"], "rows": [["1", "red"], ["2", "blue"]]}
        outputs = []
        scores = []
        for first_color in ("red", "blue", "?"):
            solution = {"header": truth["header"], "rows": [["1", first_color], ["2", "?"]]}
            outputs.append(json.dumps(solution))
            scores.append(reward.compute_pa_solution({
                "n_houses": 2, "attribute_values": {"Color": ["red", "blue"]},
                "ground_truth": truth, "solution": solution, "reasoning": ["S1: red == 1."],
            }))
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {
            "PUZZLE_DATA_RUN": root, "PUZZLE_RUN_NAME": "sample-run", "CURRENT_EPOCH": "3"
        }):
            args = (["p"] * 3, outputs, [truth] * 3, [{"id": "P1"}] * 3, scores)
            workflow.write_task1_candidates({}, *args)
            self.assertEqual(list(Path(root).iterdir()), [])
            for split in ("train", "validation"):
                workflow.write_task1_candidates({"candidate_split": split, "candidate_step": 9,
                                                 "candidate_batch_id": split, "responses_per_prompt": 3}, *args)
                path = Path(root) / "candidates/task1/epoch_0003" / f"{split}_{os.getpid()}.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual([r["output"] for r in rows], outputs)
                self.assertEqual([r["metrics"] for r in rows], scores)
                self.assertEqual([r["metrics"]["pa_accuracy_binary_pass"] for r in rows], [1, 0, 0])
                for row in rows:
                    self.assertEqual((row["task_id"], row["task_name"], row["split"], row["epoch"],
                                      row["checkpoint_step"]), (1, "partial_answer_prediction", split, 3, 9))

    def test_identical_responses_have_distinct_ids_across_groups_and_batches(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {
            "PUZZLE_DATA_RUN": root, "PUZZLE_RUN_NAME": "run", "CURRENT_EPOCH": "1"
        }):
            for batch_id in ("first", "second"):
                batch = {"candidate_step": 1, "candidate_split": "train",
                         "candidate_batch_id": batch_id, "responses_per_prompt": 8}
                workflow.write_task1_candidates(batch, ["p"] * 16, ["same"] * 16,
                                                [{}] * 16, [{"id": "P1"}] * 16, [{}] * 16)
            path, = Path(root).rglob("*.jsonl")
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len({row["candidate_id"] for row in rows}), 32)
            groups = {row["rollout_group_id"] for row in rows}
            self.assertEqual(len(groups), 4)
            for group in groups:
                self.assertEqual([r["response_index"] for r in rows if r["rollout_group_id"] == group],
                                 list(range(1, 9)))

    def test_task2_parent_selection_across_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def add(epoch, candidate, accuracy, consistency, coverage, split="train", puzzle="P1"):
                folder = root / "candidates/task1" / f"epoch_{epoch:04d}"
                folder.mkdir(parents=True, exist_ok=True)
                row = dict(candidate_id=candidate, split=split, puzzle_id=puzzle, prefix="partial",
                           metrics=dict(pa_accuracy_binary_pass=accuracy,
                                        pa_consistency_binary_pass=consistency, coverage=coverage))
                with (folder / f"{split}_1.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
            add(1, "pass-low", 1, 1, 0.4)
            add(2, "pass-high", 1, 1, 0.8)
            add(2, "wrong", 0, 1, 1.0)
            add(2, "inconsistent", 1, 0, 1.0)
            add(2, "no-parent", 0, 0, 0, puzzle="P2")
            add(2, "held-out", 1, 1, 1.0, split="validation", puzzle="P3")
            for _ in range(2):
                workflow.select_task2_parents(root)
                for split, expected in (("train", "pass-high"), ("validation", "held-out")):
                    row = json.loads((root / "task2_inputs" / f"{split}.jsonl").read_text())
                    source = (root / row["source_file"]).read_text().splitlines()[row["source_line"] - 1]
                    self.assertEqual(json.loads(source)["candidate_id"], expected)
                    self.assertEqual(row["parent_candidate_id"], expected)

    def test_best_uses_heldout_binary_pass(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"CURRENT_EPOCH": "0"}):
            settings = SimpleNamespace(default_local_dir=root, save_freq=1)
            trainer = SimpleNamespace(config=SimpleNamespace(trainer=settings), _save_checkpoint=Mock())
            for step, score in enumerate((0.5, 0.4, 0.5, 0.7)):
                trainer.global_steps = step
                workflow.save_task1_best(trainer, {workflow.BEST_METRIC: score})
            self.assertEqual(trainer._save_checkpoint.call_count, 2)
            record = json.loads((Path(root) / "best.json").read_text())
            self.assertEqual((record["step"], record["metric"], record["metric_name"]),
                             (3, 0.7, workflow.BEST_METRIC))


if __name__ == "__main__":
    unittest.main()
