"""Task 1 rollout records and validation-based checkpoint selection."""
import json
import math
import os
from pathlib import Path


TASK_ID = 1
TASK_NAME = "partial_answer_prediction"
BEST_METRIC = "val-aux/our_zebra_puzzle_new_reward_test/pa_accuracy_binary_pass/mean@1"


def write_task1_candidates(batch_info, prompts, outputs, ground_truths, extra_infos, metrics):
    root = os.environ.get("PUZZLE_DATA_RUN")
    split = batch_info.get("candidate_split")
    if not root or split not in ("train", "validation"):
        return

    epoch = int(os.environ["CURRENT_EPOCH"])
    directory = Path(root) / "candidates" / "task1" / f"epoch_{epoch:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    lines = []
    candidates = zip(
        prompts, outputs, ground_truths, extra_infos, metrics, strict=True
    )
    # Both rollout paths interleave the responses for each prompt before scoring.
    for index, (prompt, output, truth, info, scores) in enumerate(candidates):
        group_index, response_index = divmod(index, batch_info["responses_per_prompt"])
        group_id = f'{batch_info["candidate_batch_id"]}-{group_index + 1}'
        info = info or {}
        row = {
            "task_id": TASK_ID, "task_name": TASK_NAME,
            "rollout_group_id": group_id,
            "response_index": response_index + 1,
            "candidate_id": f"t1-{group_id}-{response_index + 1}",
            "puzzle_id": str(info.get("id", info.get("puzzle_id", ""))),
            "source_run": os.environ["PUZZLE_RUN_NAME"],
            "epoch": epoch, "checkpoint_step": batch_info["candidate_step"], "split": split,
            "question_prompt": info.get("question_prompt", prompt),
            "ground_truth": truth, "output": output, "prefix": output, "metrics": scores,
        }
        lines.append(json.dumps(row, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    with (directory / f"{split}_{os.getpid()}.jsonl").open("a", encoding="utf-8") as handle:
        handle.writelines(line + "\n" for line in lines)


def save_task1_best(trainer, validation_metrics):
    if trainer.config.trainer.save_freq <= 0:
        return
    score = float(validation_metrics[BEST_METRIC])
    if not math.isfinite(score):
        raise ValueError(f"Non-finite validation metric {BEST_METRIC}: {score}")
    directory = Path(trainer.config.trainer.default_local_dir)
    manifest = directory / "best.json"
    if manifest.exists() and score <= json.loads(manifest.read_text())["metric"]:
        return
    trainer._save_checkpoint()
    record = {
        "task_id": TASK_ID, "task_name": TASK_NAME, "metric_name": BEST_METRIC,
        "metric": score, "step": trainer.global_steps,
        "epoch": int(os.environ["CURRENT_EPOCH"]),
        "checkpoint": str(directory / f"global_step_{trainer.global_steps}"),
    }
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(manifest)


def select_task2_parents(run_directory):
    """Select one passing Task 1 parent per puzzle within each split."""
    root = Path(run_directory)
    source = root / "candidates" / "task1"
    files = sorted(source.glob("epoch_*/*.jsonl"))
    if not files:
        raise ValueError(f"No Task 1 candidate files under {source}")

    best = {"train": {}, "validation": {}}
    for path in files:
        relative = str(path.relative_to(root))
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                scores = row["metrics"]
                if scores.get("pa_accuracy_binary_pass") != 1 or scores.get("pa_consistency_binary_pass") != 1:
                    continue
                puzzle = row["puzzle_id"]
                if not puzzle or puzzle == "None":
                    raise ValueError(f"Missing puzzle ID at {relative}:{line_number}")
                split = row["split"]
                coverage = float(scores["coverage"])
                if not 0 < coverage <= 1:
                    raise ValueError(f"Invalid passing coverage at {relative}:{line_number}")
                rank = (coverage, scores.get("BASE_n_steps_novel_inc_clues", 0), -len(row["prefix"]))
                previous = best[split].get(puzzle)
                if previous is None or rank > previous[0]:
                    # Older v4 shards predate candidate_id. Their file and line locate the original.
                    parent = row.get("candidate_id") or f'{row["source_run"]}:{relative}:{line_number}'
                    best[split][puzzle] = (rank, dict(row, parent_candidate_id=parent,
                                                      source_file=relative, source_line=line_number))

    destination = root / "task2_inputs"
    destination.mkdir(exist_ok=True)
    for split, puzzles in best.items():
        path = destination / f"{split}.jsonl"
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for puzzle in sorted(puzzles):
                handle.write(json.dumps(puzzles[puzzle][1]) + "\n")
        temporary.replace(path)
        print(f"Selected {len(puzzles)} Task 1 parents for {split}: {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Task 2 parent lists from saved Task 1 candidates.")
    parser.add_argument("run_directory", help="Task 1 run directory containing candidates/task1")
    select_task2_parents(parser.parse_args().run_directory)
