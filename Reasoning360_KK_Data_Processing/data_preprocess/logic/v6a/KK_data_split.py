import json
import random
from pathlib import Path

SEED = 42
random.seed(SEED)

ROOT = Path.home() / "HF_cache" / "knights-and-knaves"

OUTPUT = Path("knights_and_knaves_300_train")
TRAIN_OUT = OUTPUT / "train"
TEST_OUT = OUTPUT / "test"

TRAIN_OUT.mkdir(parents=True, exist_ok=True)
TEST_OUT.mkdir(parents=True, exist_ok=True)

# Number of training examples from each puzzle size
TRAIN_COUNTS = {
    2: 43,
    3: 43,
    4: 43,
    5: 43,
    6: 43,
    7: 43,
    8: 42,
}


def read_jsonl(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def write_jsonl(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


all_train = []
all_test = []

# -----------------------------
# Training
# -----------------------------
for people, n in TRAIN_COUNTS.items():

    infile = ROOT / "train" / f"people{people}_num{'200' if people==2 else '1000'}.jsonl"

    data = read_jsonl(infile)

    sampled = random.sample(data, n)

    outfile = TRAIN_OUT / f"people{people}_num{n}.jsonl"

    write_jsonl(outfile, sampled)

    all_train.extend(sampled)

    print(f"Train {people} people : {len(sampled)}")

write_jsonl(
    OUTPUT / f"train_all_num{len(all_train)}.jsonl",
    all_train,
)

# -----------------------------
# Testing (use entire official split)
# -----------------------------
for people in range(2, 9):

    infile = ROOT / "test" / f"people{people}_num100.jsonl"

    data = read_jsonl(infile)

    outfile = TEST_OUT / f"people{people}_num100.jsonl"

    write_jsonl(outfile, data)

    all_test.extend(data)

    print(f"Test  {people} people : {len(data)}")

write_jsonl(
    OUTPUT / f"test_all_num{len(all_test)}.jsonl",
    all_test,
)

write_jsonl(
    OUTPUT / f"all_selected_num{len(all_train)+len(all_test)}.jsonl",
    all_train + all_test,
)

print("\nSummary")
print("------------------------------")
print(f"Training : {len(all_train)}")
print(f"Testing  : {len(all_test)}")
print(f"Total    : {len(all_train)+len(all_test)}")