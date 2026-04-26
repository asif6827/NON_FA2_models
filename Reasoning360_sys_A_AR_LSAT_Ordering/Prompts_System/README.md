# Puzzle Solving System with Z3 Verification

A modular puzzle solving system that combines Large Language Models (LLMs) with Z3 constraint solving for robust puzzle verification and solving.

## Overview

This system provides two distinct approaches to puzzle solving:
1. **Solution-based verification**: Generates solutions first, then verifies them using Z3 constraints and LLM verification
2. **Constraint-based verification**: Generates constraints first, solves them using Z3, then verifies the constraints

Both approaches leverage LLMs for solution/constraint generation and verification, with Z3 used for rigorous constraint checking.

## Directory Structure

```
final_code/
├── main.py                # Main entry point
├── prompts.py             # All prompt templates
├── metrics.py             # Metrics calculation
├── logger.py              # Logging functionality
├── utils/                 # Utility functions
│   ├── dataset.py         # Dataset loading
│   ├── grid.py            # Grid processing
│   └── json.py            # JSON extraction
├── z3_impl/               # Z3 implementations
│   ├── constraint_checker.py  # Constraint-based verification
│   └── grid_solver.py         # Grid-based constraint solving
└── verification_system/   # Verification systems
    ├── base.py            # Base verification system
    ├── solution_verifier.py   # Solution-based verification
    └── constraint_verifier.py # Constraint-based verification
```

## Installation

### Prerequisites
- Python 3.8+
- vLLM
- Z3 Theorem Prover
- pandas
- tqdm

### Installation Steps

1. Install required packages:
```bash
pip install vllm z3-solver pandas tqdm
```

2. Clone or download the project files to your local machine.

## Usage

### Running the System

The main entry point is `main.py`, which supports both verification modes:

```bash
# Solution-based verification (default mode)
python prompt_iterate.py --mode solution

# Constraint-based verification
python prompt_iterate.py --mode constraint
```

### Command Line Arguments

| Argument | Description | Default Value |
|----------|-------------|---------------|
| `--model_path` | Path to the local LLM model | `/root/autodl-tmp/model/Qwen2.5-3B-Instruct` |
| `--data_path` | Path to the dataset file (JSON, JSONL, or Parquet) | `/root/autodl-tmp/Asif/code/output/easy_size_data.jsonl` |
| `--output_dir` | Directory for output results | `/root/autodl-tmp/Asif/code/puzzle/output` |
| `--mode` | Verification mode: `solution` or `constraint` | `solution` |
| `--n_samples` | Number of samples per prompt | 2 |
| `--temperature` | Sampling temperature | 0.7 |
| `--top_p` | Top-p sampling parameter | 0.9 |
| `--tokenizer_mode` | Tokenizer mode | `auto` |
| `--limit` | Only process the first K samples (-1 for all) | -1 |
| `--max_attempts` | Maximum refinement attempts per sample | 15 |
| `--refinement_include_z3` | Include Z3 check results in refinement feedback (solution mode only) | True |
| `--refinement_include_accuracy` | Include accuracy info in refinement feedback (solution mode only) | False |
| `--refinement_include_verification` | Include verification results in refinement feedback (solution mode only) | True |
| `--log_level` | Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL | INFO |

## Verification Modes

### 1. Solution-based Verification

**Workflow:**
1. Generate solution using LLM
2. Verify solution with Z3 constraints
3. Verify solution against puzzle clues using LLM
4. If verification passes, check against ground truth
5. If incorrect, refine solution based on feedback and repeat

**Key Features:**
- Verification-first approach
- Supports refinement with Z3, accuracy, and verification feedback
- Uses `Z3ConstraintChecker` for constraint validation

### 2. Constraint-based Verification

**Workflow:**
1. Generate constraints from puzzle text using LLM
2. Build Z3 grid model from constraints
3. Solve using Z3
4. Verify constraints using LLM
5. If verification passes, check solution against ground truth
6. If constraints are invalid, refine them and repeat

**Key Features:**
- Constraint-first approach
- Uses `GridModel` for Z3 solving
- Supports constraint refinement

## Refinement Feedback Control (Solution Mode Only)

The solution mode supports three types of refinement feedback:

| Feedback Type | Description | Argument | Default |
|---------------|-------------|----------|---------|
| Z3 Analysis | Includes Z3 constraint check results | `--refinement_include_z3` | True |
| Accuracy | Includes ground truth accuracy information | `--refinement_include_accuracy` | False |
| Verification | Includes LLM clue verification results | `--refinement_include_verification` | True |

These can be combined to provide different levels of feedback for solution refinement.

## Z3 Constraint Usage

### Solution Mode
- Extracts attributes and values from ground truth
- Builds Z3 constraint checker with base constraints
- Checks generated solutions against these constraints
- Provides feedback for refinement

### Constraint Mode
- Generates constraints from puzzle text
- Builds grid model with these constraints
- Solves model using Z3
- Verifies constraints for correctness


### Basic Usage
```bash
# Run solution-based verification on first 5 samples
python prompt_iterate.py --mode solution --limit 5

# Run constraint-based verification with higher temperature
python prompt_iterate.py --mode constraint --temperature 0.9 --n_samples 3
```
