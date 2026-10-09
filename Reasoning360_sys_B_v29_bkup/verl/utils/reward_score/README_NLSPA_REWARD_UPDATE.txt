NL/S/PA Reward Stack Update
============================

New prompt reasoning schema
---------------------------
"reasoning": {
  "NL1": "...",
  "S1": "Arnold == 2.",
  "PA1": {
    "header": [...],
    "rows": [...]
  },
  ...
}

Files
-----
1. nlspa_reward_utils.py
   Shared schema adapters and PRM calculations.

2. check_interleved_format_v7_nlspa.py
   New format checker for the reasoning JSON object.

3. z3_reasoning_validator_v13_gt_solve_v10_nlspa.py
   Compatibility layer over your existing
   z3_reasoning_validator_v13_gt_solve_v9.py.
   It extracts only S_i and reuses the existing Z3 validity/novelty engine.

4. z3_reasoning_vs_solution_verifier_v3_nlspa.py
   Compatibility layer over your existing
   z3_reasoning_vs_solution_verifier_v2.py.
   It checks ALL S_i steps against the final solution.

5. our_puzzle_dataset_v7_nlspa.py
   New Zebra reward computation.

Required existing files
-----------------------
Keep these existing stable files beside/in the same Python package:
- our_puzzle_dataset_v6.py
- z3_reasoning_validator_v13_gt_solve_v9.py
- z3_reasoning_vs_solution_verifier_v2.py

The NL/S/PA files intentionally reuse their stable parsing/Z3 internals rather
than duplicating thousands of lines.

Recommended VERL reward module
------------------------------
Change the Zebra reward module/config from:
    our_puzzle_dataset_v6

to:
    our_puzzle_dataset_v7_nlspa

New S-PRM
---------
S-PRM is normalized by the ACTUAL number of S_i steps:
- s_parse_ratio
- s_validity_ratio
- s_novelty_ratio
- s_contradiction_ratio
- consistency_score: all S_i vs final predicted solution
- s_prm_score

This fixes the old grid-size normalization, which made the process signal very
small when the model emitted only a modest number of S_i deductions.

New PA-PRM
----------
For each PA_i:
- exact header/row structure
- attribute-domain validity
- uniqueness among resolved cells
- correctness of resolved cells vs GT
- prefix support from syntactic_clues + S1..Si positive equality closure
- monotonicity across PA checkpoints
- progressive filling

Unknown "?" cells are neutral.

Important: prefix support intentionally does NOT full-solve the puzzle with Z3.
Otherwise a unique puzzle could make every future GT cell "supported" at PA1.
It uses explicit positive equality closure from clues and the available S-prefix.

Combined process PRM
--------------------
If no PA is emitted:
    process_prm = s_prm

If PA is emitted:
    process_prm = 0.70 * s_prm + 0.30 * pa_prm

PA is therefore optional and its absence is not directly penalized.

Scalar reward
-------------
The old approximate reward scale is preserved:

If base Z3/GT formalization fails:
    0.15 * parsing
  + 0.10 * format
  + 0.60 * puzzle_accuracy
  - 0.20 * s_contradiction_ratio

Otherwise:
    base_quality =
        0.60 * puzzle_accuracy
      + 0.20 * parsing
      + 0.20 * format

    process_bonus = 0.70 * process_prm

    reward =
        base_quality
      + puzzle_accuracy * process_bonus

The process bonus remains gated by final puzzle correctness, matching the prior
training behavior.

Two old reward bugs fixed
-------------------------
1. The required <answer>...</answer> output now gets parsing_reward=1.
   In v6, extract_reasoning_and_solution returned "success_answer_tag" but the
   reward block checked different success strings.

2. Zero novel S steps no longer automatically forces reward=-0.5.
   Required inputs now depend on the presence of a valid S trace, not
   n_novel_steps > 0.

Backward-compatible metric names
--------------------------------
These are retained:
- BASE_n_steps_total
- BASE_n_steps_parsed_ok
- BASE_n_steps_valid
- BASE_n_steps_novel_inc_clues
- BASE_n_non_valid_contradiction
- novel_step_score
- contradiction_ratio
- consistency_score

They now refer only to S_i steps.

New logged metrics
------------------
- NLSPA_format_ok
- NLSPA_n_nl
- NLSPA_n_s
- NLSPA_n_pa

- s_parse_ratio
- s_validity_ratio
- s_novelty_ratio
- s_contradiction_ratio
- s_prm_score

- pa_present
- PA_n_total
- PA_n_structurally_valid
- PA_resolved_cells
- PA_correct_resolved_cells
- PA_incorrect_resolved_cells
- PA_supported_resolved_cells
- PA_unsupported_resolved_cells
- pa_structure_score
- pa_cell_precision
- pa_final_coverage
- pa_effective_progress
- pa_monotonicity_score
- pa_transition_progress_score
- pa_prefix_support_score
- pa_prm_score

- process_prm_score

Validation performed
--------------------
Against the one-shot in zebrapuzzle_to_guru_parsed_MLXL_v9.py:
- reasoning schema: PASS
- NL/S/PA counts: 9 / 9 / 2
- PA1 prefix support: 1.0
- PA2 prefix support: 1.0
- PA resolved-cell precision: 1.0
- PA monotonicity: 1.0


Canonical test/example format
-----------------------------
All updated scripts now use the same NEW reasoning representation in their
runnable examples:

"reasoning": {
  "NL1": "...",
  "S1": "Arnold == 2.",
  ...
  "PA1": {
    "header": ["House", "Name", "Color", "Children"],
    "rows": [
      ["1", "?", "?", "Bella"],
      ["2", "Arnold", "red", "?"],
      ["3", "Eric", "?", "?"]
    ]
  },
  ...
}

No updated example uses the old:
    reasoning = ["NL...", "S1: ...", ...]

The complete regression example contains:
- NL1..NL9
- S1..S9
- PA1 after S3
- PA2 after S7
- the final 3-house solution.
