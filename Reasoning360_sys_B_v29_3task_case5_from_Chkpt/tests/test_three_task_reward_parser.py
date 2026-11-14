"""Dependency-free regression tests for the adapter, using the actual prompt demo.

Run: python3 tests/test_three_task_reward_parser.py
Only parser definitions are loaded; importing the reward module requires the
cluster's numpy/Z3 environment and external puzzle dictionary.
"""
import ast
import json
import os
from pathlib import Path
import re
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
REWARD = ROOT / 'verl/utils/reward_score/our_puzzle_dataset_v6.py'
SOURCE = ast.parse(REWARD.read_text())
NAMES = {'get_puzzle_task', '_read_puzzle_stream', '_complete_table', '_task2_prefix', 'extract_reasoning_and_solution'}
NS = {'os': os, 'json': json, 're': re}
exec(compile(ast.Module(body=[n for n in SOURCE.body if isinstance(n, ast.FunctionDef) and n.name in NAMES], type_ignores=[]), str(REWARD), 'exec'), NS)
PROMPT_FILE = ROOT.parent / 'Reasoning360_sys_B_v4_data_models/data_preprocess/logic/v3task/our_pre_process_zebrapuzzle_to_guru_parsed_v3task_MLXL.py'
PROMPT = next(ast.literal_eval(n.value) for n in ast.parse(PROMPT_FILE.read_text()).body if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'SOLUTION_PROMPT_1_SHOT_SYS' for x in n.targets))
FULL = PROMPT.split('Correct Example Output:\n')[1].strip()
FORMAL, OBJECTS, EOS = NS['_read_puzzle_stream'](FULL)

def segment(obj):
    return json.dumps(obj) + '<EOSEG>'

def prefix(objects):
    return json.dumps(FORMAL) + '\n' + '\n'.join(map(segment, objects))

class ParserTests(unittest.TestCase):
    def parse(self, task, text, info=None):
        with patch.dict(os.environ, {'PUZZLE_TASK': str(task)}):
            return NS['extract_reasoning_and_solution'](text, info)

    def test_first_segment(self):
        result = self.parse(1, prefix(OBJECTS[:1]))
        self.assertEqual(result[1], OBJECTS[0]['reasoning_steps'])
        self.assertEqual(result[2], OBJECTS[0]['partial_solution'])
        self.assertEqual(result[-1], 'success_direct_json')
        self.assertEqual(self.parse(1, FULL)[-1], 'parsing_failed')

    def test_continuation_includes_prior_support(self):
        result = self.parse(2, segment(OBJECTS[1]), {'puzzle_prefix': prefix(OBJECTS[:1])})
        self.assertEqual(result[-1], 'success_direct_json')
        self.assertEqual(result[1], OBJECTS[0]['reasoning_steps'] + OBJECTS[1]['reasoning_steps'])
        self.assertEqual(result[2], OBJECTS[1]['partial_solution'])
        self.assertEqual(self.parse(2, segment(OBJECTS[1]))[-1], 'parsing_failed')

    def test_prompt_demo_is_not_used_as_context(self):
        prompt = PROMPT + '\nPUZZLE TO SOLVE\nquestion\n' + prefix(OBJECTS[:1])
        self.assertEqual(self.parse(2, segment(OBJECTS[1]), {'puzzle_prompt': prompt})[-1], 'success_direct_json')
        self.assertEqual(self.parse(2, segment(OBJECTS[1]), {'puzzle_prompt': PROMPT + '\nPUZZLE TO SOLVE\nquestion'})[-1], 'parsing_failed')

    def test_complete_prefix_accepts_only_eos(self):
        info = {'puzzle_prefix': prefix(OBJECTS)}
        result = self.parse(2, '<EOS>', info)
        self.assertEqual(result[-1], 'success_direct_json')
        self.assertEqual(len(result[1]), 18)
        self.assertEqual(self.parse(2, segment(OBJECTS[-1]), info)[-1], 'parsing_failed')
        self.assertEqual(self.parse(2, '<EOS>', {'puzzle_prefix': prefix(OBJECTS[:1])})[-1], 'parsing_failed')

    def test_final_continuation_can_complete_table(self):
        result = self.parse(2, segment(OBJECTS[2]), {'puzzle_prefix': prefix(OBJECTS[:2])})
        self.assertEqual(result[-1], 'success_direct_json')
        self.assertEqual(result[2], OBJECTS[2]['partial_solution'])

    def test_inference_flattens_all_steps(self):
        result = self.parse(3, FULL)
        self.assertEqual(result[-1], 'success_direct_json')
        self.assertEqual(result[1], sum([obj['reasoning_steps'] for obj in OBJECTS], []))
        self.assertEqual(result[2], OBJECTS[-1]['partial_solution'])
        self.assertEqual(self.parse(3, FULL.removesuffix('<EOS>'))[-1], 'parsing_failed')
        self.assertEqual(self.parse(3, prefix(OBJECTS[:1]) + '<EOS>')[-1], 'parsing_failed')

    def test_malformed_boundaries_and_trailing_data_fail(self):
        for text in (FULL.replace('<EOSEG>', '', 1), FULL + 'extra', FULL.replace('<EOSEG>', '"<EOSEG>"', 1), prefix(OBJECTS[:1]) + '{bad json', FULL.replace('"partial_solution"', '"solution"', 1)):
            with self.subTest(text=text[-60:]):
                self.assertEqual(self.parse(3, text)[-1], 'parsing_failed')

    def test_cumulative_assignments_cannot_change(self):
        altered = json.loads(json.dumps(OBJECTS[1]))
        altered['partial_solution']['rows'][1][1] = 'Peter'
        self.assertEqual(self.parse(2, segment(altered), {'puzzle_prefix': prefix(OBJECTS[:1])})[-1], 'parsing_failed')

    def test_invalid_task_is_configuration_error(self):
        for value in ('', '0', '4', 'foo'):
            with patch.dict(os.environ, {'PUZZLE_TASK': value}), self.assertRaises(ValueError):
                NS['extract_reasoning_and_solution'](FULL)

    def test_score_formula_unchanged(self):
        original = ast.parse(REWARD.with_name(REWARD.name + '.before_3task_reward.bak').read_text())
        def formula(tree):
            compute = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_score')
            return [ast.dump(n) for n in ast.walk(compute) if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id in {'reward', 'normalizer', 'novel_step_score', 'contradiction_ratio', 'base_quality', 'process_bonus', 'format_reward'} for x in n.targets)]
        self.assertEqual(formula(SOURCE), formula(original))

if __name__ == '__main__':
    unittest.main()
