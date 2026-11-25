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
NAMES = {'get_puzzle_task', '_read_puzzle_stream', '_read_task1_answer', '_read_answer_envelope', '_complete_table', '_task2_prefix', 'extract_reasoning_and_solution'}
NS = {'os': os, 'json': json, 're': re}
exec(compile(ast.Module(body=[n for n in SOURCE.body if isinstance(n, ast.FunctionDef) and n.name in NAMES], type_ignores=[]), str(REWARD), 'exec'), NS)
PROMPT_FILE = ROOT.parent / 'Reasoning360_sys_B_v4_data_models/data_preprocess/logic/v3task_v3/our_pre_process_zebrapuzzle_to_guru_parsed_v3task_v3_MLXL.py'
PROMPT = next(ast.literal_eval(n.value) for n in ast.parse(PROMPT_FILE.read_text()).body if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'SOLUTION_PROMPT_1_SHOT_SYS' for x in n.targets))
ENVELOPE = PROMPT.split('Correct Example Output:\n')[1].strip()
ANSWER = json.loads(ENVELOPE.split('<answer>', 1)[1].rsplit('</answer>', 1)[0].replace('</answer>', '').replace('<EOSEG>', ''))
OBJECTS = ANSWER['reasoning']
FORMAL = {k: v for k, v in ANSWER.items() if k != 'reasoning'}
# Tasks 2/3 retain their existing legacy stream implementation in this Task 1 change.
FULL = json.dumps(FORMAL) + '\n' + '\n'.join(json.dumps(obj) + '<EOSEG>' for obj in OBJECTS) + '<EOS>'
TASK1 = ENVELOPE.split('<EOSEG>', 1)[0] + '<EOSEG>' 

def segment(obj):
    return json.dumps(obj) + '<EOSEG>'

def prefix(objects):
    return json.dumps(FORMAL) + '\n' + '\n'.join(map(segment, objects))

class ParserTests(unittest.TestCase):
    def parse(self, task, text, info=None):
        with patch.dict(os.environ, {'PUZZLE_TASK': str(task)}):
            return NS['extract_reasoning_and_solution'](text, info)

    def test_first_segment(self):
        result = self.parse(1, TASK1)
        self.assertEqual(result[1], OBJECTS[0]['reasoning_steps'])
        self.assertEqual(result[2], OBJECTS[0]['partial_solution'])
        self.assertEqual(result[-1], 'success_direct_json')
        self.assertEqual(self.parse(1, FULL)[-1], 'parsing_failed')

    def test_task1_answer_after_thinking(self):
        result = self.parse(1, '<think>Working through the puzzle.</think>\n' + TASK1)
        self.assertEqual(result[-1], 'success_direct_json')
        self.assertEqual(result[2], OBJECTS[0]['partial_solution'])

    def test_task1_rejects_incomplete_objects_and_premature_stops(self):
        bad_outputs = [
            TASK1.replace('<answer>', '', 1),
            TASK1.removesuffix('<EOSEG>'),
            TASK1 + 'extra',
            TASK1 + '<EOS>',
            TASK1 + ',{}<EOSEG>',
            TASK1.replace('"partial_solution"', '"solution"', 1),
            TASK1.replace('"rows"', '"missing_rows"', 1),
            TASK1[:-len('<EOSEG>')].rstrip()[:-1] + '<EOSEG>',
            '<answer>' + json.dumps(FORMAL) + '<EOSEG>',
            '<answer>{"n_houses": 3, "reasoning": [' + json.dumps(OBJECTS[0]) + '<EOSEG>',
            TASK1.replace('"S1: red == 2."', '"S1: red <EOSEG> == 2."'),
        ]
        for text in bad_outputs:
            with self.subTest(text=text[-80:]):
                self.assertEqual(self.parse(1, text)[-1], 'parsing_failed')

    def test_full_example_is_valid_json_after_marker_removal(self):
        self.assertEqual(set(ANSWER), {'n_houses', 'attribute_values', 'syntactic_clues', 'reasoning'})
        self.assertEqual(len(OBJECTS), 2)
        first = OBJECTS[0]["partial_solution"]["rows"]
        self.assertEqual(sum(value != "?" for row in first for value in row[1:]), 7)
        self.assertEqual(ENVELOPE.count('<EOSEG>'), 1)
        self.assertTrue(ENVELOPE.endswith('</answer><EOS>'))
        self.assertTrue(all('?' not in row for row in OBJECTS[-1]['partial_solution']['rows']))
        self.assertEqual(self.parse(1, ENVELOPE)[-1], 'parsing_failed')
        self.assertEqual(json.loads(TASK1.split('<answer>')[1].removesuffix('</answer><EOSEG>') + '] }')['reasoning'], OBJECTS[:1])

    def test_task1_accepts_closed_complete_answer(self):
        text = '<answer>' + json.dumps(dict(FORMAL, reasoning=[{
            'reasoning_steps': sum([x['reasoning_steps'] for x in OBJECTS], []),
            'partial_solution': OBJECTS[-1]['partial_solution'],
        }])) + '</answer><EOS>'
        result = self.parse(1, text)
        self.assertEqual(result[-1], 'success_direct_json')
        self.assertEqual(len(result[1]), 18)
        self.assertEqual(result[2], OBJECTS[-1]['partial_solution'])

    def test_task1_rejects_partial_table_marked_final(self):
        text = '<answer>' + json.dumps(dict(FORMAL, reasoning=OBJECTS[:1])) + '</answer><EOS>'
        self.assertEqual(self.parse(1, text)[-1], 'parsing_failed')

    def test_repeated_envelope_full_and_prefix(self):
        result = self.parse(3, ENVELOPE)
        self.assertEqual(result[-1], 'success_direct_json')
        self.assertEqual(len(result[1]), 18)
        self.assertEqual(result[2], OBJECTS[-1]['partial_solution'])
        prefix_two = ENVELOPE.rsplit(']', 1)[0].rstrip() + '</answer><EOSEG>'
        formal, objects, eos = NS['_read_answer_envelope'](prefix_two)
        self.assertEqual(objects, OBJECTS[:2])
        self.assertFalse(eos)
        self.assertEqual(self.parse(1, prefix_two)[-1], 'parsing_failed')

    def test_repeated_envelope_rejects_corrupt_latest_segment(self):
        for text in (ENVELOPE.replace('"S9: yellow == 1."', '"broken'),
                     ENVELOPE.replace('</answer><EOSEG>', '<EOSEG>', 1),
                     ENVELOPE.replace('</answer><EOSEG>', '</answer><EOSEG><answer>', 1),
                     ENVELOPE + 'trailing text'):
            with self.subTest(text=text[-80:]):
                self.assertEqual(self.parse(3, text)[-1], 'parsing_failed')

    def test_new_task2_continuation_and_termination(self):
        continuation = ENVELOPE.split('</answer><EOSEG>', 1)[1]
        result = self.parse(2, continuation, {'puzzle_prefix': TASK1})
        self.assertEqual(result[-1], 'success_direct_json')
        self.assertEqual(result[2], OBJECTS[-1]['partial_solution'])
        complete_prefix = ENVELOPE.rsplit('\n  ]', 1)[0] + '</answer><EOSEG>'
        self.assertEqual(self.parse(2, '<EOS>', {'puzzle_prefix': complete_prefix})[-1], 'success_direct_json')

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
        result = self.parse(2, segment(OBJECTS[-1]), {'puzzle_prefix': prefix(OBJECTS[:1])})
        self.assertEqual(result[-1], 'success_direct_json')
        self.assertEqual(result[2], OBJECTS[-1]['partial_solution'])

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
