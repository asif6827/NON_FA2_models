"""Dependency-free checks of the production sampling branches and mask helper.

The real vLLM/PyTorch integration must also be checked on the training cluster.
"""
import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE_PATH = Path(__file__).resolve().parents[1] / 'verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py'
TREE = ast.parse(SOURCE_PATH.read_text())
ROLLOUT = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'vLLMRollout')
GEN = next(n for n in ROLLOUT.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_sequences')
NS = {'contextmanager': contextmanager}
helpers = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in ('_task1_stop_kwargs', '_generated_length_mask')]
exec(compile(ast.Module(body=helpers, type_ignores=[]), str(SOURCE_PATH), 'exec'), NS)
update = next(n for n in ROLLOUT.body if isinstance(n, ast.FunctionDef) and n.name == 'update_sampling_params')
exec(compile(ast.Module(body=[update], type_ignores=[]), str(SOURCE_PATH), 'exec'), NS)

# Execute the actual sampling branch from generate_sequences without requiring a GPU.
start = next(i for i, n in enumerate(GEN.body) if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'do_sample' for x in n.targets))
end = next(i for i, n in enumerate(GEN.body) if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'lora_requests' for x in n.targets))
SAMPLING = compile(ast.Module(body=GEN.body[start:end], type_ignores=[]), str(SOURCE_PATH), 'exec')

class Vector:
    def __init__(self, values): self.values = values
    def unsqueeze(self, axis): return Expanded(self.values, axis)

class Expanded:
    def __init__(self, values, axis): self.values, self.axis = values, axis
    def __lt__(self, other):
        assert self.axis == 0 and other.axis == 1
        return Matrix([[int(p < length) for p in self.values] for length in other.values])

class Matrix:
    def __init__(self, rows): self.rows = rows
    def to(self, dtype): return self.rows

class FakeTorch:
    long = 'long'
    @staticmethod
    def tensor(values, **kwargs): return Vector(values)
    @staticmethod
    def arange(width, **kwargs): return Vector(list(range(width)))

class RolloutStopTests(unittest.TestCase):
    def sampling(self, task, meta, incoming=None):
        rollout = SimpleNamespace(puzzle_task=task, config=SimpleNamespace(val_kwargs=SimpleNamespace(top_k=-1, top_p=.9, temperature=.2)))
        ns = dict(NS, self=rollout, prompts=SimpleNamespace(meta_info=meta), kwargs=dict(incoming or {}))
        exec(SAMPLING, ns)
        return ns['kwargs']

    def test_training_validation_and_greedy_enforce_boundary(self):
        for meta in ({}, {'validate': True}, {'do_sample': False}, {'validate': True, 'do_sample': False}):
            with self.subTest(meta=meta):
                settings = self.sampling('1', meta, {'stop': ['wrong'], 'detokenize': False})
                self.assertEqual(settings['stop'], ['<EOSEG>'])
                self.assertTrue(settings['detokenize'])
                self.assertTrue(settings['include_stop_str_in_output'])
                self.assertFalse(settings['skip_special_tokens'])
                self.assertEqual(settings['stop_token_ids'], [])
                self.assertEqual(settings['min_tokens'], 0)

    def test_other_tasks_keep_existing_stopping(self):
        for task in (None, '2', '3'):
            settings = self.sampling(task, {}, {'stop': ['existing'], 'detokenize': False})
            self.assertEqual(settings, {'stop': ['existing'], 'detokenize': False})

    def test_sampling_parameters_restored_on_failure(self):
        obj = SimpleNamespace(sampling_params=SimpleNamespace(stop=['old'], temperature=.8))
        with self.assertRaises(RuntimeError):
            with NS['update_sampling_params'](obj, stop=['<EOSEG>'], temperature=0):
                self.assertEqual(obj.sampling_params.stop, ['<EOSEG>'])
                raise RuntimeError('engine error')
        self.assertEqual(obj.sampling_params.stop, ['old'])
        self.assertEqual(obj.sampling_params.temperature, .8)

    def test_mask_includes_complete_generated_stop_and_excludes_padding(self):
        # Variable lengths: a multi-token stop, short native-EOS response,
        # zero output, and a response reaching the maximum length.
        response = SimpleNamespace(device='cpu', size=lambda axis: 6)
        helper_globals = NS['_generated_length_mask'].__globals__
        helper_globals['torch'] = FakeTorch
        mask = NS['_generated_length_mask'](response, [4, 1, 0, 6], 'int64')
        self.assertEqual(mask, [[1,1,1,1,0,0], [1,0,0,0,0,0], [0,0,0,0,0,0], [1,1,1,1,1,1]])

if __name__ == '__main__':
    unittest.main()
