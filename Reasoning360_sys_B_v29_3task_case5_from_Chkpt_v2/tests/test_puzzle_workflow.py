"""Run-local curriculum/checkpoint tests without GPU or optional dependencies."""
import importlib.util
import json
import os
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('workflow', ROOT / 'verl/utils/puzzle_workflow.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
fixture = runpy.run_path(str(ROOT / 'tests/test_three_task_reward_parser.py'))
ANSWER = fixture['ANSWER']


def metrics(coverage=1/3):
    return dict(stream_parse_ok=1, pa_accuracy_binary_pass=1, pa_consistency_binary_pass=1,
                BASE_sat_full_GT=1, BASE_n_non_valid_contradiction=0,
                BASE_n_steps_valid=3, BASE_n_steps_total=3,
                BASE_n_steps_novel_inc_clues=3, coverage=coverage)


def record(puzzle='a', split='train', coverage=1/3, count=1):
    answer = dict(ANSWER, reasoning=ANSWER['reasoning'][:count])
    return dict(puzzle_id=puzzle, split=split, question_prompt='question '+puzzle,
                ground_truth=ANSWER['reasoning'][-1]['partial_solution'],
                prefix=w.canonical_prefix(answer), metrics=metrics(coverage))


class WorkflowTests(unittest.TestCase):
    def test_source_label_and_job_id_select_one_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = 'task-1_runA-job-id-123-time-20261118-100000-model'
            b = 'task-1_runA-job-id-456-time-20261118-110000-model'
            for name in (a, b, 'task-1_runA'):
                (Path(tmp)/name).mkdir()
            self.assertEqual(w.resolve_source_run(tmp, 'task-1_runA', '123'), a)
            self.assertEqual(w.resolve_source_run(tmp, 'task-1_runA', '456'), b)
            self.assertEqual(w.resolve_source_run(tmp, a), a)
            self.assertEqual(w.resolve_source_run(tmp, 'task-1_runA'), 'task-1_runA')
            with self.assertRaises(ValueError):
                w.resolve_source_run(tmp, 'task-1_runA', '999')
            (Path(tmp)/'task-1_runA-job-id-123-time-duplicate').mkdir()
            with self.assertRaises(ValueError):
                w.resolve_source_run(tmp, 'task-1_runA', '123')

    def test_selection_filters_wrong_answers_and_keeps_all_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'candidates.jsonl'
            wrong = record(coverage=1)
            wrong['metrics']['pa_accuracy_binary_pass']=0
            w.write_jsonl(path,[record(),record(coverage=2/3,count=2),wrong])
            selected=w.select_prefixes([path],'train')
            self.assertEqual(len(w.read_jsonl(path)),3)
            self.assertEqual(selected[0]['metrics']['coverage'],2/3)
            self.assertEqual(w.select_prefixes([path],'validation'),[])

    def test_no_improvement_retains_previous(self):
        previous=record(coverage=2/3,count=2)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'c.jsonl'
            w.write_jsonl(path,[record()])
            self.assertEqual(w.select_prefixes([path],'train',[previous]),[previous])

    def test_complete_answer_becomes_eos_training_prefix(self):
        final=fixture['ENVELOPE']
        prefix=w.canonical_prefix(w.answer_object(final))
        self.assertTrue(prefix.endswith('</answer><EOSEG>'))
        self.assertEqual(w.answer_object(prefix),ANSWER)
        with patch.dict(os.environ,{'PUZZLE_TASK':'2'}):
            parsed=fixture['NS']['extract_reasoning_and_solution']('<EOS>',{'puzzle_prefix':prefix})
        self.assertEqual(parsed[-1],'success_direct_json')

    def test_candidate_capture_and_dataset_input(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{'PUZZLE_DATA_RUN':tmp,'CURRENT_EPOCH':'1'}):
            info={'id':'a','split':'train','workflow_training':True}
            w.capture_candidate('1','question a',fixture['TASK1'],record()['ground_truth'],info,metrics())
            paths=list(Path(tmp).glob('candidates/task1/epoch_0001/*.jsonl'))
            self.assertEqual(len(paths),1)
            selected=w.select_prefixes(paths,'train')
            row=w.dataset_rows(selected,'train')[0]
            self.assertEqual(row['prompt'][0]['content'],'question a\n'+selected[0]['prefix'])
            self.assertEqual(row['extra_info']['puzzle_prefix'],selected[0]['prefix'])
            # Training validation probes must not enter the candidate pool.
            w.capture_candidate('1','question a',fixture['TASK1'],record()['ground_truth'],{'id':'a','split':'train'},metrics())
            self.assertEqual(len(w.read_jsonl(paths[0])),1)

    def test_curriculum_epoch_transition_and_fixed_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source';dest=Path(tmp)/'dest'
            path=source/'candidates/task1/epoch_0001/all.jsonl'
            w.write_jsonl(path,[record('a'),record('v','validation')])
            written={}
            def parquet(path,rows,split):
                self.assertTrue(rows)
                written[str(path)]=w.dataset_rows(rows,split)
            with patch.object(w,'write_parquet',side_effect=parquet):
                w.prepare(source,dest)
                validation=(dest/'validation/prefixes.jsonl').read_text()
                w.write_jsonl(dest/'candidates/task2/epoch_0001/train_1.jsonl',[record('a',coverage=2/3,count=2)])
                next_file=w.advance(dest,1)
            self.assertTrue(next_file.endswith('epoch_0002.parquet'))
            selected=w.read_jsonl(dest/'prefixes/epoch_0001.jsonl')
            self.assertEqual(selected[0]['metrics']['coverage'],2/3)
            self.assertEqual((dest/'validation/prefixes.jsonl').read_text(),validation)
            self.assertEqual(set(written),{str(dest/'train/epoch_0001.parquet'),str(dest/'train/epoch_0002.parquet'),str(dest/'validation/continuations.parquet')})

    def test_prepare_rejects_train_validation_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source'
            w.write_jsonl(source/'candidates/task1/epoch_0001/a.jsonl',[record('a'),record('a','validation')])
            with self.assertRaisesRegex(ValueError,'overlap'):
                w.prepare(source,Path(tmp)/'dest')

    def test_best_checkpoint_replaces_only_its_own_previous_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage=Path(tmp)/'task1'
            saved=[]
            trainer=SimpleNamespace(config=SimpleNamespace(trainer=SimpleNamespace(default_local_dir=str(stage))),global_steps=0)
            def save():
                (stage/f'global_step_{trainer.global_steps}'/'actor').mkdir(parents=True)
                saved.append(trainer.global_steps)
            trainer._save_checkpoint=save
            key='val-aux/our_zebra_puzzle_new_reward_test/verified_coverage/mean@1'
            with patch.dict(os.environ,{'PUZZLE_TASK':'1'}):
                w.save_best(trainer,{key:0})
                trainer.global_steps=1;w.save_best(trainer,{key:0})
                trainer.global_steps=2;w.save_best(trainer,{key:.5})
            self.assertEqual(saved,[0,2])
            self.assertFalse((stage/'global_step_0').exists())
            self.assertTrue((stage/'global_step_2/actor').exists())
            self.assertEqual(w.best_checkpoint(tmp,1),str(stage/'global_step_2'))

if __name__=='__main__': unittest.main()
