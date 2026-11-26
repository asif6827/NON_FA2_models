"""Run-local artifacts and verified-prefix curriculum for the three puzzle tasks."""
import argparse
import json
import os
from pathlib import Path
import shutil


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    temporary.replace(path)


def verified(metrics):
    return (metrics.get('stream_parse_ok') == 1
            and metrics.get('pa_accuracy_binary_pass') == 1
            and metrics.get('pa_consistency_binary_pass') == 1
            and metrics.get('BASE_sat_full_GT') == 1
            and metrics.get('BASE_n_non_valid_contradiction', 0) == 0
            and metrics.get('BASE_n_steps_total', 0) > 0
            and metrics.get('BASE_n_steps_novel_inc_clues', 0) > 0
            and metrics.get('BASE_n_steps_valid', 0) == metrics.get('BASE_n_steps_total', -1))


def rank(record):
    metrics = record['metrics']
    return (float(metrics.get('coverage', 0)),
            float(metrics.get('BASE_n_steps_novel_inc_clues', 0)),
            -len(record['prefix']))


def answer_object(prefix):
    """Reconstruct an evaluated stream; evaluation gates must pass before selection."""
    text = prefix.rsplit('</think>', 1)[-1]
    text = text.split('<answer>', 1)[1]
    final = text.rstrip().endswith('</answer><EOS>')
    text = text.rsplit('</answer>', 1)[0].replace('</answer>', '').replace('<EOSEG>', '')
    return json.loads(text if final else text + ']}')


def canonical_prefix(answer):
    # Completed answers are kept as open prefixes so the next completion is EOS.
    formal = {key: answer[key] for key in ('n_houses', 'attribute_values', 'syntactic_clues')}
    return ('<answer>' + json.dumps(formal)[:-1] + ', "reasoning": ['
            + ','.join(json.dumps(obj) + '</answer><EOSEG>' for obj in answer['reasoning']))


def capture_candidate(task, prompt, output, ground_truth, info, metrics):
    directory = os.environ.get('PUZZLE_DATA_RUN')
    if not directory or task not in ('1', '2'):
        return
    info = info or {}
    split = str(info.get('split', ''))
    split = 'validation' if split in ('test', 'valid', 'validation') else 'train'
    # Training-validation probes are diagnostics, not extra GRPO candidates.
    if split == 'train' and not info.get('workflow_training', False):
        return
    old = info.get('puzzle_prefix', '')
    combined = old if task == '2' and output.strip() == '<EOS>' else old + output
    record = {
        'puzzle_id': str(info.get('id', info.get('puzzle_id', ''))),
        'split': split, 'question_prompt': info.get('question_prompt', prompt),
        'ground_truth': ground_truth, 'prefix': combined, 'output': output,
        'metrics': metrics, 'epoch': int(os.environ.get('CURRENT_EPOCH', '0')),
        'checkpoint_step': info.get('workflow_step'),
        'source_run': os.environ.get('PUZZLE_RUN_NAME'),
    }
    if verified(metrics):
        record['prefix'] = canonical_prefix(answer_object(combined))
    epoch = record['epoch']
    path = Path(directory) / 'candidates' / f'task{task}' / f'epoch_{epoch:04d}' / f'{split}_{os.getpid()}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(json.dumps(record, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')


def select_prefixes(paths, split, previous=()):
    selected = {str(row['puzzle_id']): row for row in previous}
    for path in paths:
        for record in read_jsonl(path):
            if record.get('split') != split or not verified(record.get('metrics', {})):
                continue
            key = str(record['puzzle_id'])
            if not key:
                raise ValueError('Candidate has no stable puzzle ID')
            if key in selected and selected[key]['question_prompt'] != record['question_prompt']:
                raise ValueError(f'Puzzle ID collision: {key}')
            if key not in selected or rank(record) > rank(selected[key]):
                selected[key] = record
    return [selected[key] for key in sorted(selected)]


def dataset_rows(prefixes, split):
    return [{
        'data_source': 'our_zebra_puzzle_new_reward' + ('_test' if split == 'validation' else ''),
        'prompt': [{'role': 'user', 'content': row['question_prompt'] + '\n' + row['prefix']}],
        'raw_prompt': [{'role': 'user', 'content': row['question_prompt'] + '\n' + row['prefix']}],
        'ability': 'logical_reasoning', 'apply_chat_template': False,
        'reward_model': {'style': 'rule', 'ground_truth': row['ground_truth']},
        'extra_info': {'id': row['puzzle_id'], 'split': 'test' if split == 'validation' else 'train',
                       'question_prompt': row['question_prompt'], 'puzzle_prefix': row['prefix']},
    } for row in prefixes]


def write_parquet(path, prefixes, split):
    if not prefixes:
        raise ValueError(f'No verified {split} prefixes. Collect valid Task 1 candidates first.')
    from datasets import Dataset
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.parquet')
    Dataset.from_list(dataset_rows(prefixes, split)).to_parquet(str(temp))
    temp.replace(path)


def prepare(source, destination):
    source, destination = Path(source), Path(destination)
    if (destination / 'state.json').exists():
        raise ValueError('Destination already contains a curriculum; use a fresh PUZZLE_RUN_NAME')
    paths = list((source / 'candidates' / 'task1').glob('epoch_*/*.jsonl'))
    train = select_prefixes(paths, 'train')
    validation = select_prefixes(paths, 'validation')
    if set(row['puzzle_id'] for row in train) & set(row['puzzle_id'] for row in validation):
        raise ValueError('Training and validation puzzle IDs overlap')
    if not train or not validation:
        raise ValueError('Need verified Task 1 training AND held-out validation candidates')
    write_jsonl(destination / 'prefixes' / 'initial.jsonl', train)
    write_jsonl(destination / 'validation' / 'prefixes.jsonl', validation)
    write_parquet(destination / 'train' / 'epoch_0001.parquet', train, 'train')
    write_parquet(destination / 'validation' / 'continuations.parquet', validation, 'validation')
    atomic_json(destination / 'state.json', {'epoch': 0, 'train_puzzles': len(train), 'validation_puzzles': len(validation)})


def advance(directory, epoch):
    directory = Path(directory)
    previous_file = directory / 'prefixes' / ('initial.jsonl' if epoch == 1 else f'epoch_{epoch-1:04d}.jsonl')
    previous = read_jsonl(previous_file)
    paths = list((directory / 'candidates' / 'task2' / f'epoch_{epoch:04d}').glob('train_*.jsonl'))
    selected = select_prefixes(paths, 'train', previous)
    write_jsonl(directory / 'prefixes' / f'epoch_{epoch:04d}.jsonl', selected)
    next_file = directory / 'train' / f'epoch_{epoch+1:04d}.parquet'
    write_parquet(next_file, selected, 'train')
    atomic_json(directory / 'state.json', {'epoch': epoch, 'train_puzzles': len(selected)})
    return str(next_file)


def best_checkpoint(run_directory, task):
    manifest = Path(run_directory) / f'task{task}' / 'best.json'
    record = json.loads(manifest.read_text())
    path = Path(record['checkpoint'])
    if not (path / 'actor').is_dir():
        raise ValueError(f'Missing actor checkpoint: {path}')
    return str(path)


def save_best(trainer, metrics):
    task = os.environ.get('PUZZLE_TASK', '1')
    suffix = 'verified_coverage' if task in ('1', '2') else 'PUZZLE_ACCURACY'
    key = f'val-aux/our_zebra_puzzle_new_reward_test/{suffix}/mean@1'
    if key not in metrics:
        raise ValueError(f'Missing checkpoint-selection metric: {key}')
    score = float(metrics[key])
    directory = Path(trainer.config.trainer.default_local_dir)
    manifest = directory / 'best.json'
    old = json.loads(manifest.read_text()) if manifest.exists() else None
    if old is not None and score <= old['metric']:
        return
    trainer._save_checkpoint()
    checkpoint = directory / f'global_step_{trainer.global_steps}'
    atomic_json(manifest, {'task': int(task), 'metric_name': key, 'metric': score,
                          'checkpoint': str(checkpoint), 'step': trainer.global_steps,
                          'epoch': int(os.environ.get('CURRENT_EPOCH', '0'))})
    # Delete only our own older best, after the new checkpoint was saved.
    if old:
        previous = Path(old['checkpoint'])
        if previous != checkpoint and previous.parent == directory and previous.name.startswith('global_step_') and previous.exists():
            shutil.rmtree(previous)


def refresh_trainer(trainer, epoch):
    path = advance(os.environ['PUZZLE_DATA_RUN'], epoch)
    from verl.trainer.main_ppo import create_rl_dataset
    dataset = create_rl_dataset([path], trainer.config.data, trainer.tokenizer, trainer.processor)
    if len(dataset) != len(trainer.train_dataset):
        raise ValueError('Continuation filtering changed puzzle count; increase PUZZLE_MAX_PROMPT_LENGTH instead of silently dropping prefixes')
    # Keep the original validation set, scheduler horizon, and optimizer state.
    horizon = trainer.total_training_steps
    trainer.config.data.train_files = [path]
    trainer._create_dataloader(dataset, trainer.val_dataset, trainer.train_dataloader.collate_fn, None)
    trainer.total_training_steps = horizon


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--source', required=True)
    prep.add_argument('--destination', required=True)
    checkpoint = sub.add_parser('checkpoint')
    checkpoint.add_argument('--run', required=True)
    checkpoint.add_argument('--task', required=True, type=int, choices=(1, 2))
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.source, args.destination)
    else:
        print(best_checkpoint(args.run, args.task))
