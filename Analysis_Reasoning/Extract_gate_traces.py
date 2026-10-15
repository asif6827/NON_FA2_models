#!/usr/bin/env python3
"""Group SPRING feedback by its logged accuracy/consistency gates.

Python 3.9+, standard library only. Accepts JSONL, a JSON array, or a single
feedback JSON object. Does not recompute gates or infer reasons for failures.
For each group, writes a readable .txt file and original records in .jsonl.
By default reads ./Input_Logs/ and writes ./Output/<input_filename_stem>/.
Existing output folders are preserved by adding a timestamp to new folders.
Readable output is also printed by default. Use --quiet to print only summary.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime
import json
from pathlib import Path
import sys

GROUPS = {
    (True, True): 'accuracy_pass_consistency_pass',
    (True, False): 'accuracy_pass_consistency_fail',
    (False, True): 'accuracy_fail_consistency_pass',
    (False, False): 'accuracy_fail_consistency_fail',
}
METRICS = (
    'pa_accuracy', 'pa_accuracy_pass', 'pa_consistency', 'coverage',
    'coverage_reward', 'PUZZLE_ACCURACY', 'CELL_ACCURACY',
    'BASE_sat_full_GT', 'BASE_n_steps_total', 'BASE_n_steps_parsed_ok',
    'BASE_n_steps_valid', 'BASE_n_steps_novel_inc_clues',
    'BASE_n_non_valid_contradiction', 'missed_data', 'score',
)


def gate(value):
    """Missing/invalid flags stay unclassified, rather than becoming failures."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        if value.strip().lower() in ('true', '1', '1.0'):
            return True
        if value.strip().lower() in ('false', '0', '0.0'):
            return False
    return None


def input_records(path):
    """Yield (source label, decoded record, parsing error, original input)."""
    with path.open(encoding='utf-8-sig') as source:
        if path.suffix.lower() == '.json':
            try:
                data = json.load(source)
            except (ValueError, RecursionError) as exc:
                yield 'JSON document', None, str(exc), None
                return
            records = data if isinstance(data, list) else [data]
            for index, record in enumerate(records, 1):
                yield f'JSON record {index}', record, None, None
        else:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    yield f'line {line_number}', json.loads(line), None, None
                except (ValueError, RecursionError) as exc:
                    yield f'line {line_number}', None, str(exc), line.rstrip('\n')


def display_value(value):
    return json.dumps(value, ensure_ascii=False)


def render(record, source, group, reason=None):
    """Keep each reasoning entry on its own numbered display line."""
    metrics = record.get('final_result')
    metrics = metrics if isinstance(metrics, dict) else {}
    payload = record.get('payload')
    payload = payload if isinstance(payload, dict) else {}
    lines = ['=' * 80, f'GROUP: {group}', f'SOURCE: {source}',
             f'PID: {record.get("pid", "<missing>")}',
             f'EPOCH: {metrics.get("epoch", "<missing>")}']
    if reason:
        lines.append(f'CLASSIFICATION NOTE: {reason}')
    lines.append('METRICS:')
    for key in METRICS:
        lines.append(f'  {key}: {display_value(metrics.get(key))}')
    lines.append(f'FORMAT_CHECK: {display_value(record.get("Format_Check"))}')
    lines.append(f'REWARD: {display_value(record.get("reward"))}')
    lines.append('PUZZLE TEXT:')
    puzzle = record.get('puzzle_text', '<missing>')
    lines.extend(str(puzzle).splitlines() if isinstance(puzzle, str)
                 else [display_value(puzzle)])
    for label, table in [
        ('GROUND TRUTH', record.get('ground_truth', payload.get('ground_truth'))),
        ('FINAL SOLUTION', record.get('original_prediction', payload.get('solution'))),
    ]:
        lines.append(label + ':')
        if isinstance(table, dict):
            lines.append('  HEADER: ' + display_value(table.get('header')))
            rows = table.get('rows')
            if isinstance(rows, list):
                for i, row in enumerate(rows, 1):
                    lines.append(f'  ROW {i}: {display_value(row)}')
            else:
                lines.append('  ROWS: ' + display_value(rows))
        else:
            lines.append('  ' + display_value(table))
    lines.append('ATTRIBUTE VALUES: ' + display_value(payload.get('attribute_values')))
    for label, entries in [('SYNTACTIC CLUES', payload.get('syntactic_clues')),
                           ('REASONING', payload.get('reasoning', record.get('reasoning')))]:
        lines.append(label + ':')
        if isinstance(entries, list):
            for i, entry in enumerate(entries, 1):
                # JSON quoting preserves newlines inside one numbered entry.
                lines.append(f'  [{i:03d}] {display_value(entry)}')
        elif isinstance(entries, dict):
            for key, value in entries.items():
                lines.append(f'  {key}: {display_value(value)}')
        else:
            lines.append('  ' + display_value(entries))
    # Preserve all diagnostics, including unfamiliar fields, in readable output.
    lines.append('ADDITIONAL RECORD FIELDS:')
    shown = {'pid', 'puzzle_text', 'ground_truth', 'original_prediction',
             'Format_Check', 'reward', 'reasoning'}
    for key, value in record.items():
        if key not in shown:
            lines.append(f'  {key}: {display_value(value)}')
    return '\n'.join(lines) + '\n\n'


def process_file(input_path, output_root, quiet=False):
    # Use the input filename (without extension) as the job folder name.
    folder = output_root / input_path.stem
    if folder.exists():
        folder = output_root / (
            input_path.stem + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        )
    folder.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    names = list(GROUPS.values()) + ['unclassified']
    with ExitStack() as stack:
        outputs = {name: (
            stack.enter_context((folder / (name + '.txt')).open('w', encoding='utf-8')),
            stack.enter_context((folder / (name + '.jsonl')).open('w', encoding='utf-8')),
        ) for name in names}
        errors = stack.enter_context((folder / 'input_errors.jsonl').open('w', encoding='utf-8'))
        for location, record, error, raw in input_records(input_path):
            if error or not isinstance(record, dict):
                counts['input_errors'] += 1
                issue = dict(source=location, error=error or 'Record is not a JSON object',
                             raw=raw, decoded_record=record)
                errors.write(json.dumps(issue, ensure_ascii=False) + '\n')
                print(f'WARNING: {location}: {issue["error"]}', file=sys.stderr)
                continue
            metrics = record.get('final_result')
            metrics = metrics if isinstance(metrics, dict) else {}
            flags = (gate(metrics.get('pa_accuracy_pass')),
                     gate(metrics.get('pa_consistency')))
            name = GROUPS.get(flags, 'unclassified')
            reason = ('Missing or non-boolean gate value; no failure inferred'
                      if name == 'unclassified' else None)
            counts[name] += 1
            text = render(record, location, name, reason)
            readable, machine = outputs[name]
            readable.write(text)
            machine.write(json.dumps(record, ensure_ascii=False) + '\n')
            if not quiet:
                print(text, end='')
    summary = ['SUMMARY', f'Input: {input_path.resolve()}',
               f'Output directory: {folder.resolve()}']
    summary.extend(f'{name}: {counts[name]}' for name in names)
    summary.extend([f'input_errors: {counts["input_errors"]}',
                    f'Total valid objects: {sum(counts[name] for name in names)}',
                    'Groups use logged gates; failure reasons are not recomputed.'])
    summary_text = '\n'.join(summary) + '\n'
    (folder / 'summary.txt').write_text(summary_text, encoding='utf-8')
    print(summary_text, end='')
    return folder


def main(fname='./Input_Logs/', output_dir='./Output/'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'input', type=Path, nargs='?', default=Path(fname),
        help='Input feedback file or folder; default: ./Input_Logs/',
    )
    parser.add_argument(
        '--output-dir', type=Path, default=Path(output_dir),
        help='Parent folder for job outputs; default: ./Output/',
    )
    parser.add_argument('--quiet', action='store_true',
                        help='Save every trace but print only summaries')
    args = parser.parse_args()
    if args.input.is_dir():
        files = sorted(p for p in args.input.iterdir()
                       if p.is_file() and p.suffix.lower() in ('.jsonl', '.json'))
        if not files:
            parser.error(f'No .jsonl or .json files found in {args.input}')
    elif args.input.is_file():
        files = [args.input]
    else:
        parser.error(f'Input file or folder not found: {args.input}')
    for input_path in files:
        process_file(input_path, args.output_dir, quiet=args.quiet)
    print(f'Finished processing {len(files)} input file(s).')
    return 0


if __name__ == '__main__':
    # Process every .jsonl/.json file in this folder.
    # To process only one file, set fname to its complete relative path.
    fname = './Input_Logs/'
    output_dir = './Output/'
    raise SystemExit(main(fname=fname, output_dir=output_dir))
