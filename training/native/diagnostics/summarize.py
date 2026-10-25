"""离线整理原始诊断 CSV；不参与原生训练进程。仅使用 Python 标准库。"""

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


def distribution(values):
    if not values:
        return None
    values.sort()
    n = len(values)
    total = sum(values)
    return dict(count=n, sum=total, mean=total / n, p50=values[int((n - 1) * .5)],
                p99=values[int((n - 1) * .99)], max=values[-1])


def summarize(root):
    results = []
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (run / 'window.json').exists():
            results.append(dict(name=run.name, failed=True))
            continue
        w = json.loads((run / 'window.json').read_text(encoding='utf-8-sig'))
        b = json.loads((run / 'warmup.json').read_text(encoding='utf-8-sig')) if (run / 'warmup.json').exists() else {}
        groups = defaultdict(lambda: defaultdict(list))
        games = defaultdict(lambda: defaultdict(float))
        inside = False
        with (run / 'timing.csv').open() as file:
            for r in csv.DictReader(file):
                kind = r['kind']
                if kind == 'window_begin':
                    inside = True
                    continue
                if kind == 'window_end':
                    inside = False
                    continue
                if not inside and kind != 'cuda_memory':
                    continue
                for key in 'abcdef':
                    groups[kind][key].append(float(r[key]))
                if kind in ('rules', 'policy', 'lane_queue'):
                    games[r['id']][kind + '_ms'] += float(r['a'])
                elif kind == 'root':
                    games[r['id']]['root_queue_ms'] += float(r['a'])
                    games[r['id']]['root_flight_ms'] += float(r['b'])
        d = {kind: {key: distribution(v) for key, v in fields.items()} for kind, fields in groups.items()}
        seconds = w['seconds'] - b.get('seconds', 0)
        delta = {key: w.get(key, 0) - b.get(key, 0) for key in ('inference_nodes', 'commands', 'consumed', 'updates')}
        with (root / (run.name + '.gpu.csv')).open(encoding='utf-8-sig') as file:
            gpu_rows = list(csv.reader(file))
        # 采样进程退出可能留下半行；保留原始 CSV，并显式计数，不把半行当作零读数。
        gpu = [r for r in gpu_rows if len(r) == 6]
        result = dict(name=run.name, seconds=seconds, decisions=delta['inference_nodes'],
                      decisions_s=delta['inference_nodes'] / seconds, commands=delta['commands'],
                      commands_s=delta['commands'] / seconds, consumed=delta['consumed'],
                      consumed_s=delta['consumed'] / seconds, updates=delta['updates'],
                      ready=w.get('ready_steps'), actor_version=w.get('actor_version'),
                      gpu_peak_mib=max(float(r[1]) for r in gpu),
                      gpu_incomplete_rows=len(gpu_rows) - len(gpu),
                      gpu_sm_mhz=distribution([float(r[2]) for r in gpu]), per_game=games, distributions=d,
                      raw_sha256=hashlib.sha256((run / 'timing.csv').read_bytes()).hexdigest())
        if 'root_forwards' in w:
            result.update(root_forwards=w['root_forwards'], inference_batch=w['batch'],
                          root_forwards_s=w['root_forwards_per_second'])
        if 'fixture_tokens' in w:
            result.update(concurrency=w['concurrency'], root_slots=w['root_slots'],
                          fixture_tokens=w['fixture_tokens'], tokens=w['tokens'],
                          token_capacity=w['token_capacity'], batches=w['batches'],
                          graph_replays=w['graph_replays'], graph_fallbacks=w['graph_fallbacks'])
        if 'consumed_within_lag' in w:
            result.update(lag_limit=w['lag_limit'],
                          consumed_within_lag_s=(w['consumed_within_lag'] - b.get('consumed_within_lag', 0)) / seconds,
                          consumed_lag_max=w['consumed_lag_max'],
                          outside_lag=delta['consumed'] - (w['consumed_within_lag'] - b.get('consumed_within_lag', 0)),
                          strict_lag=w.get('strict_lag', False),
                          lag_drains=w.get('lag_drains', 0) - b.get('lag_drains', 0),
                          admission_outstanding=w.get('admission_outstanding'),
                          peak_ready=w.get('peak_ready_steps'),
                          pending=w.get('pending_steps'), learning_steps=w.get('learning_steps'))
        for kind in ('rules', 'policy'):
            result[kind + '_us_decision'] = 1000 * d[kind]['a']['sum'] / delta['inference_nodes'] if delta['inference_nodes'] else None
        gpu_conversions = d.get('learn_gpu', {}).get('a', {}).get('sum', 0)
        result['learner_gpu_sampled_conversions'] = gpu_conversions
        result['learner_gpu_ms_conversion'] = d['learn_gpu']['c']['sum'] / gpu_conversions if gpu_conversions else None
        results.append(result)
        print(run.name, {k: result[k] for k in ('seconds', 'decisions_s', 'commands_s', 'consumed_s', 'gpu_peak_mib', 'learner_gpu_ms_conversion')}, flush=True)
    (root / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    summarize(Path(sys.argv[1]))
