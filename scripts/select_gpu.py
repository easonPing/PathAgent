"""Live Slurm discovery and explicitly estimated queue + workload makespan."""
import argparse
import datetime as dt
import heapq
import itertools
import math
import re
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.common import atomic_json, load_config, read_json, resolve

# Runtime ratios are engineering priors, never benchmark measurements.
GPU_PRIORS = {
    'a100_80gb': {'gres': 'a100', 'memory_gb': 80, 'speed': 1.0},
    'a100_40gb': {'gres': 'a100', 'memory_gb': 40, 'speed': .95},
    'a6000': {'gres': 'a6000', 'memory_gb': 48, 'speed': .45},
    'a40': {'gres': 'a40', 'memory_gb': 48, 'speed': .45},
    'rtxpro6000': {'gres': 'rtx_pro_6000', 'memory_gb': 96, 'speed': 1.4},
    'b200': {'gres': 'b200', 'memory_gb': 180, 'speed': 2.0},
}


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=90)
    return result.returncode, result.stdout + result.stderr


def fields(line):
    return dict(re.findall(r'(\w+)=(\S+)', line))


def seconds_until(value, now):
    try:
        stamp = dt.datetime.fromisoformat(value).astimezone()
        return max(0.0, (stamp - now).total_seconds())
    except (ValueError, TypeError):
        return None


def simulated_finish(durations, slots, queue_seconds, concurrency=16):
    if not slots or queue_seconds is None:
        raise ValueError('Queue/capacity unknown; cannot treat as zero')
    heap = [max(queue_seconds, s) for s in sorted(slots)[:concurrency]]
    heapq.heapify(heap)
    for duration in sorted(durations, reverse=True):
        heapq.heappush(heap, heapq.heappop(heap) + duration)
    return max(heap)


def discover(config, durations, minimum_memory_gb=40):
    now = dt.datetime.now().astimezone()
    outputs = {}
    for key, args in {
        'nodes': ['scontrol', 'show', 'nodes', '-o'],
        'jobs': ['scontrol', 'show', 'jobs', '-o'],
        'partitions': ['scontrol', 'show', 'partition', 'gpu', '-o'],
        'account': ['sacctmgr', '-n', '-P', 'show', 'assoc', 'user=' + __import__('getpass').getuser(),
                    'account=' + config['slurm']['account'], 'format=Account,QOS,GrpTRES,MaxTRES,MaxJobs'],
        'qos': ['sacctmgr', '-n', '-P', 'show', 'qos', 'normal',
                'format=Name,GrpTRES,MaxTRESPU,MaxJobsPU,MaxSubmitPU'],
    }.items():
        code, output = command(args)
        outputs[key] = output
        if code:
            raise RuntimeError(f'Cannot inspect Slurm {key}: {output}')
    nodes = [fields(line) for line in outputs['nodes'].splitlines() if 'NodeName=' in line]
    jobs = [fields(line) for line in outputs['jobs'].splitlines() if 'JobId=' in line]
    # Slurm end times are allocation limits, providing conservative capacity release estimates.
    release = {}
    expanded_cache = {}
    for job in jobs:
        if job.get('JobState') not in {'RUNNING', 'COMPLETING'}:
            continue
        if 'gpu' not in job.get('AllocTRES', '') and 'gpu' not in job.get('TresPerNode', ''):
            continue
        end = seconds_until(job.get('EndTime'), now)
        if end is None:
            continue
        node_list = job.get('NodeList', '')
        if node_list not in expanded_cache:
            expanded_cache[node_list] = command(['scontrol', 'show', 'hostnames', node_list])
        code, expanded = expanded_cache[node_list]
        if code == 0:
            for node in expanded.splitlines():
                release[node] = max(end, release.get(node, 0))
    candidates, rejected = [], []
    for feature, prior in GPU_PRIORS.items():
        if prior['memory_gb'] < minimum_memory_gb:
            rejected.append({'feature': feature, 'reason': 'insufficient memory with required headroom'})
            continue
        slots, selected_nodes = [], []
        for node in nodes:
            if 'gpu' not in node.get('Partitions', '').split(',') or feature not in node.get('ActiveFeatures', '').split(','):
                continue
            if any(state in node.get('State', '') for state in ['DOWN', 'DRAIN', 'FAIL', 'MAINT', 'NOT_RESPONDING']):
                continue
            match = re.search(r'gpu:' + prior['gres'] + r':(\d+)', node.get('Gres', ''))
            if not match:
                continue
            gpus = int(match.group(1))
            used_match = re.search(r'(?:^|,)gres/gpu=(\d+)', node.get('AllocTRES', ''))
            used = int(used_match.group(1)) if used_match else 0
            cpu, memory = config['slurm']['cpus_per_task'], config['slurm']['memory_gb'] * 1024
            total = min(gpus, int(node['CPUTot']) // cpu, int(node['RealMemory']) // memory)
            free = min(max(0, gpus - used), max(0, int(node['CPUTot']) - int(node['CPUAlloc'])) // cpu,
                       max(0, int(node['RealMemory']) - int(node['AllocMem'])) // memory)
            slots.extend([0.0] * free)
            if node['NodeName'] in release:
                slots.extend([release[node['NodeName']]] * max(0, total - free))
            selected_nodes.append(node['NodeName'])
        if not selected_nodes:
            rejected.append({'feature': feature, 'reason': 'no eligible nodes'})
            continue
        runtime = max(durations) / prior['speed']
        minutes = math.ceil((runtime * config['slurm']['walltime_factor'] +
                             60 * config['slurm']['walltime_overhead_minutes']) / 60)
        if minutes > 3 * 24 * 60:
            rejected.append({'feature': feature, 'reason': 'estimated shard runtime exceeds GPU partition 3-day walltime'})
            continue
        args = ['sbatch', '--test-only', '--account=' + config['slurm']['account'], '--partition=gpu',
                '--gres=gpu:' + prior['gres'] + ':1', '--constraint=' + feature,
                '--cpus-per-task=' + str(config['slurm']['cpus_per_task']), '--mem=' + str(config['slurm']['memory_gb']) + 'G',
                '--time=' + str(minutes), '--wrap=true']
        code, preview = command(args)
        match = re.search(r'start at (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)', preview)
        queue = seconds_until(match.group(1), now) if match else None
        if code or queue is None:
            rejected.append({'feature': feature, 'reason': 'scheduler start unavailable or account rejected', 'preview': preview})
            continue
        capacity_source = 'node free resources and running-job walltime releases'
        if not slots:
            # A scheduler forecast establishes one possible future slot, not an idle pool.
            slots = [queue]
            capacity_source = 'one conservative future slot from scheduler forecast; node release unknown'
        adjusted = [d / prior['speed'] for d in durations]
        candidates.append({**prior, 'feature': feature, 'queue_seconds': queue, 'slots': slots,
                           'estimated_finish_seconds': simulated_finish(adjusted, slots, queue),
                           'walltime_minutes': minutes, 'nodes': selected_nodes, 'preview': preview,
                           'capacity_source': capacity_source,
                           'runtime_source': 'A100-80GB normalized estimate with hardware speed priors'})
    candidates.sort(key=lambda c: (c['estimated_finish_seconds'], c['feature']))
    return {'observed_at': now.isoformat(), 'candidates': candidates, 'rejected': rejected,
            'account': config['slurm']['account'], 'snapshot': outputs,
            'limitations': 'Estimated start times are not reservations. Node release estimates use running-job time limits; future competition and early completion can change makespan.'}


def choose_pair(first, second, costs_first, costs_second):
    choices = []
    for a, b in itertools.product(first['candidates'], second['candidates']):
        da = [v / a['speed'] for v in costs_first]
        db = [v / b['speed'] for v in costs_second]
        if a['feature'] == b['feature']:
            # Both arrays compete for the same pool; never count its slots twice.
            finish = simulated_finish(da + db, a['slots'], max(a['queue_seconds'], b['queue_seconds']), 32)
        else:
            finish = max(simulated_finish(da, a['slots'], a['queue_seconds']),
                         simulated_finish(db, b['slots'], b['queue_seconds']))
        choices.append((finish, a['feature'], b['feature'], a, b))
    if not choices:
        raise RuntimeError('No eligible GPU pair with scheduler-supported start estimates')
    selected = min(choices, key=lambda row: row[:3])
    return {'estimated_global_finish_seconds': selected[0], 'pathmmu': selected[3], 'bcnb': selected[4]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, required=True, help='A100-80GB normalized per-shard seconds')
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--output', required=True)
    parser.add_argument('--minimum-memory-gb', type=float, default=40)
    args = parser.parse_args()
    report = discover(load_config(), [args.seconds] * args.shards, args.minimum_memory_gb)
    atomic_json(resolve(args.output), report)
    if not report['candidates']:
        raise SystemExit('No eligible GPU candidate')
    print({k: v for k, v in report['candidates'][0].items() if k not in {'slots', 'nodes', 'preview'}})


if __name__ == '__main__':
    main()
