"""Fail-closed regression tests for unattended, resource-consuming submissions."""
import copy
from types import SimpleNamespace

import pytest
import yaml

from data_processing.common import atomic_json, digest, load_config, read_json, run_identity, write_manifest
from slurm import submit_bcnb_comparison as launcher


@pytest.fixture
def validation_run(tmp_path):
    samples = [{'sample_id': 'bcnb-1'}]
    write_manifest(tmp_path / 'smoke.json', samples, dataset='bcnb', purpose='smoke', images_verified=True)
    fleet = {'gpu': 'NVIDIA RTX PRO 6000 Blackwell Server Edition', 'gpu_memory_bytes': 96 * 1024 ** 3}
    atomic_json(tmp_path / 'gpu_fleet.json', fleet)
    plan = {'root': str(tmp_path), 'source_hash': 'source', 'files': {}, 'pipelines': {}}
    for name, job in [('legacy', '101'), ('paper', '102')]:
        cfg = load_config('configs/reproduce_v2.yaml' if name == 'paper' else 'configs/reproduce_v1.yaml')
        path = tmp_path / (name + '.yaml')
        path.write_text(yaml.safe_dump(cfg))
        plan['pipelines'][name] = {'config': str(path), 'config_hash': digest(cfg), 'full_minutes': 210}
        identity = run_identity(cfg, 'source', digest(samples))
        run_hash = digest(identity)
        output = tmp_path / name / 'validation'
        atomic_json(output / 'run_lock.json', {**identity, 'run_hash': run_hash})
        atomic_json(output / 'profile.json', {'passed': True, 'real_gpu': True, 'smoke': True,
            'dataset': 'bcnb', 'slurm_job_id': job, 'code_hash': 'source', 'config_hash': digest(cfg),
            'manifest_hash': digest(samples), 'run_hash': run_hash, **fleet})
        atomic_json(output / 'results' / (digest('bcnb-1') + '.json'),
                    {'sample_id': 'bcnb-1', 'status': 'ok', 'run_hash': run_hash})
        atomic_json(tmp_path / 'receipts' / (name + '-validation.json'), {'job_id': job})
    atomic_json(tmp_path / 'plan.json', plan)
    return tmp_path, plan


@pytest.mark.parametrize('failure', ['missing', 'invalid', 'stale_job', 'different_gpu', 'false_pass', 'scheduler'])
def test_either_failure_blocks_every_full_submission(validation_run, monkeypatch, failure):
    root, plan = validation_run
    output = root / 'paper' / 'validation'
    profile = read_json(output / 'profile.json')
    result_path = output / 'results' / (digest('bcnb-1') + '.json')
    if failure == 'missing': result_path.unlink()
    if failure == 'invalid':
        result = read_json(result_path)
        result['status'] = 'invalid'
        atomic_json(result_path, result)
    if failure == 'stale_job': profile['slurm_job_id'] = '999'
    if failure == 'different_gpu': profile['gpu'] = 'NVIDIA A100'
    if failure == 'false_pass': profile['passed'] = False
    atomic_json(output / 'profile.json', profile)
    monkeypatch.setattr(launcher, 'code_hash', lambda: 'source')
    def scheduler(jobs):
        if failure == 'scheduler': raise RuntimeError('Nonzero exit')
    monkeypatch.setattr(launcher, 'completed_successfully', scheduler)
    submitted = []
    monkeypatch.setattr(launcher, 'submit_once', lambda *args: submitted.append(args))
    with pytest.raises((RuntimeError, FileNotFoundError)):
        launcher.gate(root)
    assert submitted == []
    assert read_json(root / 'gate_decision.json')['passed'] is False


def test_both_pass_submit_two_unthrottled_arrays_and_dependent_scores(validation_run, monkeypatch):
    root, plan = validation_run
    monkeypatch.setattr(launcher, 'code_hash', lambda: 'source')
    monkeypatch.setattr(launcher, 'completed_successfully', lambda jobs: None)
    submissions = []
    def submit(root, name, command):
        submissions.append((name, command))
        return str(200 + len(submissions))
    monkeypatch.setattr(launcher, 'submit_once', submit)
    launcher.gate(root)
    assert [name for name, _ in submissions] == ['legacy-full', 'legacy-score', 'paper-full', 'paper-score']
    for name, command in submissions:
        assert '--account=cs6501-cbx8wm' in command and '--qos=class' in command
        if name.endswith('-full'):
            assert '--array=0-15' in command
            assert '--gres=gpu:rtx_pro_6000:1' in command
            assert '--constraint=rtxpro6000' in command
        else:
            assert '--dependency=afterok:' + ('201' if name.startswith('legacy') else '203') in command
    assert read_json(root / 'full_submitted.json')['both_arrays_submitted']


def test_scheduler_failure_does_not_accept_success_profile(monkeypatch):
    monkeypatch.setattr(launcher.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *a, **k:
                        SimpleNamespace(stdout='101|COMPLETED|0:0\n102|FAILED|1:0\n'))
    with pytest.raises(RuntimeError, match='did not both complete'):
        launcher.completed_successfully(['101', '102'])


def test_submission_receipt_prevents_duplicates(tmp_path, monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout='12345\n')
    monkeypatch.setattr(launcher.subprocess, 'run', run)
    for _ in range(2):
        assert launcher.submit_once(tmp_path, 'legacy-full', ['sbatch', 'run.sh']) == '12345'
    assert len(calls) == 1
