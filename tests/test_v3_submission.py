"""Gate must reject stale/invalid validation before submitting any benchmark."""
import copy
import pytest
from data_processing.common import atomic_json, digest, load_config, run_identity, write_manifest, read_manifest
from slurm import submit_v3_benchmarks as launch


def validation_fixture(tmp_path):
    import yaml
    config = load_config('configs/reproduce_v3.yaml')
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    plan = {'root': str(tmp_path), 'config': str(path), 'config_hash': digest(config), 'source_hash': 'source'}
    sample = {'sample_id': 'cptac:1'}
    write_manifest(tmp_path / 'cptac/smoke.json', [sample], dataset='cptac', purpose='smoke')
    manifest = read_manifest(tmp_path / 'cptac/smoke.json')
    identity = run_identity(config, 'source', manifest['samples_hash'])
    run_hash = digest(identity)
    base = tmp_path / 'cptac/validation'
    atomic_json(base / 'run_lock.json', {**identity, 'run_hash': run_hash})
    fleet = {'gpu': 'NVIDIA RTX PRO 6000 Blackwell Server Edition', 'gpu_memory_bytes': 96 * 1024**3}
    atomic_json(tmp_path / 'gpu_fleet.json', fleet)
    profile = dict(passed=True, real_gpu=True, smoke=True, dataset='cptac', code_hash='source',
        config_hash=digest(config), manifest_hash=manifest['samples_hash'], run_hash=run_hash, slurm_job_id='123', **fleet)
    atomic_json(base / 'profile.json', profile)
    result = dict(sample, status='ok', run_hash=run_hash)
    atomic_json(base / 'results' / (digest(sample['sample_id']) + '.json'), result)
    return plan, base, profile, result


@pytest.mark.parametrize('change', ['none', 'invalid', 'wrong_job', 'wrong_gpu', 'wrong_manifest'])
def test_validation_identity_and_results(tmp_path, change):
    plan, base, profile, result = validation_fixture(tmp_path)
    if change == 'invalid':
        result['status'] = 'invalid'
        atomic_json(base / 'results' / (digest(result['sample_id']) + '.json'), result)
    if change == 'wrong_job': profile['slurm_job_id'] = '124'
    if change == 'wrong_gpu': profile['gpu'] = 'A100'
    if change == 'wrong_manifest': profile['manifest_hash'] = 'other'
    atomic_json(base / 'profile.json', profile)
    if change == 'none':
        assert launch.verify_validation(plan, 'cptac', '123') == profile
    else:
        with pytest.raises(RuntimeError): launch.verify_validation(plan, 'cptac', '123')


def test_gate_never_submits_if_either_validation_fails(tmp_path, monkeypatch):
    atomic_json(tmp_path / 'plan.json', {'root': str(tmp_path), 'source_hash': 'source', 'files': {}})
    for d in launch.DATASETS:
        atomic_json(tmp_path / 'receipts' / (d + '-validation.json'), {'job_id': '123'})
    monkeypatch.setattr(launch, 'code_hash', lambda: 'source')
    monkeypatch.setattr(launch, 'completed_successfully', lambda jobs: None)
    def verify(plan, dataset, job):
        if dataset == 'cptac': raise RuntimeError('invalid CPTAC result')
        return {'passed': True}
    monkeypatch.setattr(launch, 'verify_validation', verify)
    calls = []
    monkeypatch.setattr(launch, 'submit_once', lambda *args: calls.append(args))
    with pytest.raises(RuntimeError, match='invalid CPTAC'):
        launch.gate(tmp_path)
    assert not calls
    assert not launch.read_json(tmp_path / 'gate_decision.json')['passed']
