import copy
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from data_processing.common import digest, load_config, read_manifest, write_manifest
from data_processing.datasets import answer_label, make_sample, normalize_choices, public_input
from data_processing.regions import ImageSource, Region, whole_roi, wsi_region, zoom_regions
from data_processing.sharding import make_shards
from eval.metrics import score
from models.agent import retrieve, run_agent
from models.inference import parse_json_object
from scripts.select_gpu import simulated_finish, choose_pair


def sample(sid='one', image='slide'):
    return make_sample('test', 'test', sid, image, '/unused', 'Which?', {'A': 'yes', 'B': 'no'}, 'A')


class FakeBackend:
    def __init__(self, zoom=False, sufficient=False):
        self.zoom, self.sufficient = zoom, sufficient
        self.queries, self.described, self.final = [], [], None
    def encode_text(self, text):
        self.queries.append(text)
        return [1., 0.]
    def encode_regions(self, regions):
        self.candidates = regions
        return [[1., 0.] for _ in regions]
    def describe(self, region, question, missing_info=None):
        self.described.append((region.region_id, question, missing_info))
        return 'observed'
    def evaluate(self, findings, row, kind, allowed):
        assert 'ground_truth' not in row and 'ground_truth_label' not in row
        return {'answer': 'A', 'thinking_steps': 'evidence', 'sufficient': 'Yes' if self.sufficient else 'No',
                'missing_info': 'mitoses', 'zoom_recommendation': 'Yes' if self.zoom else 'No', 'zoom_level': 10}
    def conclude(self, evidence, trajectory, row):
        self.final = (copy.deepcopy(evidence), copy.deepcopy(trajectory))
        return {'answer': 'A', 'explanation': 'all evidence'}


def inputs(n):
    regions = [Region(f'{i:04}', '/unused', i*4096, 0, 4096, 4096, 512, 512, 'physical', 5, .25)
               for i in range(n)]
    return regions, {r.region_id: 'generic' for r in regions}, {r.region_id: [1., 0.] for r in regions}


def test_ceil_full_denominator_unique_and_all_evidence():
    backend = FakeBackend()
    regions, descriptions, features = inputs(101)
    original = copy.deepcopy(descriptions)
    result = run_agent(sample(), regions, descriptions, features, backend, load_config())
    assert [len(e['findings']) for e in result['process']] == [11, 6, 6, 6, 6]
    assert result['total_examined_regions'] == 35
    assert len({f['region']['region_id'] for f in backend.final[0]}) == 35
    assert len(backend.described) == 25
    assert len(backend.final[1]) == 5
    assert result['stop_reason'] == 'max_iterations'
    assert 'next_regions' not in result['process'][-1]
    assert descriptions == original
    assert backend.queries == ['Which?'] + ['mitoses'] * 4


def test_zoom_all_current_regions_uses_missing_info_and_stops():
    regions, descriptions, features = inputs(101)
    backend = FakeBackend(zoom=True)
    result = run_agent(sample(), regions, descriptions, features, backend, load_config())
    assert len(backend.candidates) == 11 * 4
    assert len(result['process']) == 1
    assert len(backend.final[0]) == 12
    assert backend.queries == ['Which?', 'mitoses']
    assert backend.described[-1][2] == 'mitoses'
    assert result['stop_reason'] == 'zoom_completed'


def test_tiny_n_and_cases_are_isolated():
    regions, descriptions, features = inputs(1)
    first = run_agent(sample('1'), regions, descriptions, features, FakeBackend(zoom=True), load_config())
    backend = FakeBackend(sufficient=True)
    second = run_agent(sample('2'), regions, descriptions, features, backend, load_config())
    assert first['process'][0]['zoom_finding']['region']['scale'] == 10
    assert second['process'][0]['findings'][0]['region']['scale'] == 5
    assert backend.final[0][0]['description'] == 'generic\nobserved'


def test_strict_labels_and_invalid_denominator():
    choices = normalize_choices(['A) yes', 'B) no', ''])
    assert choices == {'A': 'yes', 'B': 'no'}
    assert answer_label('A) yes', choices) == 'A'
    assert answer_label('NO', choices) == 'B'
    for value in ['', 'zzz', 'A) no', 'Answer: A', 'yes maybe', None]:
        assert answer_label(value, choices) is None
    rows = [sample(str(i)) for i in range(4)]
    results = {rows[0]['sample_id']: {'status': 'ok', 'pred_answer': 'A'},
               rows[1]['sample_id']: {'status': 'ok', 'pred_answer': ''},
               rows[2]['sample_id']: {'status': 'invalid', 'pred_answer': 'A'}}
    metrics = score(rows, results)
    assert metrics['closed']['closed_overall']['accuracy_percent'] == 25
    assert metrics['missing'] == 1 and metrics['invalid'] == 1
    assert not metrics['complete']


def test_native_pixel_zoom_and_mpp_provenance(tmp_path):
    arr = np.zeros((40, 80, 3), dtype=np.uint8)
    arr[:,40:] = [255, 0, 0]
    path = tmp_path / 'test.png'
    Image.fromarray(arr).save(path)
    with ImageSource(path, .5) as source:
        base = wsi_region(source, 0, 0, size=40, magnification=5)
        assert base.output_width == 10 and base.magnification_source == 'assumed'
        children = zoom_regions(base, 20)
        assert len(children) == 16
        assert all(r.width == 10 and r.output_width == 10 for r in children)
        roi = whole_roi(source)
        crops = zoom_regions(roi, 2)
        assert crops[1].x == 40 and crops[1].width == 40
        assert np.asarray(source.read(crops[1]))[0, 0].tolist() == [255, 0, 0]
        assert crops[1].scale_kind == 'relative'


def test_retrieval_ties_stable_and_zero_features_rejected():
    assert retrieve(['b', 'a'], [[1,0], [1,0]], [1,0], 1) == ['a']
    with pytest.raises(ValueError):
        retrieve(['a'], [[0,0]], [1,0], 1)


def test_sharding_indivisible_nonempty_deterministic():
    rows = [sample(str(i) + '-' + str(j), str(i)) for i in range(37) for j in range(i % 7 + 1)]
    shards, totals = make_shards(rows)
    reversed_shards, _ = make_shards(list(reversed(rows)))
    assert shards == reversed_shards
    assert len(shards) == 16 and all(shards)
    assert sum(map(len, shards)) == len(rows)
    owner = {}
    for i, shard in enumerate(shards):
        for row in shard:
            assert owner.setdefault(row['image_id'], i) == i
    with pytest.raises(ValueError):
        make_shards(rows[:2])


def test_manifest_tamper_detected(tmp_path):
    path = tmp_path / 'manifest.json'
    write_manifest(path, [sample()])
    assert read_manifest(path)['samples_hash'] == digest([sample()])
    path.write_text(path.read_text().replace('Which?', 'tampered'))
    with pytest.raises(ValueError):
        read_manifest(path)


def test_json_nested_quoted_braces_and_no_gold():
    assert parse_json_object('```json\n{"answer":"a } b", "nested":{"x":1}}\n```')['nested'] == {'x': 1}
    assert parse_json_object('invalid') is None
    assert not {'ground_truth', 'ground_truth_label'} & set(public_input(sample()))


def test_makespan_accounts_for_pool_competition():
    a = {'feature':'a', 'speed':1, 'slots':[0], 'queue_seconds':0}
    b = {'feature':'b', 'speed':1, 'slots':[0], 'queue_seconds':5}
    selected = choose_pair({'candidates':[a,b]}, {'candidates':[a,b]}, [10], [10])
    assert selected['pathmmu']['feature'] != selected['bcnb']['feature']
    assert selected['estimated_global_finish_seconds'] == 15
    assert simulated_finish([10]*16, [0]*4, 5) == 45
    with pytest.raises(ValueError):
        simulated_finish([10], [0], None)


def test_discovery_uses_real_config_and_forecast_for_busy_pool(monkeypatch):
    import datetime as dt
    import scripts.select_gpu as selector
    calls = []
    start = (dt.datetime.now() + dt.timedelta(hours=2)).strftime('%Y-%m-%dT%H:%M:%S')
    def command(args):
        calls.append(args)
        if args[:3] == ['scontrol','show','nodes']:
            return 0, ('NodeName=n1 Partitions=gpu ActiveFeatures=b200 State=MIXED Gres=gpu:b200:8 '
                       'CPUTot=120 CPUAlloc=120 RealMemory=2000000 AllocMem=1000000 AllocTRES=gres/gpu=8')
        if args[0] == 'sbatch':
            return 0, 'sbatch: Job 123 to start at ' + start
        return 0, ''
    monkeypatch.setattr(selector, 'command', command)
    report = selector.discover(load_config(), [100])
    assert report['candidates'][0]['feature'] == 'b200'
    assert report['candidates'][0]['queue_seconds'] > 7000
    assert len(report['candidates'][0]['slots']) == 1
    sbatch = next(c for c in calls if c[0] == 'sbatch')
    assert '--test-only' in sbatch and '--cpus-per-task=8' in sbatch
    assert '--account=lia-lab-members' in sbatch


def test_reproduction_config_rejects_silent_algorithm_change(tmp_path):
    import yaml
    cfg = load_config()
    cfg['algorithm']['initial_ratio'] = .2
    path = tmp_path / 'wrong.yaml'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        load_config(path)
