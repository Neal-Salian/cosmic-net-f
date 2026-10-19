"""Load exact real inputs and recovered policy; never execute notebook cells."""
import copy
import io
import importlib.metadata
import json
import platform
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from data.provenance import hash_payload, sha256_file
from data.loaders.base_loader import get_loader
from graph.graph_builder import GraphBuilder
from model.model import load_model
from rls.benchmark_scorers import PairScorer
from rls.constraints import PhysicalPairLayout
from rls.notebook_benchmark import as_data
from rls.notebook_workflow import prepared_splits
from rls.policy import build_policy
from rls.provenance import record_backbone
from .core import BASE_COMMIT

REGISTERED_INPUTS = {
    'csv': '6c148f2cae68abac7bc5c34d370ff0130a647b49e79c32880bcf36ecc0ab4d53',
    'backbone': '42764f0db204863b28672d6ec6e207443602de16dcfa926765a0429b3451c63f',
    'recovered_archive': '800c65be6fe3c02c96967c1db4fd4c77c6324d7df19f72898e60151f0659bf40',
}


def validate_registered_inputs(csv,backbone,archive):
    records={name:sha256_file(path) for name,path in
        [('csv',csv),('backbone',backbone),('recovered_archive',archive)]}
    for name,record in records.items():
        if record['sha256']!=REGISTERED_INPUTS[name]:
            raise ValueError(f'Registered input identity differs: {name}')
    return records


def source_record():
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True, timeout=10).strip()
    release = root/'ABLATION_SOURCE.json'
    if not (root/'.git').exists() and release.is_file():
        manifest=json.loads(release.read_text())
        for name,checksum in manifest['files'].items():
            if not (root/name).is_file() or sha256_file(root/name)['sha256']!=checksum:
                raise ValueError(f'Offline source checksum differs: {name}')
        if manifest['base_commit']!=BASE_COMMIT:
            raise ValueError('Offline source has the wrong base commit')
        head=manifest['commit']
    else:
        head = git('rev-parse', 'HEAD')
    # Bundle mode works before this local-only branch is published.
    if (root/'.git').exists() and head != BASE_COMMIT:
        subprocess.run(['git','-C',str(root),'merge-base','--is-ancestor',BASE_COMMIT,head], check=True, timeout=10)
    files = {str(p.relative_to(root)): sha256_file(p)['sha256']
             for folder in ('rls','data','graph','model','experiments/paper_ablations')
             for p in sorted((root/folder).rglob('*.py'))}
    dependencies={}
    for name in ('torch-geometric','numpy','scipy','pandas','networkx','h5py','PyYAML'):
        try: dependencies[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: dependencies[name]=None
    return dict(base_commit=BASE_COMMIT, evaluator_commit=head, source_files=files,
                source_sha256=hash_payload(files), python=platform.python_version(),
                torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                dependencies=dependencies,
                device_available='cuda' if torch.cuda.is_available() else 'cpu')


def load_inputs(csv, backbone, archive, cache, device='cpu'):
    input_records=validate_registered_inputs(csv,backbone,archive)
    with zipfile.ZipFile(archive) as z:
        prefix='recovered_training/'
        manifest=json.loads(z.read(prefix+'training_checkpoint.json'))
        if (manifest.get('artifact_type')!='notebook_policy_training_checkpoint'
                or manifest.get('status')!='policy_training_complete'
                or manifest.get('stage')!='policy_training'):
            raise ValueError('Expected a completed recovered policy-training handoff')
        payload={k:v for k,v in manifest.items() if k!='manifest_sha256'}
        if hash_payload(payload)!=manifest['manifest_sha256']:
            raise ValueError('Recovered manifest checksum changed')
        for name,record in manifest['artifacts'].items():
            import hashlib
            if hashlib.sha256(z.read(prefix+name)).hexdigest()!=record['sha256']:
                raise ValueError(f'Recovered training artifact changed: {name}')
        weights=torch.load(io.BytesIO(z.read(prefix+'policy.pt')),map_location=device,weights_only=True)
    for name,path in [('tng_csv',csv),('checkpoint',backbone)]:
        if sha256_file(path)['sha256']!=manifest['context']['dataset'][name]['sha256']:
            raise ValueError(f'Exact original dataset/checkpoint required: {name}')
    cfg=copy.deepcopy(manifest['config_used'])
    cfg['data']['tng'].update(clustered_file=str(Path(csv).resolve()),cache_dir=str(Path(cache).resolve()))
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    gnn=load_model(str(backbone),cfg,device,research_label='legacy_total_radius').eval().requires_grad_(False)
    if record_backbone('frozen',gnn)!=manifest['context']['backbone']:
        raise ValueError('Frozen backbone identity differs')
    loader=get_loader(cfg)
    # A missing supplied CSV must never fall back to downloaded/synthetic data.
    loader._fetch_from_api=lambda: (_ for _ in ()).throw(RuntimeError('Network fallback forbidden'))
    halos=loader.load()
    if loader._catalog_origin!='clustered_csv':
        raise ValueError('Only the exact clustered CSV is allowed; no fallback catalog')
    builder=GraphBuilder(cfg); builder.backend_used='portable_torch'
    splits=dict(zip(('train','val','test'),loader.split_data(halos)))
    split_graphs={name:[builder.build_graph(h) for h in items] for name,items in splits.items()}
    ids={name:[g.cluster_id for g in items] for name,items in split_graphs.items()}
    if ids!=manifest['context']['split_cluster_ids']:
        raise ValueError('Original split membership/order could not be reproduced')
    prepared=prepared_splits(split_graphs,gnn,device)
    policy=build_policy(cfg,node_emb_dim=gnn.output_dim).to(device)
    policy.load_state_dict(weights,strict=True); policy.eval()
    identity=dict(dataset=input_records,
        split_cluster_ids=ids, recovered_training_source=manifest['training_source'],
        recovered_policy_sha256=manifest['artifacts']['policy.pt']['sha256'],
        selected_epoch=manifest['policy_validation'].get('selected_epoch'),
        research_label='legacy_total_radius', frozen_backbone=record_backbone('frozen',gnn))
    return cfg,gnn,policy,prepared,identity


def policy_scores(policy,graph,cfg):
    layout=PhysicalPairLayout.from_edge_index(graph['edge_index'],len(graph['x']))
    cached=SimpleNamespace(call=lambda *a,**kw:SimpleNamespace(embeddings=graph['emb']))
    return PairScorer('provided_rl',policy=policy,edge_feature_names=cfg['graph']['edge_features']).score(
        as_data(graph),layout,0,meter=cached).scores.detach()
