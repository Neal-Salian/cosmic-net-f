"""Verify committed output bytes and halo/seed populations without running models."""
import argparse
import gzip
import json
from pathlib import Path
import pandas as pd
from data.provenance import sha256_file,hash_payload
from .analyze import validate_population


def verify_hashes(directory,manifest):
    directory=Path(directory)
    for name,record in manifest['outputs'].items():
        if sha256_file(directory/name)['sha256']!=record['sha256']:
            raise ValueError(f'output checksum mismatch: {name}')


def verify(directory,evidence):
    directory=Path(directory);evidence=Path(evidence)
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['status']!='executed':raise ValueError('run was not executed')
    verify_hashes(directory,manifest)
    analysis=json.loads((directory/'analysis_manifest.json').read_text())
    if sha256_file(directory/'manifest.json')['sha256']!=analysis['source_manifest']['sha256']:
        raise ValueError('analysis source manifest checksum mismatch')
    for record in analysis['outputs']:
        name=Path(record['path']).name
        path=directory/name if (directory/name).is_file() else evidence/name
        if sha256_file(path)['sha256']!=record['sha256']:raise ValueError(f'analysis output checksum mismatch: {path}')
    splits=json.loads((directory/'split_ids.json').read_text())
    cfg=json.loads((directory/'config.json').read_text())
    old=pd.read_csv(directory/'historical_replay.csv');old['replicate_seed']=42
    new=pd.read_csv(directory/'new_predictions.csv');val=pd.read_csv(directory/'validation_predictions.csv')
    for frame,split in [(old,'test'),(new,'test'),(val,'val')]:
        for _,group in frame.groupby(['method','retention']):validate_population(group,splits[split])
    expected={'random_repeated':cfg['random_mask_seeds'],'rl_sampled':cfg['policy_sampling_seeds'],
              'rl_tie_marks':cfg['tie_mark_seeds']}
    for method,seeds in expected.items():
        for q in cfg['retentions']:
            group=new[(new.method==method)&(new.retention==q)]
            if set(group.replicate_seed)!=set(seeds):raise ValueError('missing or mislabeled seed family')
    masks=0
    with gzip.open(directory/'masks.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line);selected=row['selected_pairs'];diag=row['diagnostics']
            if hash_payload(selected)!=row['selected_pairs_sha256']:raise ValueError('mask checksum mismatch')
            if sum(selected)!=diag['requested_pair_count'] or not diag['constraint_satisfied'] or not diag['budget_satisfied']:
                raise ValueError('invalid exported mask budget/diagnostic')
            masks+=1
    if masks!=len(new)+len(val):raise ValueError('unexported prediction masks')
    return {'status':'passed','historical_predictions':len(old),'new_test_predictions':len(new),
            'validation_predictions':len(val),'new_exported_masks':masks}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',required=True);p.add_argument('--evidence',default='paper/evidence')
    args=p.parse_args();print(json.dumps(verify(args.results,args.evidence),indent=2))
