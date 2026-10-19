"""Export committed source for Kaggle without depending on a published branch."""
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from .core import BASE_COMMIT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); root=Path(__file__).resolve().parents[2]
    def git(*a): return subprocess.check_output(['git','-C',str(root),*a],timeout=15)
    commit=git('rev-parse','HEAD').decode().strip()
    subprocess.run(['git','-C',str(root),'merge-base','--is-ancestor',BASE_COMMIT,commit],check=True,timeout=15)
    prefixes=('rls/','data/','graph/','model/','training/','symbolic/','experiments/','config/','tests/','notebooks/')
    names=[n for n in git('ls-tree','-r','--name-only',commit).decode().splitlines()
        if (n.startswith(prefixes) and (n.endswith(('.py','.yaml','.ipynb','.json','.md')))
            and not n.startswith('experiments/paper_ablations/results/'))
            or n in ('conftest.py','pytest.ini','requirements.txt')]
    files={name:git('show',f'{commit}:{name}') for name in names}
    # Refuse to claim that uncommitted executable edits belong to HEAD.
    for name,data in files.items():
        if (root/name).read_bytes()!=data:
            raise ValueError(f'Commit source changes before packaging: {name}')
    manifest=dict(commit=commit,base_commit=BASE_COMMIT,
        files={name:hashlib.sha256(data).hexdigest() for name,data in files.items()})
    with zipfile.ZipFile(args.output,'x',zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items(): z.writestr(name,data)
        z.writestr('ABLATION_SOURCE.json',json.dumps(manifest,indent=2))
    print(json.dumps(dict(path=str(args.output.resolve()),commit=commit,
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest()),indent=2))


if __name__=='__main__': main()
