"""Regenerate manuscript assets from immutable campaign snapshots; no training."""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((PAPER / 'evidence/source-manifest.json').read_text())

def source(suffix):
    matches = [r for r in MANIFEST['files'] if r['campaign_relative_path'].endswith(suffix)]
    assert len(matches) == 1, suffix
    r = matches[0]
    path = PAPER.parent / r['snapshot']
    content = path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == r['snapshot_sha256']
    if path.suffix == '.gz':
        content = gzip.decompress(content)
    assert hashlib.sha256(content).hexdigest() == r['sha256_uncompressed']
    return content

def rows(suffix):
    return list(csv.DictReader(source(suffix).decode().splitlines()))

def number(row, key):
    return float(row[key])

for record in MANIFEST['files']:
    source(record['campaign_relative_path'])

C = rows('notebook_C/results_table.csv')
D = rows('notebook_D/tta_results.csv')
DV = rows('notebook_D/tta_validation.csv')
RAW = json.loads(source('notebook_C/matched_per_example.json'))
PAIRED = json.loads(source('scientific-analysis-20261016/paired-comparisons.json'))
LABELS = {'full':'Full graph', 'random':'Random', 'distance':'Distance', 'degree':'Degree',
          'stellar_binding_proxy':'Stellar binding', 'edge_feature_grad_x_input':r'Gradient $\times$ input',
          'supervised_raw':'Trained selector', 'provided_rl':'RL policy'}
ORDER = list(LABELS)
Q = [.4,.6,.8,1.]
lookup = {(r['method'], number(r,'frac')):r for r in C}
assert len(C) == 32 and len(RAW) == 2624
assert len({r['graph_id'] for r in RAW}) == 82
for key, row in lookup.items():
    raw = [r for r in RAW if (r['method'],r['requested_keep_fraction']) == key]
    assert len(raw) == 82 and all(r['status']=='success' for r in raw)
    pred = np.array([r['full_prediction'] if key[0]=='full' else r['pruned_prediction'] for r in raw])
    target = np.array([r['target'] for r in raw])
    rmse = np.sqrt(np.mean((pred-target)**2))
    assert abs(rmse-number(row,'rmse')) < 1e-10
    if key[0] != 'full':
        for r in raw:
            assert r['requested_pair_count'] == int(np.ceil(key[1]*r['available_pair_count']))
            assert r['final_pair_count'] == r['requested_pair_count']
            assert r['physical_isolates'] == 0 and r['budget_satisfied'] and r['constraint_satisfied']
            indices = np.array(r['selected_pairs'],dtype=int)
            selected = np.zeros(r['available_pair_count'],dtype=bool)
            assert len(np.unique(indices)) == len(indices)
            assert (indices >= 0).all() and (indices < len(selected)).all()
            selected[indices] = True
            inverse = np.array(r['layout_inverse'],dtype=int)
            real = np.array(r['stored_real'],dtype=bool)
            mask = np.array(r['stored_mask'],dtype=bool)
            assert np.array_equal(mask[real],selected[inverse]) and mask[~real].all()
            pairs = np.array(r['layout_pairs'],dtype=int)
            assert set(pairs[selected].reshape(-1)) == set(pairs.reshape(-1))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
                     'axes.labelsize':8,'legend.fontsize':7,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
COLORS = {'full':'#20252b','random':'#8c8c8c','distance':'#c77b24','degree':'#6c6bb1',
          'stellar_binding_proxy':'#b84444','edge_feature_grad_x_input':'#16857a',
          'supervised_raw':'#a36fa8','provided_rl':'#205cab'}
def save(fig,name):
    fig.savefig(PAPER/'figures'/f'{name}.pdf',bbox_inches='tight',metadata={'CreationDate':None})
    fig.savefig(PAPER/'figures'/f'{name}.png',dpi=220,bbox_inches='tight')
    plt.close(fig)

fig, ax = plt.subplots(figsize=(3.35,2.65),layout='constrained')
for method in ORDER:
    values=[number(lookup[method,q],'rmse') for q in Q]
    ax.plot(np.array(Q)*100,values,label=LABELS[method],color=COLORS[method],
            marker=None if method=='full' else 'o',markersize=3,
            linewidth=1.8 if method in ('provided_rl','full') else 1,
            linestyle='--' if method=='full' else '-')
ax.set(xlabel='Requested physical-pair retention (%)',ylabel='Test RMSE (dex)',xticks=[40,60,80,100])
ax.grid(alpha=.15)
ax.legend(ncol=2,loc='upper right',frameon=False,handlelength=1.2,columnspacing=.65)
ax.set_ylim(.108,.24)
save(fig,'accuracy_tradeoff')

q40={r['control']:r for r in PAIRED['contrasts'] if r['requested_keep_fraction']==.4}
controls=['edge_feature_grad_x_input','random','supervised_raw','degree','distance','stellar_binding_proxy','full']
fig, ax=plt.subplots(figsize=(3.35,2.5),layout='constrained')
for i,method in enumerate(controls):
    r=q40[method];x=r['delta_rmse_rl_minus_control']
    ax.errorbar(x,i,xerr=[[x-r['ci95_bca_low']],[r['ci95_bca_high']-x]],
                fmt='o',color=COLORS[method],capsize=2,markersize=4)
ax.axvline(0,color='black',linestyle='--',linewidth=.8)
ax.set_yticks(range(len(controls)),[LABELS[m] for m in controls]);ax.invert_yaxis()
ax.set(xlabel='RMSE(RL) - RMSE(control) [dex]',xlim=(-.105,.044))
ax.grid(axis='x',alpha=.15)
save(fig,'paired_uncertainty')

fig,axes=plt.subplots(1,2,figsize=(7.0,2.5),layout='constrained')
ax=axes[0];rl=lookup['provided_rl',.4];full=lookup['full',.4]
total=number(rl,'mean_inference_time_s')
parts=[number(rl,k) for k in ['mean_scoring_time_s','mean_decoder_time_s','mean_prediction_time_s']]
parts.append(max(0,total-sum(parts)))
left=0
for value,label,color in zip(parts,['Scoring incl. embeddings','Exact decoder','Pruned prediction','Other region overhead'],['#205cab','#dda94c','#16857a','#c6ccd1']):
    ax.barh(1,value,left=left,color=color,label=label,height=.45);left+=value
ax.barh(0,number(full,'mean_inference_time_s'),height=.45,color='#20252b',label='Full prediction')
ax.set(yticks=[0,1],yticklabels=['Full graph','RL, q=40%'],xlabel='Measured inference region (s/halo)',xlim=(0,.50),ylim=(-.55,2.3))
ax.text(number(full,'mean_inference_time_s')+.007,0,f"{number(full,'mean_inference_time_s'):.5f}",va='center',fontsize=7)
ax.text(total+.007,1,f'{total:.3f}',va='center',fontsize=7)
ax.legend(loc='upper left',fontsize=6.6,frameon=False,ncol=2,columnspacing=.5,handlelength=1)
ax.set_title('(a) C: scoring and decoding dominate',loc='left')
ax=axes[1]
lat=np.array([number(r,'mean_time_s') for r in D]);err=np.array([number(r,'rmse') for r in D])
ax.plot(lat,err,'o-',color='#205cab',markersize=4,linewidth=1)
for r,x,y in zip(D,lat,err):
    offset={'0':(7,0),'5':(7,4),'10':(-38,7),'20':(-5,-12)}[r['K']]
    ax.annotate('K='+r['K'],(x,y),xytext=offset,textcoords='offset points',fontsize=7)
ax.set(xlabel='D adaptation + evaluation (s/halo)',ylabel='Test RMSE (dex)',xlim=(.40,1.30),ylim=(.13015,.13132))
ax.ticklabel_format(axis='y',style='plain',useOffset=False);ax.grid(alpha=.15)
ax.set_title('(b) D: separate q=50% experiment',loc='left')
save(fig,'efficiency_and_tta')

fig,axes=plt.subplots(1,2,figsize=(7,2.65),layout='constrained')
def by_id(method):return {r['graph_id']:r for r in RAW if r['method']==method and r['requested_keep_fraction']==.4}
rr,gg,ff=map(by_id,['provided_rl','edge_feature_grad_x_input','full']);ids=sorted(rr)
y=np.array([rr[i]['target'] for i in ids]);pr=np.array([rr[i]['pruned_prediction'] for i in ids]);pg=np.array([gg[i]['pruned_prediction'] for i in ids]);pf=np.array([ff[i]['full_prediction'] for i in ids])
ax=axes[0];ax.scatter(y,pf,s=14,label='Full graph',alpha=.65,color=COLORS['full']);ax.scatter(y,pr,s=16,marker='x',label='RL, q=40%',color=COLORS['provided_rl']);lo=min(y.min(),pr.min(),pf.min())-.05;hi=max(y.max(),pr.max(),pf.max())+.05;ax.plot([lo,hi],[lo,hi],color='#777777',linestyle='--',linewidth=.7);ax.set(xlabel='Stored target (log mass)',ylabel='Prediction (log mass)',xlim=(lo,hi),ylim=(lo,hi));ax.legend(frameon=False,fontsize=7)
ax=axes[1];a=np.abs(pg-y);b=np.abs(pr-y);hi=max(a.max(),b.max())+.02;ax.scatter(a,b,s=15,alpha=.75,color=COLORS['provided_rl']);ax.plot([0,hi],[0,hi],color='#777777',linestyle='--',linewidth=.7);ax.set(xlabel=r'Absolute error: gradient $\times$ input (dex)',ylabel='Absolute error: RL (dex)',xlim=(0,hi),ylim=(0,hi));ax.text(.04,.92,'42/82 favor RL; 40/82 favor saliency',transform=ax.transAxes,fontsize=7)
save(fig,'per_halo_diagnostics')

def table(name,header,body,cols):
    (PAPER/'tables'/name).write_text('\\begin{tabular}{'+cols+'}\n\\toprule\n'+header+' \\\\\n\\midrule\n'+'\n'.join(body)+'\n\\bottomrule\n\\end{tabular}\n')

body=[]
for method in ORDER:
    label=LABELS[method].replace('$\\times$','$\\times$')
    body.append(label+' & '+' & '.join(f"{number(lookup[method,q],'rmse'):.6f}" for q in Q)+r' \\')
table('accuracy.tex',r'Method & 40\% & 60\% & 80\% & 100\%',body,'lrrrr')
body=[]
for method in controls:
    r=q40[method];label=LABELS[method]
    body.append(f"{label} & {r['delta_rmse_rl_minus_control']:+.6f} & [{r['ci95_bca_low']:+.6f}, {r['ci95_bca_high']:+.6f}] & {r['paired_permutation_mse_p_two_sided']:.5f} & {r['holm_p_q40_family7']:.5f}"+r' \\')
table('paired.tex',r'Control & $\Delta$ (dex) & Marginal 95\% BCa interval (dex) & Paired $p$ & Holm $p$ (7)',body,'lrrrr')
body=[]
for r,v in zip(D,DV):
    body.append(f"{r['K']} & {number(v,'rmse'):.9f} & {number(r,'rmse'):.9f} & {number(r,'mean_time_s'):.3f} & {number(r,'mean_steps'):.2f}"+r' \\')
table('tta.tex',r'$K$ & Validation RMSE & Test RMSE & s/halo & Steps',body,'rrrrr')

summary={'source_files_verified':len(MANIFEST['files']),'configurations':32,'test_halos':82,'raw_rows':len(RAW),'nonfull_masks_verified':sum(r['method']!='full' for r in RAW),'rmse_recomputed_from_predictions':True,'all_exact_budgets_symmetry_loops_and_coverage_verified':True,'q40_rl_vs_full_pct':100*(number(rl,'rmse')/number(full,'rmse')-1),'q40_rl_slowdown':total/number(full,'mean_inference_time_s'),'q40_physical_retention':number(rl,'mean_keep_frac'),'q40_stored_retention':number(rl,'mean_stored_keep_frac'),'D_improvement_pct':100*(1-err[-1]/err[0]),'D_latency_factor':lat[-1]/lat[0]}
(PAPER/'evidence/asset-verification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
