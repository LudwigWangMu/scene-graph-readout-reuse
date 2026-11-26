"""Single-writer snapshot validation with exact incremental GNN replay."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
from time import perf_counter_ns
from collections import defaultdict, deque
import csv,json,platform,sys
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
DEPTH=3; WIDTH=12; DELAY=3
def dump(name,header,rows):
    with (ROOT/'data'/(name+'.csv')).open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(header);wr.writerows(rows)

def lattice(side):
    n=side**2+1;pad=n;a=np.full((n,5),pad,dtype=int)
    for i in range(n-1):
        r,c=divmod(i,side)
        for slot,j in enumerate((i-side if r else pad,i+side if r+1<side else pad,
                                 i-1 if c else pad,i+1 if c+1<side else pad)):a[i,slot]=j
    return a

def activation(prev,neighbors,weights,ids):
    # Sentinel row of zeros makes all reduction orders identical.
    x=prev[ids];messages=prev[neighbors[ids]].sum(axis=1)
    degree=(neighbors[ids]!=len(neighbors)).sum(axis=1).clip(1)
    return np.tanh(x@weights[0]+(messages/degree[:,None])@weights[1])

def full(x,a,w):
    n=len(a); layers=[x]
    for ww in w:
        y=np.zeros_like(x);y[:n]=activation(layers[-1],a,ww,slice(None,n));layers.append(y)
    return layers

def update(cache,a,w,changed,edges,version,stamps):
    dirty=np.array(changed,dtype=int);n=len(a);work=0
    for k,ww in enumerate(w):
        succ=a[dirty].ravel()
        dirty=np.unique(np.concatenate([dirty,succ[succ<n],edges])).astype(int)
        if len(dirty):cache[k+1][dirty]=activation(cache[k],a,ww,dirty)
        work+=len(dirty)
    stamps[dirty]=version
    return work

def attach(a,old,new):
    if old==new:return np.empty(0,dtype=int)
    hand=len(a)-1;pad=len(a)
    a[old,-1]=pad;a[new,-1]=hand;a[hand,0]=new
    return np.array([hand,old,new],dtype=int)

def cell(s,side):
    uv=np.clip(np.floor((s[:2]+[.45,.45])/.9*side),0,side-1).astype(int)
    return int(uv[0]*side+uv[1])

def encoding(s):
    return np.r_[s,s[:3]**2,1.]

def main():
    groups=defaultdict(list)
    with (ROOT/'data'/'trace.csv').open() as f:
        for row in csv.reader(f):
            if row[0]=='task':continue
            groups[(int(row[0]),int(row[1]))].append(np.array(row[3:],dtype=float))
    records=[];decisions=[];maxerr=0
    for side in (8,16,32,64):
        n=side**2+1;rng=np.random.default_rng(71)
        w=rng.normal(0,.18,(DEPTH,2,WIDTH,WIDTH))
        for (task,ep),seq in groups.items():
            a=lattice(side);x=rng.normal(0,.2,(n+1,WIDTH));x[-1]=0
            cur=cell(seq[0],side);hand=n-1;a[hand,0]=cur;a[cur,-1]=hand
            x[hand]=encoding(seq[0]);cache=full(x,a,w);stamp=np.zeros(n,dtype=int)
            queue=deque();prior=cache[-1].copy()
            for v,s in enumerate(seq[1:],1):
                dest=cell(s,side)
                # At issue time a readout may concern the local object or a uniformly sampled cell.
                read=cur if v%2 else int(rng.integers(0,side**2))
                queue.append((v-1,read,prior[read].copy()))
                changes=attach(a,cur,dest);cur=dest
                nextx=encoding(s);changed=[hand] if np.any(x[hand]!=nextx) else []
                x[hand]=nextx
                # Both timers exclude shared input decoding, adjacency writes and differential checks.
                def dense():
                    start=perf_counter_ns();result=full(x,a,w)
                    return (perf_counter_ns()-start)/1000,result
                def sparse():
                    start=perf_counter_ns();nn=update(cache,a,w,changed,changes,v,stamp)
                    return (perf_counter_ns()-start)/1000,nn
                if v%2:tf,reference=dense();ti,work=sparse()
                else:ti,work=sparse();tf,reference=dense()
                err=np.max(np.abs(reference[-1]-cache[-1]));maxerr=max(maxerr,err)
                records.append([side,task,ep,v,int(len(changes)>0),work,n*DEPTH,tf,ti,err])
                if len(queue)>=DELAY:
                    issued,node,old=queue.popleft()
                    altered=np.max(np.abs(reference[-1][node]-old))>1e-10
                    scoped=stamp[node]<=issued
                    decisions.append([side,task,ep,v,issued,node,int(scoped),int(altered)])
                    if scoped and altered:raise AssertionError('Stale output accepted')
                prior=reference[-1]
    dump('updates',['side','task','episode','version','rewire','rows_updated','full_rows','full_us','incremental_us','max_abs_error'],records)
    dump('decisions',['side','task','episode','release','issue','read_node','accepted','reference_changed'],decisions)
    summary=[]
    for side in (8,16,32,64):
        rs=[r for r in records if r[0]==side and r[3]>4];ds=[d for d in decisions if d[0]==side]
        summary.append([side,len(rs),np.median([r[7] for r in rs]),np.quantile([r[7] for r in rs],.95),
                    np.median([r[8] for r in rs]),np.quantile([r[8] for r in rs],.95),
                    np.mean([r[5]/r[6] for r in rs]),len(ds),sum(d[6] for d in ds),
                    sum(d[7] for d in ds),sum(d[6]*d[7] for d in ds)])
    dump('results',['side','timed','full_p50_us','full_p95_us','inc_p50_us','inc_p95_us','fraction_rows','decisions','scoped_accept','unvalidated_stale','scoped_stale'],summary)
    (ROOT/'data'/'environment.json').write_text(json.dumps(dict(os=platform.platform(),python=sys.version,numpy=np.__version__,cpu=platform.processor(),blas_threads=1,max_error=float(maxerr)),indent=2))
    print(summary,maxerr)
    plt.rcParams.update({'pdf.fonttype':42,'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(7.0,2.6))
    for k,label,c in [(2,'Full snapshot','#9d5828'),(4,'Incremental','#145973')]:
        axs[0].plot([r[0]**2+1 for r in summary],[r[k] for r in summary],marker='o',label=label,c=c)
        axs[0].plot([r[0]**2+1 for r in summary],[r[k+1] for r in summary],ls='--',c=c)
    axs[0].set(xscale='log',yscale='log',xlabel='Graph nodes',ylabel='Update time (microseconds)');axs[0].legend(fontsize=7,frameon=False)
    grid=np.arange(len(summary));axs[1].bar(grid-.17,[r[8]/r[7] for r in summary],width=.32,color='#145973',label='Scoped reuse')
    axs[1].bar(grid+.17,[r[9]/r[7] for r in summary],width=.32,color='#b66c32',label='Stale if unchecked')
    axs[1].set(xticks=grid,xticklabels=[r[0]**2+1 for r in summary],xlabel='Graph nodes',ylabel='Fraction of delayed requests',ylim=(0,1))
    axs[1].legend(fontsize=7,frameon=False)
    for ax in axs:ax.grid(axis='y',alpha=.16)
    fig.tight_layout();fig.savefig(ROOT/'figures'/'runtime.pdf');plt.close(fig)
if __name__=='__main__':main()
