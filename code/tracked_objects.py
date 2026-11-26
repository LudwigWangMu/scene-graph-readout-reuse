"""Repeat exactness and latency checks on small public tracked-object graphs."""
from experiment import *
import argparse,hashlib,urllib.request,zipfile
URL='https://github.com/CEA-LIST/robot-watchdog-code/releases/download/v1.0.0/robot_watchdog_data.zip'
SHA='84e2d48bec79679a51cc5baa4f3baef07cbaf9357043852119db146d3d51193d'
def graph(pos):
    d=np.linalg.norm(pos[:,None]-pos[None,:],axis=2);n=len(pos)
    ids=np.broadcast_to(np.arange(n),(n,n))
    return np.where((d<=80)&(~np.eye(n,dtype=bool)),ids,n)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path)
    archive=ap.parse_args().archive
    from prepare_objects import extract
    data = extract(archive) if archive else np.load(ROOT/'data'/'object_tracks.npz')
    rows=[];rgen=np.random.default_rng(712);w=rgen.normal(0,.18,(DEPTH,2,WIDTH,WIDTH))
    for key in sorted(data):
        task,episode=key.rsplit('__',1);ei=int(episode)
        if ei>=10:continue
        pos=data[key];num,n,_=pos.shape
        if not np.isfinite(pos).all():raise ValueError('Nonfinite coordinate')
        x=np.zeros((n+1,WIDTH));x[:n]=rgen.normal(0,.2,(n,WIDTH));x[:n,:2]=pos[0]/512
        a=graph(pos[0]);cache=full(x,a,w);stamps=np.zeros(n,dtype=int)
        for v,j in enumerate(np.linspace(1,num-1,min(64,num-1),dtype=int),1):
            a1=graph(pos[j]);changed=np.flatnonzero(np.any(x[:n,:2]!=pos[j]/512,axis=1))
            edges=np.flatnonzero((a!=a1).any(1));x[:n,:2]=pos[j]/512;a=a1
            def dense():
                before=perf_counter_ns();res=full(x,a,w);return res,(perf_counter_ns()-before)/1000
            def sparse():
                before=perf_counter_ns();cnt=update(cache,a,w,changed,edges,v,stamps);return cnt,(perf_counter_ns()-before)/1000
            if v%2:ref,td=dense();work,ts=sparse()
            else:work,ts=sparse();ref,td=dense()
            err=float(np.max(np.abs(ref[-1]-cache[-1])))
            if err>1e-10:raise AssertionError(err)
            rows.append([task,ei,v,n,len(edges),len(changed),work,td,ts,err])
    dump('tracked_updates',['task','episode_position','update','nodes','edge_receivers','feature_edits','rows_updated','full_us','incremental_us','error'],rows)
    rs=[r for r in rows if r[2]>4]
    print('updates',len(rows),'timed',len(rs),'episodes',len({(r[0],r[1]) for r in rows}),'nodes',sorted({r[3] for r in rows}),
          'p50/95',np.quantile([r[7] for r in rs],[.5,.95]),np.quantile([r[8] for r in rs],[.5,.95]),'error',max(r[-1] for r in rows))
if __name__=='__main__':main()
