"""Host-side queued latency and adversarial ordering tests for a versioned local decoder."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import csv,json,platform,threading,queue
from time import perf_counter_ns as clock
from collections import Counter
import numpy as np
from tracked_objects import graph
from experiment import full, activation, ROOT

def features(history):
    pos=history[-1]
    lag=[(pos-history[-1-h])/(4*h) for h in (1,3,6)]
    return np.column_stack([pos/512,*lag,*[np.linalg.norm(v,axis=1) for v in lag],np.ones(len(pos))])

class Dense:
    def __init__(self,n,d):
        self.m=np.zeros((n,n));self.h=np.empty((n,d));self.neigh=self.h.copy();self.tmp=self.h.copy()
    def forward(self,x,adj,w):
        self.m[:]=0
        rows,cols=np.nonzero(adj<len(adj))
        self.m[rows,adj[rows,cols]]=1.
        self.m/=np.maximum(1,self.m.sum(1,keepdims=True))
        h=x
        for w0,w1 in w:
            np.matmul(self.m,h,out=self.neigh);np.matmul(h,w0,out=self.tmp)
            np.matmul(self.neigh,w1,out=self.h);np.add(self.tmp,self.h,out=self.h);np.tanh(self.h,out=self.h)
            h=self.h
        return h

def directed_update(cache,adj,w,changed,edits,version,stamp):
    # Rows list incoming neighbors. Transpose traversal locates all consuming receivers.
    n=len(adj); dirty=np.zeros(n+1,dtype=bool);dirty[changed]=True
    previous=0
    for layer,weights in enumerate(w):
        dirty[:n]|=dirty[adj].any(1)
        dirty[edits]=True; ids=np.flatnonzero(dirty[:n]);previous+=len(ids)
        if len(ids):cache[layer+1][ids]=activation(cache[layer],adj,weights,ids)
    stamp[dirty[:n]]=version
    return previous

def run_pipe(pos,w,head,mode):
    incoming=queue.Queue(1);outgoing=queue.Queue(1);ack=queue.Queue(1)
    errors=[];rows=[]
    def worker():
        try:
            cache=None;old=None;previous_token=None
            while True:
                packet=incoming.get()
                if packet is None:break
                idx,t0,history=packet;t1=clock();xx=features(history);a=graph(history[-1]);n=len(a)
                if old is None:
                    dense=Dense(n,xx.shape[1]);stamps=np.zeros(n,dtype=int)
                    if mode=='local':cache=full(np.vstack([xx,np.zeros((1,xx.shape[1]))]),a,w)
                if mode=='local':
                    if old is not None:
                        change=np.flatnonzero((old!=xx).any(1));rewire=np.flatnonzero((adj!=a).any(1))
                        cache[0][:n]=xx
                        directed_update(cache,a,w,change,rewire,idx,stamps)
                    z=cache[-1][:n]
                else:z=dense.forward(xx,a,w)
                t2=clock();read=idx%n;reused=False
                if previous_token is None:command=z[read]@head
                else:
                    prevrow,previssued,previous,prevembedding=previous_token;read=prevrow
                    reused=(stamps[read]<=previssued if mode=='local' else np.array_equal(prevembedding,z[read]))
                    command=previous if reused else z[read]@head
                nextread=idx%n
                previous_token=(nextread,idx,(z[nextread]@head).copy(),z[nextread].copy())
                old=xx;adj=a;t3=clock()
                outgoing.put((idx,t0,t1,t2,t3,read,reused,command.copy()))
        except BaseException as exc:errors.append(repr(exc))
        finally:outgoing.put(None)
    def sink():
        while True:
            pkt=outgoing.get()
            if pkt is None:break
            idx,t0,t1,t2,t3,node,reuse,command=pkt
            # Software sink acknowledges receipt of the two predicted coordinates.
            x,y=map(float,command);stamp=clock();ack.put((idx,node,reuse,x,y,t1-t0,t2-t1,t3-t2,stamp-t3,stamp-t0))
    ta=threading.Thread(target=worker);tb=threading.Thread(target=sink);ta.start();tb.start()
    try:
        for idx,j in enumerate(np.unique(np.linspace(6,len(pos)-5,min(64,len(pos)-10),dtype=int))):
            t0=clock();incoming.put((idx,t0,pos[j-6:j+1].copy()))
            try:rows.append(ack.get(timeout=5))
            except queue.Empty:raise RuntimeError(errors or 'worker timed out')
    finally:
        incoming.put(None);ta.join(6);tb.join(6)
    return rows

def order_stress(w,head,n,degree):
    rng=np.random.default_rng(895)
    x=np.vstack([rng.normal(0,.3,(n,12)),np.zeros((1,12))]);a=np.full((n,degree),n)
    for i in range(n):a[i]=rng.choice(np.delete(np.arange(n),i),degree,replace=False)
    cache=full(x,a,w);stamps=np.zeros(n,int);generation=np.zeros(n,int)
    epoch=0;aux=0;version=0;last=[-1,-1];pending=[];rows=[];kinds=Counter();maxerr=0.
    def add(source):
        local=np.random.default_rng(1301+source);pack=[]
        for k in range(180):
            owner=2*int(local.integers((n-source+1)//2))+source
            pack.append((source,k,owner,local.normal(0,.3,12)))
            if k%11==6:pack[-1],pack[-2]=pack[-2],pack[-1]
            if k%13==7:pack.append(pack[-1])
        for start in range(0,len(pack),8):
            batches[source].put(pack[start:start+8])
        batches[source].put(None)
    batches=[queue.Queue(),queue.Queue()];ts=[threading.Thread(target=add,args=(s,)) for s in range(2)]
    for th in ts:th.start()
    finished=set();i=0
    while len(finished)<2:
        source=i%2;i+=1
        if source in finished:continue
        block=batches[source].get()
        if block is None:finished.add(source);continue
        for src,seq,node,value in block:
            if seq<=last[src]:kinds['old_or_duplicate']+=1;continue
            last[src]=seq;version+=1
            read=int(rng.integers(n));pending.append((read,version-1,int(generation[read]),epoch,aux,(cache[-1][read]@head+aux*.01).copy()))
            x[node]=value;edits=np.array([],int)
            if version%5==0:
                recv=int(rng.integers(n));a[recv]=rng.choice(np.delete(np.arange(n),recv),degree,replace=False);edits=np.array([recv]);kinds['rewire']+=1
            if version%23==0:generation[node]+=1;kinds['replace_id']+=1
            if version%41==0:aux+=1;kinds['aux_change']+=1
            if version%67==0:
                w=w.copy();w[0,0,0,0]+=.05;epoch+=1;cache=full(x,a,w);stamps[:]=version;kinds['weights']+=1
            else:directed_update(cache,a,w,np.array([node]),edits,version,stamps)
            fresh=full(x,a,w)[-1];maxerr=max(maxerr,float(np.max(abs(fresh-cache[-1]))))
            if len(pending)>=4:
                who,issued,gen,e,u,y=pending.pop(0)
                ok=(generation[who]==gen and epoch==e and aux==u and stamps[who]<=issued)
                delta=float(np.max(abs(y-(fresh[who]@head+aux*.01))))
                rows.append((version,src,seq,who,issued,int(ok),delta))
                if ok and delta>1e-10:raise AssertionError(rows[-1])
    for th in ts:th.join()
    if maxerr>1e-10:raise AssertionError(maxerr)
    return rows,dict(kinds),maxerr

def write(name,head,rows):
    with (ROOT/'data'/name).open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(head);wr.writerows(rows)

def main():
    data=np.load(ROOT/'data/object_tracks.npz');wt=np.load(ROOT/'data/learned_weights.npz')
    w,head=wt['layers'],wt['readout'];rows=[];maxdelta=0.
    for k,key in enumerate(sorted(key for key in data if int(key.rsplit('__',1)[1])>=10)):
        for repeat in range(2):
            results={}
            for mode in (['full','local'] if (k+repeat)%2==0 else ['local','full']):
                results[mode]=run_pipe(data[key],w,head,mode)
                rows.extend((key,repeat,mode,*r) for r in results[mode])
            for s,t in zip(results['full'],results['local']):
                assert s[:2]==t[:2]
                maxdelta=max(maxdelta,float(np.max(abs(np.array(s[3:5])-t[3:5]))))
    if maxdelta>1e-10:raise AssertionError(maxdelta)
    write('queued_packets.csv',['track','repeat','mode','frame','read','reuse','output_x','output_y','input_queue_ns','graph_ns','decoder_ns','sink_ns','injection_to_ack_ns'],rows)
    stress,counts,err=order_stress(w,head,19,3)
    sparse,s_counts,s_err=order_stress(w,head,61,1)
    assert counts==s_counts
    def audit(rs):
        return dict(releases=len(rs),accepted=sum(r[-2] for r in rs),unchecked_stale=sum(r[-1]>1e-10 for r in rs),accepted_stale=sum(r[-2] and r[-1]>1e-10 for r in rs))
    write('directed_sparse_release.csv',['commit','writer','source_sequence','read','issued','accepted','oracle_delta'],sparse)
    write('directed_release.csv',['commit','writer','source_sequence','read','issued','accepted','oracle_delta'],stress)
    summary={}
    for mode in ['full','local']:
        ms=np.array([r[-1] for r in rows if r[2]==mode and r[3]>=4])/1e6
        summary[mode]={'samples':len(ms),'p50_p95_p99_ms':np.quantile(ms,[.5,.95,.99]).tolist(),'max_ms':float(ms.max())}
    info=dict(queue=summary,output_delta=maxdelta,directed_tests=len(stress),accepted=sum(r[-2] for r in stress),unchecked_stale=sum(r[-1]>1e-10 for r in stress),accepted_stale=sum(r[-2] and r[-1]>1e-10 for r in stress),events_per_design=counts,sparse=audit(sparse),sparse_layer_delta=s_err,layer_delta=err,numpy=np.__version__,platform=platform.platform())
    (ROOT/'data'/'queued_summary.json').write_text(json.dumps(info,indent=2))
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(3.45,1.95))
    for mode,label,col in [('full','Full, preallocated','#a36231'),('local','Local, directed','#196779')]:
        t=np.sort([r[-1]/1e6 for r in rows if r[2]==mode and r[3]>=4])
        ax.plot(t,np.arange(1,len(t)+1)/len(t),label=label,color=col,linewidth=1.5)
    ax.set(xlabel='Host injection to software receipt (ms)',ylabel='Empirical CDF',xlim=(0,max(summary[m]['p50_p95_p99_ms'][-1] for m in summary)*1.15),ylim=(0,1.03))
    ax.tick_params(labelsize=7);ax.xaxis.label.set_size(8);ax.yaxis.label.set_size(8)
    ax.legend(fontsize=6,frameon=False,loc='lower right');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(pad=.45);fig.savefig(ROOT/'figures'/'queued_cdf.pdf');plt.close(fig)
    print(info)
if __name__=='__main__':main()

