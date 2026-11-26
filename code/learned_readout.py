"""Exercise the version check with a fitted object-motion decoder on held-out tracks."""
from tracked_objects import graph
from experiment import *
import torch
from torch import nn
torch.set_num_threads(2)
def features(pos,t):
    lag=[(pos[t]-pos[t-h])/(h*4) for h in (1,3,6)]
    return np.column_stack([pos[t]/512,*lag,*[np.linalg.norm(v,axis=1) for v in lag],np.ones(len(pos[t]))])
def examples(pos):
    for t in np.unique(np.linspace(6,len(pos)-5,min(96,len(pos)-10),dtype=int)):
        yield features(pos,t),graph(pos[t]),pos[t+4]-pos[t]
class Network(nn.Module):
    def __init__(self):
        super().__init__();self.w=nn.Parameter(torch.randn(3,2,12,12)*.18)
        self.head=nn.Parameter(torch.randn(12,2)*.1)
    def forward(self,x,adj):
        for w in self.w:x=torch.tanh(x@w[0]+(adj@x)@w[1])
        return x@self.head
def main():
    torch.manual_seed(334);rng=np.random.default_rng(352)
    d=np.load(ROOT/'data'/'object_tracks.npz');samples=[]
    mx=max(x.shape[1] for x in d.values())
    for key in sorted(d):
        if int(key.rsplit('__',1)[1])>=10:continue
        for x,a,y in examples(d[key]):
            n=len(x);xx=np.zeros((mx,WIDTH));xx[:n]=x;adj=np.zeros((mx,mx))
            for i in range(n):
                ids=a[i,a[i]<n];adj[i,ids]=1/max(1,len(ids))
            yy=np.zeros((mx,2));yy[:n]=y/12;samples.append([xx,adj,yy,np.arange(mx)<n])
    X,A,Y,M=[torch.tensor(np.stack(v),dtype=torch.float32) for v in zip(*samples)]
    model=Network();opt=torch.optim.AdamW(model.parameters(),lr=.003)
    for _ in range(35):
        for ix in torch.randperm(len(X)).split(96):
            res=(model(X[ix],A[ix])-Y[ix]).square().sum(-1)
            loss=(res*M[ix]).sum()/M[ix].sum();opt.zero_grad();loss.backward();opt.step()
    w=model.w.detach().numpy().astype('f8');head=model.head.detach().numpy().astype('f8')*12
    np.savez(ROOT/'data'/'learned_weights.npz',layers=w,readout=head)
    rows=[];dec=[];maxerr=0
    for key in sorted(d):
        if int(key.rsplit('__',1)[1])<10:continue
        queue=deque()
        for version,(xx,new,y) in enumerate(examples(d[key])):
            n=len(xx)
            if version==0:
                x=np.r_[xx,np.zeros((1,WIDTH))];a=new;cache=full(x,a,w);stamp=np.zeros(n,dtype=int)
            else:
                node=int(rng.integers(n));queue.append((version-1,node,reference[node].copy()))
                changed=np.flatnonzero((x[:n]!=xx).any(1));edges=np.flatnonzero((a!=new).any(1))
                x[:n]=xx;a=new;update(cache,a,w,changed,edges,version,stamp)
            reference=full(x,a,w)[-1][:n]@head;pred=cache[-1][:n]@head
            maxerr=max(maxerr,np.max(np.abs(pred-reference)))
            if len(queue)>=3:
                issued,node,old=queue.popleft();reuse=stamp[node]<=issued;stale=np.max(abs(old-reference[node]))>1e-10
                dec.append([key,version,node,int(reuse),int(stale)])
                if reuse and stale:raise AssertionError('Changed read accepted')
            rows.append([key,version,float(np.linalg.norm(reference-y,axis=1).mean()),float(np.linalg.norm(y,axis=1).mean())])
    dump('learned_errors',['track','version','motion_error_px','persistence_error_px'],rows)
    dump('learned_decisions',['track','version','node','reused','stale_unchecked'],dec)
    episodes=sorted({r[0] for r in rows});scores=np.mean([np.mean([r[2:] for r in rows if r[0]==e],axis=0) for e in episodes],axis=0)
    print('fit windows',len(X),'test windows',len(rows),'episodes',len(episodes),'MAE',scores,'error',maxerr,'reused',sum(x[3] for x in dec),'of',len(dec),'unchecked stale',sum(x[4] for x in dec))
if __name__=='__main__':main()
