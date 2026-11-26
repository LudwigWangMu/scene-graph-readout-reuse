"""Paired guard ablations on directed graphs, with an independent identity and numeric oracle."""
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from experiment import ROOT,full
from queued_runtime import directed_update

DELAYS=(1,4,8,16)
GUARDS=("Latest only","Row","Row + weights","Row + weights + context","Full contract","Unchecked")

def write(name,head,rows):
    with (ROOT/"data"/name).open("w",newline="") as f:
        wr=csv.writer(f);wr.writerow(head);wr.writerows(rows)

def simulate(seed,degree,initial,head):
    rng=np.random.default_rng(seed)
    n=64;w=initial.copy()
    a=np.array([rng.choice(np.delete(np.arange(n),i),degree,replace=False) for i in range(n)])
    x=np.vstack([rng.normal(0,.3,(n,12)),np.zeros((1,12))])
    cache=full(x,a,w); stamps=np.zeros(n,int);gen=np.zeros(n,int)
    epoch=0; ctx=0; history={}; rows=[]; events=defaultdict(int)
    for v in range(1,257):
        node=int(rng.integers(n))
        history[v-1]=(node,int(gen[node]),epoch,ctx,(cache[-1][node]@head+.013*ctx).copy())
        # Each version carries only one event. Generate identical schedules across guard policies.
        who=int(rng.integers(n)); typ=rng.choice(["feature","rewire","identity","context","weight"],p=[.66,.17,.08,.055,.035])
        events[typ]+=1
        if typ=="weight":
            epoch+=1;w[0,0,0,0]+=.08;cache=full(x,a,w)
        elif typ=="context":ctx+=1
        elif typ=="identity":gen[who]+=1
        else:
            if typ=="feature":
                x[who]=rng.normal(0,.3,12);changed=[who];edges=[]
            else:
                a[who]=rng.choice(np.delete(np.arange(n),who),degree,replace=False);changed=[];edges=[who]
            directed_update(cache,a,w,np.array(changed,int),np.array(edges,int),v,stamps)
        fresh=full(x,a,w)[-1]
        assert np.max(abs(cache[-1]-fresh))<1e-10
        for delay in DELAYS:
            if v<delay:continue
            issue=v-delay;read,g,e,c,old=history[issue]
            val=bool(np.max(np.abs(old-(fresh[read]@head+.013*ctx)))>1e-10)
            obj=bool(g!=gen[read]);invalid=val or obj
            row=stamps[read]<=issue;we=e==epoch;cx=c==ctx
            decisions=(v==issue,row,row and we,row and we and cx,row and we and cx and not obj,True)
            for guard,ok in zip(GUARDS,decisions):
                if guard=="Full contract" and ok and invalid:raise AssertionError((seed,v,delay))
                rows.append((seed,degree,v,delay,read,typ,guard,int(ok),int(val),int(obj)))
    return rows

def main():
    with np.load(ROOT/"data/learned_weights.npz") as data:w,head=data["layers"],data["readout"]
    detail=[]
    for degree in [1,3]:
        for seed in range(301,309):detail.extend(simulate(seed,degree,w,head))
    write("release_sweep.csv",["seed","degree","version","delay","read","event","guard","accepted","numeric_stale","identity_stale"],detail)
    summary=[]
    for degree in [1,3]:
        for d in DELAYS:
            for guard in GUARDS:
                rr=[r for r in detail if r[1]==degree and r[3]==d and r[6]==guard]
                bad=sum(r[7] and (r[8] or r[9]) for r in rr)
                true=sum(not (r[8] or r[9]) for r in rr)
                accepted=sum(r[7] for r in rr)
                summary.append((degree,d,guard,len(rr),accepted,bad,true,true-(accepted-bad)))
    write("release_sweep_summary.csv",["degree","delay","guard","queries","accepted","stale_release","oracle_valid","valid_rejected"],summary)
    fig,axes=plt.subplots(1,2,figsize=(6.9,2.5))
    for ax,k,lab in zip(axes,[4,5],["Released (%)","Stale releases (% of requests)"]):
        for dg,style in [(1,"-"),(3,"--")]:
            for gu,col,mk in [("Row","#ac532d","s"),("Full contract","#12677a","o")]:
                rs=[r for r in summary if r[0]==dg and r[2]==gu]
                ax.plot(DELAYS,[100*r[k]/r[3] for r in rs],color=col,ls=style,marker=mk,ms=3,label=f"{gu}, degree {dg}")
        ax.set_xlabel("Delay (commits)");ax.set_ylabel(lab);ax.set_xticks(DELAYS);ax.spines[["right","top"]].set_visible(False);ax.grid(axis="y",alpha=.2)
    axes[0].legend(fontsize=6,frameon=False,ncol=2);fig.tight_layout();fig.savefig(ROOT/"figures/release_sweep.pdf",bbox_inches="tight");plt.close(fig)
    with (ROOT/"data/queued_packets.csv").open() as f:packets=list(csv.DictReader(f))
    rr=[];fields=["input_queue_ns","graph_ns","decoder_ns","sink_ns"]
    for mode in ["full","local"]:
        packets_m=[r for r in packets if r["mode"]==mode and int(r["frame"])>=4]
        ms=np.array([[float(r[k]) for k in fields] for r in packets_m])/1e6
        total=np.array([float(r["injection_to_ack_ns"]) for r in packets_m])/1e6
        assert np.allclose(ms.sum(1),total,atol=1e-12,rtol=0)
        cut=np.quantile(total,.99)
        for name,subset in [("All",np.ones(len(total),dtype=bool)),("Top percentile",total>=cut)]:
            z=ms[subset];rr.append((mode,name,len(z),*z.mean(0),total[subset].mean()))
    write("queue_stages.csv",["mode","subset","n","input","graph","decoder","sink","total_ms"],rr)
if __name__=="__main__":main()

