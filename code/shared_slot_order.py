"""Compare register conflict rules with a reference computed from all delivered observations."""
import csv
import numpy as np
from experiment import ROOT,full

RULES=("arrival","per-source","ordered","biased-clock")

def run(seed,w,head):
    rng=np.random.default_rng(seed);n=24;shared=4
    x=np.vstack([rng.normal(0,.35,(n,12)),np.zeros((1,12))])
    adj=np.array([rng.choice(np.delete(np.arange(n),i),2,replace=False) for i in range(n)])
    # Sample physical scene observations once. A source may measure the same slot as its peer.
    msgs=[];arrivals=[]
    for src in (0,1):
        for tick in range(96):
            slot=int(rng.integers(shared));value=rng.normal(0,.3,12)
            idx=len(msgs);msgs.append((tick,src,slot,value))
            due=tick+int(rng.integers(0,10));arrivals.append((due,rng.random(),idx))
            if tick%11==3:arrivals.append((due+int(rng.integers(1,6)),rng.random(),idx))
    states={rule:x.copy() for rule in RULES};ref=x.copy()
    choices={rule:[-1]*shared for rule in RULES}
    seen=[set() for _ in range(shared)];last={};summary=[]
    for number,(arr,_,idx) in enumerate(sorted(arrivals),1):
        tick,src,slot,value=msgs[idx];seen[slot].add(idx)
        winner=max(seen[slot],key=lambda j:(msgs[j][0],msgs[j][1]))
        ref[slot]=msgs[winner][3];oracle=full(ref,adj,w)[-1][slot]@head
        for rule in RULES:
            current=choices[rule][slot]
            if rule=="arrival":accept=True
            elif rule=="per-source":accept=tick>last.get((src,slot),-1)
            else:
                def key(j):
                    if j<0:return (-1,-1)
                    t,s=msgs[j][:2]
                    return (t+(3*s if rule=="biased-clock" else 0),s)
                accept=key(idx)>key(current)
            if accept:
                choices[rule][slot]=idx;states[rule][slot]=value
            chosen=choices[rule][slot]
            wrong=chosen!=winner
            result=full(states[rule],adj,w)[-1][slot]@head
            dist=float(np.linalg.norm(result-oracle))
            if rule=="ordered":assert not wrong and dist<1e-10
            summary.append((seed,number,arr,tick,src,slot,rule,int(accept),int(wrong),dist))
        last[src,slot]=max(tick,last.get((src,slot),-1))
    return summary

def main():
    with np.load(ROOT/"data/learned_weights.npz") as z:w,head=z["layers"],z["readout"]
    rr=[]
    for seed in range(901,917):rr.extend(run(seed,w,head))
    with (ROOT/"data/shared_slot_packets.csv").open("w",newline="") as f:
        wr=csv.writer(f);wr.writerow(["seed","arrival_index","delivery_tick","capture_tick","source","slot","rule","write","wrong_winner","output_distance"]);wr.writerows(rr)
    with (ROOT/"data/shared_slot_summary.csv").open("w",newline="") as f:
        wr=csv.writer(f);wr.writerow(["rule","arrivals","writes","wrong_winner","numeric_mismatch","median_distance_when_mismatch","max_distance"])
        for name in RULES:
            rows=[r for r in rr if r[6]==name];delta=np.array([r[9] for r in rows]);bad=delta>1e-10
            wr.writerow([name,len(rows),sum(r[7] for r in rows),sum(r[8] for r in rows),int(bad.sum()),np.median(delta[bad]) if bad.any() else 0,delta.max()])
if __name__=="__main__":main()

