"""Track-by-track medians and additive timing components on the observed host."""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def records(name):
    with (ROOT/"data"/name).open(newline="") as f:return list(csv.DictReader(f))
def main():
    stage=records("queue_stages.csv")
    raw=[r for r in records("queued_packets.csv") if int(r["frame"])>=4]
    tracks=sorted({r["track"] for r in raw})
    mat=np.array([[np.median([float(r["injection_to_ack_ns"])/1000 for r in raw if r["track"]==t and r["mode"]==m]) for t in tracks] for m in ["full","local"]])
    fig,(left,right)=plt.subplots(1,2,figsize=(7,2.65),gridspec_kw={"width_ratios":[1.1,1]})
    for i in range(len(tracks)):left.plot([0,1],mat[:,i],lw=.8,alpha=.52,color="#5d7985")
    left.scatter(np.zeros(len(tracks)),mat[0],c="#166579",s=9);left.scatter(np.ones(len(tracks)),mat[1],c="#a95228",s=9)
    left.set(xlim=(-.22,1.22),xticks=[0,1],xticklabels=["Full","Local"],ylabel="Track median (microseconds)")
    keys=["input","graph","decoder","sink"];cols=["#89a8ac","#19657c","#b48a48","#704e78"]
    labels=["Input queue","Graph","Readout","Sink"];loc=np.arange(4);base=np.zeros(4)
    for k,co,lab in zip(keys,cols,labels):
        value=np.array([float(r[k])*1000 for r in stage])
        right.barh(loc,value,left=base,color=co,label=lab);base+=value
    right.set(yticks=loc,yticklabels=["Full / all","Full / tail","Local / all","Local / tail"],xlabel="Mean time (microseconds)");right.invert_yaxis()
    right.legend(ncol=2,fontsize=6,frameon=False,loc="lower right",bbox_to_anchor=(1,1.0))
    for ax in [left,right]:
        ax.tick_params(labelsize=7);ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout();fig.savefig(ROOT/"figures/queue_stage_attribution.pdf",bbox_inches="tight");plt.close(fig)
if __name__=="__main__":main()

