"""Benchmark a resident dense learned readout with host input and returned host output."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,csv,ctypes,json,platform,time
import numpy as np
import torch
from queued_runtime import ROOT,features,graph
def data():
    z=np.load(ROOT/'data/object_tracks.npz');wt=np.load(ROOT/'data/learned_weights.npz');items=[]
    for key in sorted(z):
        if int(key.rsplit('__',1)[1])<10:continue
        track=z[key]
        for t in np.linspace(6,len(track)-5,48,dtype=int):
            x=features(track[t-6:t+1]).astype('f4');a=graph(track[t]);n=len(x)
            mask=np.zeros((n,n),dtype='f4');i,j=np.where(a<n);mask[i,a[i,j]]=1
            mask/=np.maximum(1,mask.sum(1,keepdims=True))
            items.append((key,int(t),x,mask))
    return items,wt['layers'].astype('f4'),wt['readout'].astype('f4')
class Numpy:
    def __init__(self,n,w,read):
        self.w=w;self.read=read;self.m=np.empty((n,12),'f4');self.h=self.m.copy();self.a=self.m.copy();self.out=np.empty((n,2),'f4')
    def __call__(self,x,adj):
        h=x
        for w0,w1 in self.w:
            np.matmul(adj,h,out=self.m);np.matmul(h,w0,out=self.a);np.matmul(self.m,w1,out=self.h)
            np.add(self.a,self.h,out=self.h);np.tanh(self.h,out=self.h);h=self.h
        return np.matmul(h,self.read,out=self.out)
class Captured:
    def __init__(self,n,w,read):
        self.x=torch.zeros(n,12,device='cuda');self.adj=torch.zeros(n,n,device='cuda')
        self.wt=torch.from_numpy(w).cuda();self.head=torch.from_numpy(read).cuda()
        def forward():
            h=self.x
            for w0,w1 in self.wt:h=torch.tanh(h@w0+(self.adj@h)@w1)
            return h@self.head
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(4):self.out=forward()
        torch.cuda.current_stream().wait_stream(stream);self.g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):self.out=forward()
    def __call__(self,x,adj):
        self.x.copy_(torch.from_numpy(x));self.adj.copy_(torch.from_numpy(adj))
        self.g.replay();return self.out.cpu().numpy()
class Energy:
    def __init__(self):
        self.lib=ctypes.WinDLL('nvml.dll') if os.name=='nt' else ctypes.CDLL('libnvidia-ml.so.1')
        def check(ret):
            if ret:raise RuntimeError('NVML status '+str(ret))
        self.check=check;check(self.lib.nvmlInit_v2());self.h=ctypes.c_void_p()
        check(self.lib.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(self.h)))
    def joules(self):
        v=ctypes.c_ulonglong();self.check(self.lib.nvmlDeviceGetTotalEnergyConsumption(self.h,ctypes.byref(v)));return v.value/1000
def write(name,header,rows):
    with (ROOT/'data'/name).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)
def main():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    pts,w,h=data();ns=sorted({len(r[2]) for r in pts})
    cpu={n:Numpy(n,w,h) for n in ns};gpu={n:Captured(n,w,h) for n in ns};eng=Energy()
    for _,_,x,adj in pts[:24]:gpu[len(x)](x,adj)
    raw=[];diff=0.
    for rep in range(2):
        for j,(track,t,x,adj) in enumerate(pts):
            results={}
            for name in (['cpu','cuda'] if (j+rep)%2==0 else ['cuda','cpu']):
                runner=(cpu if name=='cpu' else gpu)[len(x)]
                ts=time.perf_counter_ns();y=runner(x,adj);ns1=time.perf_counter_ns()-ts
                results[name]=y.copy();raw.append([track,t,rep,name,ns1])
            diff=max(diff,float(np.max(abs(results['cpu']-results['cuda']))))
    if diff>2e-5:raise AssertionError(diff)
    records=[]
    for rep,order in enumerate([['idle','cpu','cuda'],['cuda','idle','cpu'],['cpu','cuda','idle']]):
        for mode in order:
            t=time.perf_counter();e0=eng.joules();k=0
            if mode=='idle':time.sleep(3.)
            else:
                runners=cpu if mode=='cpu' else gpu
                while time.perf_counter()-t<3.:
                    _,_,x,adj=pts[k%len(pts)];runners[len(x)](x,adj);k+=1
            e1=eng.joules();elapsed=time.perf_counter()-t
            records.append([rep,mode,k,elapsed,e1-e0,(e1-e0)/elapsed])
    write('device_readouts.csv',['track','frame','repeat','device','host_input_to_result_ns'],raw)
    write('board_energy.csv',['repeat','mode','completions','duration_s','board_j','mean_board_w'],records)
    stat={}
    for dev in ['cpu','cuda']:
        us=np.array([r[-1]/1000 for r in raw if r[3]==dev]);stat[dev]=np.quantile(us,[.5,.95,.99]).tolist()
    powers={m:sum(r[4] for r in records if r[1]==m)/sum(r[3] for r in records if r[1]==m) for m in ['idle','cpu','cuda']}
    out=dict(quantiles_us=stat,mean_board_w=powers,samples_per_device=len(pts)*2,max_output_difference_px=diff,gpu=torch.cuda.get_device_name(),torch=torch.__version__,driver_note='NVML device 0; float32; TF32 disabled; whole discrete board',platform=platform.platform())
    (ROOT/'data/device_summary.json').write_text(json.dumps(out,indent=2))
    print(out)
if __name__=='__main__':main()

