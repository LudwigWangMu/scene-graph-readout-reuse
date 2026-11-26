"""Independent random feature bursts on a 32 by 32 lattice."""
from experiment import *
def main():
    a=lattice(32);n=len(a);a[n-1,0]=512;a[512,-1]=n-1
    rng=np.random.default_rng(2903);w=rng.normal(0,.18,(DEPTH,2,WIDTH,WIDTH));rows=[]
    for k in [1,4,16,64,256,1024]:
        x=rng.normal(0,.2,(n+1,WIDTH));x[-1]=0;cache=full(x,a,w);stamps=np.zeros(n,dtype=int)
        for v in range(1,105):
            ids=rng.choice(n-1,k,replace=False);x[ids]+=rng.normal(0,.04,(k,WIDTH))
            def timed_dense():
                t=perf_counter_ns();r=full(x,a,w);return r,(perf_counter_ns()-t)/1000
            def timed_inc():
                t=perf_counter_ns();work=update(cache,a,w,ids,[],v,stamps);return work,(perf_counter_ns()-t)/1000
            if v%2:ref,tf=timed_dense();work,ti=timed_inc()
            else:work,ti=timed_inc();ref,tf=timed_dense()
            err=float(np.max(np.abs(ref[-1]-cache[-1])))
            if err>1e-10:raise AssertionError(err)
            if v>4:rows.append([k,v,tf,ti,work/(DEPTH*n),err])
    dump('density',['edited_nodes','version','full_us','incremental_us','work_fraction','max_error'],rows)
    for k in [1,4,16,64,256,1024]:
        rs=np.asarray([r for r in rows if r[0]==k]);print(k,np.median(rs[:,2:5],axis=0))
if __name__=='__main__':main()
