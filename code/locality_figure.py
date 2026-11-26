from experiment import *
def main():
    side=9; a=lattice(side);n=len(a);old=39;new=40;hand=n-1
    a[hand,0]=old;a[old,-1]=hand;edges=attach(a,old,new);dirty=np.array([hand])
    pos=np.array([(i%side,i//side) for i in range(side**2)]+[(new%side+.35,new//side+.55)])
    fig,axes=plt.subplots(1,3,figsize=(7,2.05))
    for k,ax in enumerate(axes,1):
        nb=a[dirty].ravel();dirty=np.unique(np.r_[dirty,nb[nb<n],edges])
        for i in range(n-1):
            for j in a[i,:4]:
                if i<j<n-1:ax.plot(*pos[[i,j]].T,color='#e1e6e8',lw=.65,zorder=0)
        ax.scatter(*pos[:-1].T,s=5,c='#acb7bd');ax.scatter(*pos[dirty].T,s=12,c='#23617d')
        ax.plot(*pos[[hand,old]].T,'--',color='#aa5a2b',lw=1.1);ax.plot(*pos[[hand,new]].T,c='#aa5a2b',lw=1.3)
        ax.scatter(*pos[hand],s=22,c='#b26024',marker='s')
        ax.set_title(f'Layer {k}  |  {len(dirty)} rows',fontsize=8);ax.set_aspect('equal');ax.axis('off')
    fig.tight_layout(pad=.5);fig.savefig(ROOT/'figures'/'locality.pdf',bbox_inches='tight');plt.close(fig)
if __name__=='__main__':main()
