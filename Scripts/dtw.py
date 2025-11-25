# dtw.py — time warping and alignment

import numpy as np
from config import CANON_FRAMES

def time_warp_to_canon(X, L=CANON_FRAMES):
    T,N=X.shape; t_src=np.linspace(0,1,T); t_dst=np.linspace(0,1,L)
    Xc=np.zeros((L,N),float)
    for n in range(N): Xc[:,n]=np.interp(t_dst,t_src,X[:,n])
    mx=Xc.max(axis=1,keepdims=True); mx[mx<1e-12]=1.0
    return Xc/mx

def dtw_align_to_proto(X, P_canon):
    T1,N=X.shape; T2=P_canon.shape[0]
    C=np.zeros((T1,T2),np.float32)
    for i in range(T1):
        dif=(X[i]-P_canon)**2
        C[i]=np.sum(dif,axis=1)
    INF=1e18
    D=np.full((T1+1,T2+1),INF,np.float64); D[0,0]=0.0
    back=np.zeros((T1,T2),np.int8)
    for i in range(1,T1+1):
        for j in range(1,T2+1):
            choices=(D[i-1,j-1], D[i-1,j], D[i,j-1])
            k=int(np.argmin(choices))
            D[i,j]=C[i-1,j-1]+choices[k]; back[i-1,j-1]=k
    i,j=T1,T2; path=[]
    while i>0 and j>0:
        path.append((i-1,j-1)); k=back[i-1,j-1]
        if k==0: i-=1; j-=1
        elif k==1: i-=1
        else: j-=1
    path.reverse()
    buckets=[[] for _ in range(T2)]
    for ii,jj in path: buckets[jj].append(ii)
    Xa=np.zeros((T2,N),float)
    for jj in range(T2):
        if buckets[jj]: Xa[jj]=np.mean(X[buckets[jj]],axis=0)
        else: Xa[jj]=Xa[jj-1] if jj>0 else X[0]
    mx=Xa.max(axis=1,keepdims=True); mx[mx<1e-12]=1.0
    return Xa/mx
