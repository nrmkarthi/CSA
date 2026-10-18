"""Convert the HAI CSV files (hai-21.03, hai-20.07) into cached .npz arrays used by all experiments."""
import pandas as pd, numpy as np, glob, os, sys
from common import ROOT, CACHE
R = os.environ.get('HAI_DATA', os.path.join(ROOT, 'data', 'hai'))   # folder containing hai-21.03/ and hai-20.07/
out = CACHE; os.makedirs(out, exist_ok=True)
for ver,sep in [('hai-21.03',','),('hai-20.07',';')]:
    for f in sorted(glob.glob(f'{R}/{ver}/*.csv.gz')):
        df=pd.read_csv(f,sep=sep)
        name=os.path.basename(f).replace('.csv.gz','')
        df.columns=[c.strip() for c in df.columns]
        tcol=df.columns[0]
        lab=[c for c in df.columns if c.startswith('attack')]
        feats=[c for c in df.columns if c not in lab and c!=tcol]
        X=df[feats].to_numpy(np.float32)
        L=df[lab].to_numpy(np.int8) if lab else np.zeros((len(df),4),np.int8)
        np.savez(f'{out}/{ver}_{name}.npz',X=X,L=L,feats=np.array(feats),labs=np.array(lab),t0=str(df[tcol].iloc[0]))
        if lab:
            a=L[:,0]; d=np.diff(np.r_[0,a,0]); st=np.where(d==1)[0]; en=np.where(d==-1)[0]
            per=[L[:,i].sum() for i in range(1,L.shape[1])]
            print(ver,name,len(df),'feats',len(feats),'episodes',len(st),'attack_frac',round(a.mean(),4),'perproc',per, 'durs',list(en-st)[:40])
        else: print(ver,name,len(df),len(feats))
