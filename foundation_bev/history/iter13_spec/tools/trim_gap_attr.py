"""iter10: why is the final main edge missing at z = 10 m although edges_raw has it? raw placed on the grid WITH its
holes (noraw / raw_starts_beyond10 / raw_nan_at10_gap / short_run_gap / div / conv / cross / ego).
usage: RUN=<run dir> trim_gap_attr.py munich main_R"""
import os
import json,sys,numpy as np
from trimattr import xg,attr,Z,i10,LM
def grid(pts):
    o=np.full(len(Z),np.nan)
    for x,z in pts:
        i=int(round((z-Z[0])/(Z[1]-Z[0])))
        if 0<=i<len(Z): o[i]=x
    return o
def gaps(x):
    f=np.isfinite(x); idx=np.flatnonzero(f); g=[]
    for a,b in zip(idx[:-1],idx[1:]):
        if b-a>1: g.append((Z[a],Z[b]))
    return g
if __name__=='__main__':
    clip,role=sys.argv[1],sys.argv[2]; side=1 if role=='main_R' else -1
    d=json.load(open(os.environ.get('RUN','/workspace/foundation_bev/iter9')+f'/polylines/{clip}_polylines.json'))
    cats={}
    for F in d['frames']:
        fr=F['frame']; raw=F.get('edges_raw',{}).get(role)
        fin=[xg(p) for r,p in zip(F['edges'],F['edges_bev']) if r.get('role')==role]
        if fin and np.isfinite(fin[0][i10]): continue
        if raw is None: cats.setdefault('noraw',[]).append(fr); continue
        x=grid(raw)
        lanes=[{'xg':xg(p)} for p in F['lanes_bev']]
        t=LM.trim_edge(x,side,lanes)
        if t is not None and np.isfinite(t[i10]): cats.setdefault('trim_keeps_now',[]).append(fr); continue
        if not np.isfinite(x[i10]):
            g=gaps(x); k='raw_nan_at10_gap' if np.isfinite(x[:i10]).any() else 'raw_starts_beyond10'
            cats.setdefault(k,[]).append(fr); continue
        cr,eg,cv,dv,dev=attr(np.where(np.isfinite(x),x,np.nan),side,lanes)
        k='+'.join(n for n,m in (('cross',cr),('ego',eg),('conv',cv),('div',dv)) if m[i10]) or 'short_run_gap'
        cats.setdefault(k,[]).append(fr)
    for k,v in cats.items(): print(k,len(v),v)
