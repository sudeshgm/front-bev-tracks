"""iter10: attribute missing main edges at z = 10 m to trim sub-rules (iter9 polylines, edges_raw INTERPOLATED - superseded by trim_gap_attr.py which keeps the raw holes)."""
import json,sys,numpy as np
import os; sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','src')); import lines_mt as LM
Z=LM.ZS; i10=int(np.argmin(abs(Z-10)))
def xg(p):
    p=np.asarray(p,float);o=np.full(len(Z),np.nan)
    if p.ndim==2 and len(p)>=2:
        s=np.argsort(p[:,1]);p=p[s];m=(Z>=p[0,1]-1e-6)&(Z<=p[-1,1]+1e-6);o[m]=np.interp(Z[m],p[:,1],p[:,0])
    return o
def attr(x,side,lanes):
    f=np.isfinite(x); tol=0.5
    cr=np.zeros(len(Z),bool)
    for r in lanes:
        lx=r['xg']; both=f&np.isfinite(lx); idx=np.flatnonzero(both)
        if len(idx)<6 or np.median((x[idx[:4]]-lx[idx[:4]])*side)<=0: continue
        c=both&((x-lx)*side<-tol); ci=np.flatnonzero(c)
        if len(ci)>=2: cr[ci[0]:]=True
        else: cr|=c
    eg=f&(x*side<LM.EGO_HALF_M)
    la=[v for v in (LM._near_angle(r['xg']) for r in lanes) if v is not None]
    ref=np.radians(float(np.median(la))) if la else 0.0
    xs=np.where(f,x,np.interp(Z,Z[f],x[f])); dev=np.degrees(np.arctan(np.gradient(xs,Z))-ref)*side
    near=f&(Z<=20); cv=near&(dev<0)&(-dev>LM.EDGE_CONV_DEG); dv=near&(dev>0)&(dev>LM.EDGE_DIV_DEG)
    return cr,eg,cv,dv,dev
if __name__=='__main__':
    clip=sys.argv[1]; role=sys.argv[2]; side=1 if role=='main_R' else -1
    d=json.load(open(os.environ.get('RUN','/workspace/foundation_bev/iter9')+f'/polylines/{clip}_polylines.json'))
    cats={}; rows=[]
    for F in d['frames']:
        fr=F['frame']; raw=F.get('edges_raw',{}).get(role)
        fin=[xg(p) for r,p in zip(F['edges'],F['edges_bev']) if r.get('role')==role]
        has=bool(fin) and np.isfinite(fin[0][i10])
        if has: continue
        if raw is None: cats.setdefault('noraw',[]).append(fr); continue
        x=xg(raw)
        if not np.isfinite(x[i10]): cats.setdefault('raw_not_at10',[]).append(fr); continue
        lanes=[dict(r,xg=xg(p)) for r,p in zip(F['lanes'],F['lanes_bev'])]
        cr,eg,cv,dv,dev=attr(x,side,lanes)
        t=LM.trim_edge(x,side,lanes)
        if t is not None and np.isfinite(t[i10]): k='replay_keeps(lanes/hold differ)'
        else:
            k='+'.join(n for n,m in (('cross',cr),('ego',eg),('conv',cv),('div',dv)) if m[i10]) or ('norun' if t is None else 'nearest_run_before')
        cats.setdefault(k,[]).append(fr)
        rows.append((fr,k,round(float(dev[i10]),1),round(float(x[i10]),2)))
    for k,v in sorted(cats.items(),key=lambda kv:-len(kv[1])): print(k,len(v),v[:60])
    if '-v' in sys.argv:
        for r in rows: print(r)
