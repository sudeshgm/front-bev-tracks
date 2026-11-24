"""iter10 replay of the edge gap bridging: trim_edge on the iter9 GPU edges_raw dump with / without bridge_gaps;
z = 10 m presence gained / lost per clip and role (final presence estimated as iter9 final + gained frames).
usage: RUN=/workspace/foundation_bev/iter9 OUT=replay/replay_gap.json replay_gap.py"""
import json,sys,os,numpy as np
from trim_gap_attr import grid,Z,i10,LM,xg
CL=['kitti_lane_split_candidate','kitti_highway_exit_approach','kitti_highway_sample','kitti_highway_late','kitti_highway_final','kitti_city_cyclist_candidate','munich','a2d2a','a2d2b']
res={}
for clip in CL:
    p=os.environ.get('RUN','/workspace/foundation_bev/iter9')+f'/polylines/{clip}_polylines.json'
    if not os.path.exists(p): continue
    d=json.load(open(p)); n=len(d['frames'])
    for role,side in (('main_L',-1),('main_R',1)):
        old=new=fin=gain=loss=0; flr=[]; gl=[]
        for F in d['frames']:
            fr=F['frame']
            f_=[xg(pp) for r,pp in zip(F['edges'],F['edges_bev']) if r.get('role')==role]
            hasf=bool(f_) and np.isfinite(f_[0][i10]); fin+=hasf
            raw=F.get('edges_raw',{}).get(role)
            if raw is None: continue
            x=grid(raw); lanes=[{'xg':xg(pp)} for pp in F['lanes_bev']]
            g_=LM.EDGE_GAP_M; LM.EDGE_GAP_M=0.0; o=LM.trim_edge(x,side,lanes); LM.EDGE_GAP_M=g_
            t=LM.trim_edge(x,side,lanes)
            ok_o=o is not None and np.isfinite(o[i10]); ok_n=t is not None and np.isfinite(t[i10])
            old+=ok_o; new+=ok_n
            if ok_n and not ok_o:
                gain+=1; gl.append(fr)
                if not hasf and LM.is_flared(t,side,lanes): flr.append(fr)
            if ok_o and not ok_n: loss+=1
        newfin=fin+sum(1 for F in d['frames'] if F['frame'] in gl and not any(np.isfinite(xg(pp)[i10]) for r,pp in zip(F['edges'],F['edges_bev']) if r.get('role')==role))
        res[(clip,role)]=dict(n=n,final=round(100*fin/n,1),final_est=round(100*newfin/n,1),trim_old=old,trim_new=new,gain=gain,loss=loss,flared_gain=flr,gain_frames=gl)
        print(clip[:24],role,'final %.1f -> est %.1f'%(100*fin/n,100*newfin/n),'trim z10 %d->%d gain %d loss %d flared %s'%(old,new,gain,loss,flr[:10]),gl[:40])
json.dump({f'{a}|{b}':v for (a,b),v in res.items()},open(os.environ.get('OUT','replay_gap.json'),'w'),indent=1,default=int)
