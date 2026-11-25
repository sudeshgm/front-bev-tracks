"""iter15: find which KITTI raw drive / frame a public KITTI clip comes from. Clip frames (first / middle / last)
-> 64x48 normalised grey thumbnails; KITTI raw image_02 frames are read by HTTP range requests from the public S3 sync
zips (rzip.py; nothing is downloaded in full). An exact match scores NCC 1.000.
usage: python3 match.py DATE:DRIVE:START:STOP:STEP [...]   (e.g. 2011_10_03:42:0:1200:5)"""
import cv2, numpy as np, concurrent.futures as cf, json, sys
from rzip import RZip
import os
DS=os.environ.get('FBEV_PUB_DS', '/workspace/gpubot/foundation-bev-23nov/ds')
clips=['kitti_highway_sample','kitti_highway_exit_approach','kitti_highway_late','kitti_highway_final','kitti_city_cyclist_candidate','kitti_lane_split_candidate']
def th(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
    t=cv2.resize(g,(64,48),interpolation=cv2.INTER_AREA).astype(np.float32); t-=t.mean(); return t/ (t.std()+1e-6)
Q={}
for c in clips:
    cap=cv2.VideoCapture(f'{DS}/{c}.mp4'); fr=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        fr.append(f)
    print(c,len(fr),fr[0].shape,flush=True)
    for k in (0,len(fr)//2,len(fr)-1): Q[(c,k)]=th(fr[k])

jobs=[a.split(':') for a in sys.argv[1:]]  # date:drive:start:stop:step
def scan(j):
    date,d,a,b,st=j[0],int(j[1]),int(j[2]),int(j[3]),int(j[4])
    try:
        z=RZip(url(date,d)); ims=sorted(k for k in z.ent if '/image_02/data/' in k and k.endswith('.png'))
    except Exception as e: return j,str(e)
    out=[]
    def one(i):
        t=th(cv2.imdecode(np.frombuffer(z.read(ims[i]),np.uint8),cv2.IMREAD_GRAYSCALE))
        return i,{f'{c}|{k}':float((t*q).mean()) for (c,k),q in Q.items()}
    with cf.ThreadPoolExecutor(8) as ex: out=list(ex.map(one,range(a,min(b,len(ims)),st)))
    return j+[len(ims)],out
best={}
for j,out in map(scan,jobs):
    if isinstance(out,str): print('ERR',j,out,flush=True); continue
    for i,s in out:
        for key,v in s.items():
            if v>best.get(key,(-9,))[0]: best[key]=(v,f'{j[0]}_{j[1]}',i)
    print('done',j,flush=True)
for k in sorted(best): print(k,'ncc %.3f %s frame %d'%best[k])
