import sys, json, concurrent.futures as cf
from rzip import RZip
date, d = sys.argv[1], int(sys.argv[2])
z = RZip(f'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/{date}_drive_{d:04d}/{date}_drive_{d:04d}_sync.zip')
names = sorted(k for k in z.ent if '/oxts/data/' in k and k.endswith('.txt'))
ts = z.read([k for k in z.ent if k.endswith('oxts/timestamps.txt')][0]).decode().split('\n')
with cf.ThreadPoolExecutor(6) as ex: rows = list(ex.map(lambda k: [float(v) for v in z.read(k).split()], names))
json.dump({"names": names, "ts": ts, "vf": [r[8] for r in rows], "wz": [r[19] for r in rows], "yaw": [r[5] for r in rows]}, open(f'oxts_{date}_{d:04d}.json', 'w'))
print(date, d, len(rows))
