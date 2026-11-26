"""Regenerate compact, derived operator-motion traces from pinned LIBERO-10 records."""
import argparse,csv,hashlib,json,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
REV='551d7d86f25edd0ffeda8b60053c15438cfd1d6a'
REPO='lerobot/libero_10'
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache',type=Path,default=Path('libero-cache'))
    cache=parser.parse_args().cache;cache.mkdir(exist_ok=True)
    base='https://huggingface.co'
    entries=json.load(urllib.request.urlopen(f'{base}/api/datasets/{REPO}/tree/{REV}/data/chunk-000'))
    def fetch(e):
        path=cache/Path(e['path']).name
        if not path.exists():urllib.request.urlretrieve(f"{base}/datasets/{REPO}/resolve/{REV}/{e['path']}",path)
        if path.stat().st_size!=e['size']:raise IOError(path)
        return path
    with ThreadPoolExecutor(4) as pool: files=list(pool.map(fetch,entries))
    groups={}
    for path in sorted(files):
        for row in pq.read_table(path,columns=['episode_index','frame_index','task_index','observation.state']).to_pylist():
            key=(row['task_index'],row['episode_index'])
            if row['frame_index']<128 and row['frame_index']%2==0:
                groups.setdefault(key,[]).append((row['frame_index'],row['observation.state']))
    selected=[]
    for task in sorted({k[0] for k in groups}):
        selected += sorted(k for k in groups if k[0]==task and len(groups[k])==64)[:2]
    with (ROOT/'data'/'trace.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['task','episode','frame','x','y','z','qx','qy','qz','qw','gripper'])
        for task,episode in selected:
            for frame,obs in sorted(groups[(task,episode)]):writer.writerow([task,episode,frame,*obs])
    print(len(files),'shards;',len(selected),'episodes',selected)
if __name__=='__main__':main()
