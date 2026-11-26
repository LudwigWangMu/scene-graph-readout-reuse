"""Pack the public centroids used by the runtime and learned-readout audit."""
import argparse,urllib.request,hashlib,zipfile,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
URL='https://github.com/CEA-LIST/robot-watchdog-code/releases/download/v1.0.0/robot_watchdog_data.zip'
SHA='84e2d48bec79679a51cc5baa4f3baef07cbaf9357043852119db146d3d51193d'
def extract(path):
    if not path.exists():urllib.request.urlretrieve(URL,path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=SHA:raise ValueError('Archive checksum mismatch')
    tracks={}
    with zipfile.ZipFile(path) as z:
        for name in sorted(n for n in z.namelist() if n.endswith('expert.json')):
            task=name.split('/')[1]
            for i,episode in enumerate(json.loads(z.read(name))[:15]):
                pts=[np.asarray(v['points'],dtype=np.float64) for k,v in episode.items() if k.isdigit()]
                n=min(map(len,pts))
                tracks[f'{task}__{i:02d}']=np.stack([p[:n] for p in pts],axis=1)
    return tracks
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,default=Path('watchdog.zip'))
    np.savez_compressed(ROOT/'data'/'object_tracks.npz',**extract(ap.parse_args().archive))

