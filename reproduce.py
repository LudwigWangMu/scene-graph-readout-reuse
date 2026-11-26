"""Regenerate the archived numerical analyses and vector figures."""
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent
for name in ["experiment","tracked_objects","density_sweep","learned_readout","locality_figure","queued_runtime","release_sweep","queue_stage_figure","shared_slot_order"]:
    subprocess.run([sys.executable,str(root/'code'/(name+'.py'))],cwd=root,check=True)

