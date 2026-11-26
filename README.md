# Reusing Scene-Graph Readouts Across Asynchronous Robot Updates
  

## Verify the published results

Download and extract the [v1.0.0 release](https://github.com/LudwigWangMu/scene-graph-readout-reuse/releases/tag/v1.0.0), then run the following command in the extracted directory using Python 3.12. It checks the distributed file hashes and recomputes selected results from the archived observation-level records. The check runs offline with the Python standard library.

    python verify.py

For a full rerun, use a fresh working copy and follow the instructions below. After regenerating outputs, use `python verify.py --results-only` to check the numerical summaries.

## CPU replay

From this directory, use a fresh Python 3.12 environment.

```sh
python -m pip install -r requirements.txt
python code/prepare_trace.py
python reproduce.py
```

Trace preparation extracts eight state components from 20 LIBERO-10 episodes at a pinned revision. Use `--cache PATH` to reuse previously downloaded parquet shards. The timing experiments use the first ten nominal tracks from each Robot-WATCHDOG task and test the fitted consumer on the next five. To rebuild the distributed centroid array from the original v1.0.0 archive, run `python code/prepare_objects.py`. [LIBERO-10 release](https://huggingface.co/datasets/lerobot/libero_10/tree/551d7d86f25edd0ffeda8b60053c15438cfd1d6a) · [Robot-WATCHDOG data v1.0.0](https://github.com/CEA-LIST/robot-watchdog-code/releases/tag/v1.0.0)

`data/object_tracks.npz` is a reformatted subset of centroid trajectories provided by Quentin Rolland, Fabrice Mayran de Chamisso and Jean-Baptiste Mouret. The selected objects are aligned by frame and packed as arrays for 15 nominal episodes per task. The authors distribute those data under CC BY 4.0. [Dataset description and authors](https://github.com/CEA-LIST/robot-watchdog-code#data) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

| Measurement | Files |
| --- | --- |
| Queued host packets and directed release checks | `data/queued_packets.csv`, `data/directed_release.csv`, `data/queued_summary.json` |
| Shared-slot ordering and independent recomputation | `data/shared_slot_packets.csv`, `data/shared_slot_summary.csv` |
| Scoped and global guard sweeps | `data/release_sweep.csv`, `data/release_sweep_summary.csv` |
| Device latency and discrete-board energy | `data/device_readouts.csv`, `data/board_energy.csv`, `data/device_summary.json` |

## GPU measurement

Run `python code/device_readout.py` in a separate CUDA-enabled environment with the CPU dependencies and a compatible PyTorch build. The original Windows host used an RTX 5060 Laptop GPU, PyTorch 2.7.1+cu128 and the NVIDIA NVML board-energy counter. Local reruns produce new timings and board-energy readings. Logical release decisions can be compared across machines. The root verifier reconstructs the shared-slot error counts and the number of stale directed releases from the archived packets.
