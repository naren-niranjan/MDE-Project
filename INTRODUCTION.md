# MDE-Project

Code archive for the MSc thesis **"Multi-Monocular Camera Depth Estimation Using Vision Foundation Models for Near Real-Time 3D Perception on Embedded Hardware"**.

Naren Niranjan Pattanam Ravikumar, MSc Robotics, University of Twente.
Graduation project carried out at AWL-Techniek B.V., Harderwijk, the Netherlands.
Academic supervisor: dr.ir. Soheil Arastehfar (UT). Industrial supervisor: Erik Buit (AWL).

The pipeline runs Depth Anything 3 (DA3) across four Lucid Triton GigE cameras mounted above a conveyor in a FANUC robot cell and produces a fused metric depth map for parcel pick guidance on an NVIDIA Jetson AGX Thor. The thesis investigates three sub-questions: which vision foundation model to use (SRQ1), the minimum viable camera count and arrangement (SRQ2), and embedded performance on the Jetson (SRQ3).

This repository contains the scripts, configuration and calibration exports needed to reproduce the experiments. Image captures, ground-truth point clouds and benchmark outputs (about 200 GB) are not stored here; see [Data](#data).

## Repository layout

The folders reflect the chronological development of the work. The final pipeline used for the thesis results lives in `MDE/new/`; earlier folders are kept for traceability.

| Path | Contents |
|---|---|
| `MDE/new/` | **Final pipeline.** DA3 streaming and offline inference, camera-subset sweeps and grading, ground-truth capture and alignment, depth correction fitting, per-stage benchmarking. |
| `MDE/DA3*/`, `MDE/stream*/`, `MDE/improve/` | Earlier DA3 iterations (single-view, multi-view, streaming) leading to `MDE/new/`. |
| `MDE/Moge2/`, `MultiCamera/` | MoGe-2 experiments used in the SRQ1 model comparison. |
| `Calibration*/`, `SingleCameraCalib/` | ChArUco intrinsic and extrinsic calibration. `Calibration5/` is the version used for the final results (`auto_capture.py`, `calibrate_intrinsics.py`, `calibrate_extrinsics.py`, `config.py`). |
| `Image Capture*/`, `Image_Capture_4_*/` | Arena SDK capture scripts for the Lucid Triton cameras. |
| `Comparison/` | Model comparison scripts and tables. |
| `ArenaSDK/README_ARM64` | Notes on installing the Lucid Arena SDK on ARM64 (the SDK tarballs themselves are not included). |

Key scripts in `MDE/new/`:

| Script | Purpose |
|---|---|
| `da3_stream.py` | Live four-camera capture and DA3 inference with per-stage timing. |
| `da3_offline.py` | DA3 inference on saved captures. |
| `sweep_subsets.py`, `grade_subsets.py` | Run and score every camera subset for the SRQ2 camera-count study. |
| `capture_gt.py`, `board_gt.py`, `depth_align.py`, `pin_alignment.py` | Ground-truth capture (Roboception rc_viscore) and alignment to the camera rig frame. |
| `fit_correction.py`, `depth_correction.json` | Fit and store the linear metric depth correction. |
| `bench_pipeline.py` | Per-stage processing time, rate and peak memory on the Jetson (SRQ3). |
| `box_scene.py`, `box_segment.py`, `box_geometry.py`, `face_consensus.py` | Parcel top-face extraction from the fused depth. |
| `snap4.py` | Synchronised four-camera snapshot. |
| `persist.json` | Camera serials, resolution and streaming parameters. |

## Hardware

- NVIDIA Jetson AGX Thor Developer Kit (JetPack 7, CUDA 13, 128 GB unified memory)
- 4 × Lucid Triton TRI050S-C GigE cameras with 12 mm Edmund Optics lenses, connected through a PoE switch
- Roboception rc_viscore stereo sensor (ground truth only)
- ChArUco calibration board

Development and offline evaluation were also done on a Windows 11 laptop (RTX A2000 8 GB, WSL2).

## Software setup

### 1. Arena SDK (camera access)

Download the Arena SDK for Linux ARM64 from [Lucid Vision Labs](https://thinklucid.com/downloads-hub/) and follow `ArenaSDK/README_ARM64`. Install the Python bindings (`arena_api`) into the environment you create below.

### 2. Python environments

Two environments were used. Frozen package lists are in the repository.

```bash
# Depth Anything 3
python3 -m venv ~/venvs/da3
source ~/venvs/da3/bin/activate
pip install -r MDE/DA3/requirements_da3_new.txt

# MoGe-2 (only for the SRQ1 comparison)
python3 -m venv ~/venvs/moge2
source ~/venvs/moge2/bin/activate
pip install -r MDE/Moge2/requirements_moge2.txt
```

On the Jetson, PyTorch must be the NVIDIA JetPack build; install it from the NVIDIA Jetson PyPI index before the remaining requirements.

### 3. Models

Clone the upstream repositories next to the scripts that use them; they are not vendored here.

```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3 MDE/DA3/depth-anything-3
git clone https://github.com/microsoft/MoGe MDE/Moge2/MoGe
```

The DA3 checkpoint used throughout is `depth-anything/da3nested-giant-large-1.1`, downloaded automatically from Hugging Face on first run.

## Reproducing the pipeline

All commands are run from the repository root with the `da3` environment active.

1. **Calibrate the cameras.** Capture ChArUco views with `Calibration5/auto_capture.py`, then run `calibrate_intrinsics.py` followed by `calibrate_extrinsics.py`. Board geometry and camera serials are set in `Calibration5/config.py`. Results are written to `Calibration5/results/`. Hardware-triggered, synchronised capture is required; unsynchronised frames were the cause of the poor extrinsic residuals seen in earlier calibration folders.

2. **Capture a scene.** `python MDE/new/snap4.py` saves one synchronised frame per camera into `MDE/new/captures/`.

3. **Run depth inference.** `python MDE/new/da3_offline.py --captures MDE/new/captures --out MDE/new/runs_da3` for saved frames, or `python MDE/new/da3_stream.py` for live operation. Always pass the calibrated intrinsics *and* the extrinsic pose prior; DA3 ignores supplied intrinsics when no pose prior is given.

4. **Apply the metric correction.** `depth_correction.json` holds the fitted scale and offset. The correction is valid only within the height band above the conveyor deck in which it was fitted; clamp outside that range.

5. **Evaluate.** `sweep_subsets.py` and `grade_subsets.py` reproduce the camera-count study; `bench_pipeline.py` reproduces the timing and memory figures. Each run writes a diagnostic JSON alongside its outputs.

Run any script with `--help` for its full argument list.

## Data

Raw captures, ground-truth point clouds, calibration image sets and benchmark outputs total roughly 200 GB and are excluded via `.gitignore`. They are archived at AWL-Techniek and can be made available on request, subject to AWL approval, by contacting the author.

## Citation

If you use this code, please cite the thesis:

> N. N. Pattanam Ravikumar, "Multi-Monocular Camera Depth Estimation Using Vision Foundation Models for Near Real-Time 3D Perception on Embedded Hardware," MSc thesis, University of Twente, Enschede, 2026.

## Acknowledgements

Depth Anything 3 (ByteDance Seed) and MoGe-2 (Microsoft Research) are used under their respective licences. Camera access uses the Lucid Vision Labs Arena SDK.

## Contact

Naren Niranjan, awlniranjan@gmail.com
