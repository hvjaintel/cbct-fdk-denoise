# System Requirements

## Software

| Component | Notes |
|-----------|--------|
| Python 3.10+ | 3.11 recommended |
| Node.js 18+ | Only for building the React UI |
| Docker / Compose | Optional; for container path |
| OpenVINO runtime | Via `pip install openvino` |

## Hardware

| Target | Guidance |
|--------|----------|
| CPU | Any modern x86_64; default demo sizes finish in seconds–minutes |
| Intel iGPU / GPU | Optional for OpenVINO stub denoise (`--device GPU`) |
| RAM | ≥ 8 GB recommended |

## Demo sizing

Default dashboard settings (`48³` volume, 60 angles, `64²` detector) are chosen for interactive demos. Larger sizes increase FDK runtime roughly with angles × volume voxels.
