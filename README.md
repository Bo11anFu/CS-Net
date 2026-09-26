# CSNet

Official PyTorch implementation of **CSNet: A Center-Surround Motion Saliency Integration Network for Moving Infrared Small Target Detection**.

## News

- **2026-09-01:** Our dataset, code, and trained model weights are released.

## Getting Started

### 1. Prerequisites

The reference environment targets Linux x86_64. The main package versions are:

| Package | Version |
| --- | --- |
| Python | 3.8.20 |
| CUDA | 11.3 |
| PyTorch | 1.10.1+cu113 |
| torchvision | 0.11.2+cu113 |
| Triton | 3.0.0 |
| NumPy | 1.21.0 |
| timm | 0.5.4 |
| einops | 0.4.1 |

See `requirements.txt` for the complete dependency list. Windows is not the reference platform; when using `triton-windows`, ensure that its bundled or system C compiler is discoverable through the `CC` environment variable.

### 2. Installation

**Step 1.** Create a conda environment and activate it.

```bash
conda create -n csnet python=3.8.20 -y
conda activate csnet
```

**Step 2.** Install PyTorch with CUDA support.

```bash
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

**Step 3.** Install the remaining dependencies.

```bash
pip install -r requirements.txt
```

## Basic Usage

Update the dataset roots, experiment output paths, and checkpoint paths in the selected YAML file before running the commands below. Please refer to the configuration files in `configs/` for detailed training settings. The reported training setup used two 24 GB GPUs. The released MIST and NUDT-MIRSDT configurations both use a total batch size of 12.

### 1. Training

```bash
# Train on MIST
python train.py --config ./configs/multiframe/CSNet/train_CSNet_MIST.yaml

# Train on NUDT-MIRSDT
python train.py --config ./configs/multiframe/CSNet/train_CSNet_NUDTMIRSDT.yaml
```

### 2. Test

Our trained model weights are available at [[Google Drive](https://drive.google.com/drive/folders/1G8irLttLT2DLg1cfqA-IIrwN78aWHRhG?usp=drive_link)]. Download the corresponding weights and update `test.checkpoint` in the selected configuration file before evaluation.

```bash
# Test on MIST
python test.py --config ./configs/multiframe/CSNet/test_CSNet_MIST.yaml

# Test on NUDT-MIRSDT
python test.py --config ./configs/multiframe/CSNet/test_CSNet_NUDTMIRSDT.yaml
```
