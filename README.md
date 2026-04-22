# GenDA: Global 3D Ocean Data Assimilation with Conformalized Generative Models

Official code repository for the paper: *"Global 3D Ocean Data Assimilation with Conformalized Generative Models"*

> GenDA is a probabilistic framework that combines **conditional flow matching** with **conformal prediction** to reconstruct global three-dimensional ocean temperature, salinity, and velocity fields from sparse Argo float profiles.


## Requirements

- Python >= 3.13
- PyTorch >= 2.0
- xarray, numpy, pandas, matplotlib, cartopy
- copernicusmarine (for data download)
- argopy (for Argo data processing)

1. Create and activate the conda environment:

```bash
conda create -n genda python=3.13 -y
conda activate genda
```

2. Install PyTorch (please visit [pytorch.org](https://pytorch.org) to select the appropriate version for your CUDA/driver setup).

3. Install remaining dependencies:

```bash
pip install pyyaml tensorboard pandas xarray netcdf4 tqdm einops matplotlib cartopy
```

## Data Preparation

The dataset used in this project can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1nb9PitgdqfqfEDaFVefGgf3HK8v4PZ14?usp=drive_link).

```bash
gdown --folder 1nb9PitgdqfqfEDaFVefGgf3HK8v4PZ14
```

> **Note**: Due to Google Drive download quota limits, you may encounter the following error when using `gdown`:
> ```
> Too many users have viewed or downloaded this file recently. Please
> try accessing the file again later. If the file you are trying to
> access is particularly large or is shared with many people, it may
> take up to 24 hours to be able to view or download the file.
> ```
> If this happens, please try again later or download the files manually via a browser.

### Data Directory Structure

After downloading and extracting, the data directory should be organized as follows:

```
data/
├── argo/
│   └── argo_profiles_by_date/           # 5127 daily profile files
│       ├── argo_profiles_2011-01-01.nc
│       ├── argo_profiles_2011-01-02.nc
│       └── ...                           # argo_profiles_YYYY-MM-DD.nc
└── glorys12/
    ├── train/                           # Training set (160720 files)
    │   ├── 20110101_d00.nc
    │   ├── 20110101_d01.nc
    │   └── ...                           # YYYYMMDD_dXX.nc
    ├── cal/                             # Calibration set (14600 files)
    │   ├── 20220101_d00.nc
    │   └── ...                           # YYYYMMDD_dXX.nc
    ├── test/                            # Test set (14600 files)
    │   ├── 20230101_d00.nc
    │   └── ...                           # YYYYMMDD_dXX.nc
    └── statistics_train.json            # Training statistics
```

- **GLORYS12**: Each `.nc` file (`YYYYMMDD_dXX.nc`) contains ocean state variables (temperature, salinity, velocity) for one ensemble member on one date. `dXX` denotes the ensemble member index.
- **Argo**: `argo_profiles_by_date/` contains daily merged Argo float profiles in NetCDF format (`argo_profiles_YYYY-MM-DD.nc`).

## Quick Start

### Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 main.py --mode train
```

### Sampling

Generate ensemble predictions using a trained model:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 main.py --mode sample
```

## Configuration

All configurations are in `configs/configs.yaml`. Key settings include:

- **Model architecture**: `DANet-L/16`, image size, dropout rates
- **Training hyperparameters**: Learning rate, batch size, epochs, EMA decay
- **Flow matching**: Noise scale, prediction parameterization (x-prediction), ODE solver settings
- **Sampling parameters**: Sampling method (`heun`), steps (50), ensemble size (10)
- **Data path**: Path to GLORYS12 dataset and Argo profiles
- **Checkpointing**: Output directory, resume path, save frequency
- **Argo Fusion**: 
  - `argo_fusion_method`: Controls how Argo data is fused into the model
    - `'geo_fusion'`: Location-aware attention — fuses Argo profiles into spatially corresponding patches based on geographic coordinates via scatter-add operations (O(N) w.r.t. profiles, recommended)
    - `'crossattention'`: Standard cross-attention mechanism (O(N·P) complexity, baseline)
- **Classifier-Free Guidance**: CFG scale and dropout probability for conditional generation

The `--mode` argument automatically sets `evaluate_gen`:
- `--mode train`: Sets `evaluate_gen=false` for training
- `--mode sample`: Sets `evaluate_gen=true` for sampling

## Output Directory

Outputs are saved in timestamped directories:
- Training output: `./output/results/run_YYYYMMDD_HHMMSS/`
- Sampling output: `./output/results/sample_YYYYMMDD_HHMMSS/`

The configuration file and code backup are automatically saved to the output directory for each run.

## License

This project is licensed under the MIT License.
