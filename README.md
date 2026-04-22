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
pip install pyyaml tensorboard pandas xarray netcdf4 tqdm einops matplotlib cartopy gdown
```

## Data Preparation

The dataset and pre-trained model weights used in this project can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1nb9PitgdqfqfEDaFVefGgf3HK8v4PZ14?usp=drive_link).

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
├── checkpoints/
│   └── GenDA_checkpoint.pth             # Pre-trained model weights (~9 GB)
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
- **GenDA_checkpoint.pth**: Pre-trained model weights. Place this file under the `data/checkpoints/` directory. You can use it to run sampling directly without training from scratch.

## Quick Start

You can either train the model from scratch or directly run sampling with the pre-trained weights provided in `data/checkpoints/GenDA_checkpoint.pth`.

### Sampling with Pre-trained Weights

Run ensemble predictions directly using the provided checkpoint without training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 main.py \
    --mode sample \
    --output_dir './output/results/genda' \
    --resume './data' \
    --checkpoint_name 'GenDA_checkpoint.pth' \
    --val_sample_ratio 0.1
```

Sampling results will be saved to a timestamped directory:
```
./output/results/genda/sample_YYYYMMDD_HHMMSS/
└── samples/               # Generated NetCDF files (YYYYMMDD_dXX.nc)
```

### Training

Train the model from scratch:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 main.py \
    --mode train \
    --batch_size 25 \
    --output_dir './output/results/genda'
```

Checkpoints and logs will be saved to a timestamped directory:
```
./output/results/genda/run_YYYYMMDD_HHMMSS/
├── checkpoints/           # Saved model checkpoints (checkpoint-XX.pth)
├── code_backup/           # Snapshot of all Python source files
├── configs.yaml           # Config used for this run
└── run_YYYYMMDD_HHMMSS.log
```

### Sampling from Your Own Training

Generate ensemble predictions using a model you trained:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 main.py \
    --mode sample \
    --output_dir './output/results/genda' \
    --resume './output/results/genda/<run_timestamp>' \
    --checkpoint_name 'checkpoint-49.pth' \
    --val_sample_ratio 0.1
```

> Replace `<run_timestamp>` with the actual folder name generated during your training (e.g., `run_20260101_120000`).
>
> **Note on `--val_sample_ratio`**: The full 2023 test set is large, and sampling over the entire set can take a considerable amount of time. Setting `--val_sample_ratio 0.1` samples only 10% of the test set (whole days are sampled or skipped — no partial days) for a quick preview of the results. Increase this value (up to 1.0) when you need to evaluate on the full test set.

### Visualizing Sampling Results

After sampling is complete, you can visualize the generated ensemble and vertical profiles using `visualization.py`:

```bash
python visualization.py \
    --sample_dir './output/results/genda/sample_YYYYMMDD_HHMMSS/samples' \
    --ground_truth_dir './data/glorys12/test' \
    --n_samples 10 \
    --save_format jpg \
    --date_str 20230222 \
    --dpi 300
```

Key arguments:

- `--sample_dir` (**required**): Path to the `samples/` directory produced by sampling (e.g., `./output/results/genda/sample_20260101_120000/samples`).
- `--n_samples`: Number of ensemble members to visualize per region (default: `10`).
- `--save_format`: Output image format — `jpg`, `png`, `pdf`, etc. (default: `jpg`).
- `--date_str`: Date string (`YYYYMMDD`) for vertical-profile plots.
- `--dpi`: Resolution of saved figures (default: `300`).

This will generate:
- **Regional ensemble maps** for the North Atlantic, a southern region, the Arctic, and the Antarctic.
- **Vertical profile plots** comparing ensemble statistics against ground truth for the specified date.

**Output directory structure** (relative to your sampling run directory, e.g., `./output/results/genda/sample_YYYYMMDD_HHMMSS/`):

```
.
├── visualizations_ensemble_comparison/     # Regional ensemble maps
│   └── <base_id>/                          # One subdirectory per sampled date
│       ├── <base_id>_ensemble_mean.<fmt>   # Ensemble mean
│       ├── <base_id>_gt.<fmt>              # Ground truth
│       ├── <base_id>_error.<fmt>           # Error (mean - GT)
│       ├── <base_id>_ensemble_std.<fmt>    # Ensemble standard deviation
│       ├── <base_id>_summary.<fmt>         # 4×4 summary grid (4 vars × GT/Sample/Mean/Std)
│       ├── <base_id>_spectral_power.pdf    # Global spectral power plot
│       ├── <base_id>_spectral_power_regional_1.pdf  # Regional spectral power plots
│       ├── <base_id>_summary_regional_1.<fmt>       # Regional summary plots
│       └── <member_id>.<fmt>               # Individual ensemble member plots
└── visualizations_vertical_profiles/       # Vertical profile plots
    └── vertical_profiles_<date_str>.<fmt>  # 3D vertical cross-section visualization
```

## Configuration
The default hyperparameters used in the paper are provided in `configs/configs.yaml`. You can adjust them in two ways:

1. **Edit the config file directly**: Modify `configs/configs.yaml` (or create a copy and specify it via `--config`).
2. **Override from the command line**: Append `--key value` to the launch command to override any config parameter without modifying the file (see below).

Key settings include:

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

> **Note**: Command-line overrides are applied *before* the `--mode` override, so `evaluate_gen` is always determined by `--mode` regardless of command-line overrides.

### Command-Line Config Overrides

Any parameter defined in `configs/configs.yaml` can be overridden directly from the command line using `--key value` syntax. This works for both training and sampling modes.

```bash
# Training with custom batch size and learning rate
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 main.py --mode train \
    --batch_size 16 --blr 1e-4 --epochs 100

# Sampling with custom steps and ensemble size
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 main.py --mode sample \
    --num_sampling_steps 100 --num_samples_per_input 20 --gen_bsz 16

# Use a different config file and override specific parameters
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 main.py --mode train \
    --config ./configs/my_config.yaml --weight_decay 0.01

# Resume training from a checkpoint with adjusted learning rate
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 main.py --mode train \
    --resume ./output/results/genda/run_20250101_120000 --lr 1e-5
```

Type inference is automatic: integers, floats, booleans (`true`/`false`), and null (`null`/`none`) are converted to the appropriate Python types. All other values are treated as strings.

## License

This project is licensed under the MIT License.
