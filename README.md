# GenDA: Global 3D Ocean Data Assimilation with Conformalized Generative Models

Official code repository for the paper: *"Global 3D Ocean Data Assimilation with Conformalized Generative Models"*

> GenDA is a probabilistic framework that combines **conditional flow matching** with **conformal prediction** to reconstruct global three-dimensional ocean temperature, salinity, and velocity fields from sparse Argo float profiles.

## Overview

Reconstructing three-dimensional ocean states from sparse observations is a fundamental challenge in oceanography. Existing approaches face three key limitations:

1. **Deterministic methods** (trained with MSE) estimate the conditional mean and produce overly smooth fields that suppress mesoscale features
2. **Heterogeneous data fusion** between sparse Lagrangian observations (Argo profiles) and gridded Eulerian fields remains challenging
3. **Generative models** lack rigorous uncertainty quantification with finite-sample guarantees

GenDA addresses all three challenges through:

- **Conditional Flow Matching**: Models the full conditional distribution P(Y|X) rather than the conditional mean E[Y|X], preserving fine-scale ocean structures
- **Location-Aware Attention**: Fuses sparse Argo profiles into spatially corresponding grid patches based on geographic coordinates (O(N) complexity w.r.t. number of profiles)
- **Conformal Calibration**: Provides distribution-free, finite-sample coverage guarantees for continuous predictive distributions from generative models

## Key Results

Applied to global ocean reconstruction during 2012–2023:

| Variable | Mean RMSE |
|---|---|
| Temperature | 0.86 °C |
| Salinity | 0.23 PSU |
| Eastward velocity | 0.12 m/s |
| Northward velocity | 0.11 m/s |

- Maintains spectral fidelity across scales from 10,000 km to 100 km
- Recovers dynamics in data-sparse polar regions (Arctic/Antarctic)
- Produces prediction intervals that attain nominal coverage levels after conformal calibration

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- xarray, numpy, pandas, matplotlib, cartopy
- copernicusmarine (for data download)
- argopy (for Argo data processing)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

### GLORYS12 Reanalysis Data

Download GLORYS12 data from [Copernicus Marine Service](https://marine.copernicus.eu/):

```bash
python load_data.py --start-year 2012 --end-year 2023 --username YOUR_USERNAME --password YOUR_PASSWORD
```

GLORYS12 provides daily global ocean state estimates at 1/12° (~8 km) horizontal resolution on 50 depth levels. This study uses the upper 40 depth levels (surface to 2000 m), consistent with the operational depth range of Argo floats.

### Argo Float Data

The Argo data processing pipeline is included in `modules/datasets/load_argo.py`. See the `ArgoProcessorCSV` class for details. Argo profiles within a 5-day window (T−2 to T+2) are used as input for each day T.

## Quick Start

### Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29503 main.py --mode train
```

### Sampling

Generate ensemble predictions using a trained model:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29504 main.py --mode sample
```

### Using Custom Configuration

```bash
torchrun --nproc_per_node=1 main.py --mode train --config path/to/your/config.yaml
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

## Method Details

### Conditional Flow Matching

GenDA adopts the flow matching framework to characterize the full conditional distribution P(Y|X). The model generates the ocean state at each depth level conditioned on Argo observations and depth, using x-prediction (clean data prediction) parameterization with optimal transport paths. Sampling is performed by solving the learned ODE from t=0 to t=1.

### Location-Aware Attention

For each Argo profile, the corresponding patch index on the grid is computed from geographic coordinates. Profile embeddings are then aggregated into their corresponding patches via scatter-add operations, reducing complexity from O(N·P) to O(N) compared to generic cross-attention.

### Conformal Calibration

A post-hoc calibration procedure provides finite-sample coverage guarantees for the continuous predictive distributions:
1. Generate an ensemble of K predictions for each calibration sample
2. Fit Gaussian KDEs to obtain uncalibrated CDFs at each spatial location
3. Compute PIT (Probability Integral Transform) values
4. Fit isotonic regression to correct miscalibration
5. At test time, apply the calibration map to obtain calibrated prediction intervals at any confidence level (1−α)

This guarantees: P(Y ∈ Ĉ_{1−α}) ≥ 1 − α − 1/(M+1), where M is the number of calibration samples.

## Output Directory

Outputs are saved in timestamped directories:
- Training output: `./output/results/run_YYYYMMDD_HHMMSS/`
- Sampling output: `./output/results/sample_YYYYMMDD_HHMMSS/`

The configuration file and code backup are automatically saved to the output directory for each run.

## Project Structure

```
├── main.py                      # Entry point for training and sampling
├── load_data.py                 # Data download and preprocessing
├── post_process.py              # Post-processing utilities
├── configs/
│   └── configs.yaml             # Default configuration
├── modules/
│   ├── conformal.py             # Conformal calibration with isotonic regression
│   ├── postprocess.py           # Post-processing functions
│   ├── utils.py                 # Utility functions
│   ├── datasets/
│   │   ├── glorys12_dataset.py  # GLORYS12 + Argo dataset class
│   │   ├── load_argo.py         # Argo data loader and processor
│   │   └── load_glorys12.py     # GLORYS12 data downloader
│   ├── flow_matching/
│   │   ├── denoiser.py          # Flow matching clean data predictor (DANet)
│   │   ├── engine.py            # Training and evaluation engine
│   │   └── losses.py            # Flow matching loss functions
│   ├── networks/
│   │   ├── fibonacci.py         # Fibonacci sphere sampling
│   │   └── model.py             # DANet model architecture
│   ├── plot/
│   │   ├── plot.py              # Visualization functions
│   │   ├── plot_call.py         # Plot invocation helpers
│   │   └── plot_conformal.py    # Conformal calibration plots
│   └── util/
│       ├── crop.py              # Data cropping utilities
│       ├── lr_sched.py          # Learning rate schedulers
│       ├── misc.py              # Miscellaneous utilities
│       └── model_util.py        # Model utility functions
```

## Citation

If you find this code useful, please consider citing our paper:

```bibtex
@article{genda2025,
  title={Global 3D Ocean Data Assimilation with Conformalized Generative Models},
  author={Anonymous},
  journal={Journal of the American Statistical Association},
  year={2025},
  note={Under review}
}
```

## License

This project is licensed under the MIT License.
