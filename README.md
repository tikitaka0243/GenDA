# Ocean Data Assimilation

Official code repository for the paper: *"Generative Ocean Data Assimilation via Conditional Flow Matching"*.

This project implements a diffusion-based ocean data assimilation system that fuses satellite remote sensing (GLORYS12) with in-situ observations (Argo floats) using conditional flow matching.

## Features

- **Conditional Flow Matching**: Generates ensemble ocean state estimates conditioned on GLORYS12 reanalysis and Argo profiles
- **Two Argo Fusion Methods**:
  - `crossattention`: Cross-Attention mechanism to attend to Argo profiles (Standard)
  - `geo_fusion`: Directly fuses Argo profiles into spatially corresponding patches based on geographic coordinates (Lightweight, O(1) w.r.t patches)
- **Conformal Calibration**: Post-hoc uncertainty quantification with coverage guarantees
- **Distributed Training**: Multi-GPU training via `torchrun`

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
python load_data.py --start-year 2012 --end-year 2025 --username YOUR_USERNAME --password YOUR_PASSWORD
```

### Argo Float Data

The Argo data processing pipeline is included in `modules/datasets/load_argo.py`. See the `ArgoProcessorCSV` class for details.

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

- **Model architecture**: Model type, image size, dropout rates
- **Training hyperparameters**: Learning rate, batch size, epochs, EMA decay
- **Sampling parameters**: Sampling method, steps, number of images
- **Data path**: Path to GLORYS12 dataset and Argo profiles
- **Checkpointing**: Output directory, resume path, save frequency
- **Distributed training**: World size, distributed settings
- **Argo Fusion**: 
  - `argo_fusion_method`: Controls how Argo data is fused into the model
    - `'crossattention'`: Uses Cross-Attention to attend to Argo profiles (Standard)
    - `'geo_fusion'`: Directly fuses Argo profiles into spatially corresponding patches based on geographic coordinates (Lightweight, O(1) w.r.t patches)

The `--mode` argument automatically sets `evaluate_gen`:
- `--mode train`: Sets `evaluate_gen=false` for training
- `--mode sample`: Sets `evaluate_gen=true` for sampling

## Output Directory

Outputs are saved in timestamped directories:
- Training output: `./output/results/run_YYYYMMDD_HHMMSS/`
- Sampling output: `./output/results/sample_YYYYMMDD_HHMMSS/`

The configuration file used is automatically saved to the output directory for each run.

## Project Structure

```
├── main.py                      # Entry point for training and sampling
├── load_data.py                 # Data download and preprocessing
├── post_process.py              # Post-processing utilities
├── configs/
│   └── configs.yaml             # Default configuration
├── modules/
│   ├── conformal.py             # Conformal calibration
│   ├── postprocess.py           # Post-processing functions
│   ├── utils.py                 # Utility functions
│   ├── datasets/
│   │   ├── glorys12_dataset.py  # GLORYS12 dataset class
│   │   ├── load_argo.py         # Argo data loader
│   │   └── load_glorys12.py     # GLORYS12 data downloader
│   ├── flow_matching/
│   │   ├── denoiser.py          # Flow matching denoiser
│   │   ├── engine.py            # Training and evaluation engine
│   │   └── losses.py            # Loss functions
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

If you find this code useful, please consider citing our paper (citation will be added upon publication).

## License

This project is licensed under the MIT License.
