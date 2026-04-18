"""
Post-processing module for model sampling results visualization and export
"""

import os
import random
import csv
import json
import numpy as np
import xarray as xr
import concurrent.futures
import re
from datetime import datetime
from typing import List, Optional, Tuple, Union
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from modules.plot.plot import plot_ocean_variables, plot_variable_distribution, plot_diff_variables, plot_rmse_boxplots, plot_ensemble_summary, plot_spectral_power, plot_polar_ensemble_summary, plot_vertical_profiles_3d


class SamplePostProcessor:
    """
    Sample Post-Processor Class
    
    Process model-generated ocean data samples, including random sampling, visualization, and export.
    """
    
    def __init__(self, 
                 sample_dir: str,
                 output_dir: Optional[str] = None,
                 variables: List[str] = None):
        """
        Initialize post-processor
        
        Args:
            sample_dir (str): Sample result directory path (containing .npy or .npz files)
            output_dir (str, optional): Output directory path. If None, creates postprocess subdirectory under sample_dir
            variables (List[str], optional): Variable name list. Defaults to ['thetao', 'so', 'uo', 'vo']
        """
        self.sample_dir = sample_dir
        
        # Set output directory
        if output_dir is None:
            self.output_dir = os.path.dirname(sample_dir)
        else:
            self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set variable names
        if variables is None:
            self.variables = ['thetao', 'so', 'uo', 'vo']
        else:
            self.variables = variables
        
        # Get all sample files
        self.sample_files = self._get_sample_files()
        
        print(f"Post-processor initialized")
        print(f"  Sample directory: {self.sample_dir}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Number of samples found: {len(self.sample_files)}")
    
    def _get_sample_files(self) -> List[str]:
        """
        Get all sample file paths
        
        Returns:
            List[str]: List of sample file paths
        """
        if not os.path.exists(self.sample_dir):
            raise FileNotFoundError(f"Sample directory does not exist: {self.sample_dir}")
        
        # Find all .npy, .npz and .nc files
        sample_files = []
        for file in os.listdir(self.sample_dir):
            if file.endswith('.npy') or file.endswith('.npz') or file.endswith('.nc'):
                sample_files.append(os.path.join(self.sample_dir, file))
        
        # Sort by file name
        sample_files.sort()
        
        if len(sample_files) == 0:
            print(f"Warning: No .npy or .npz files found in {self.sample_dir}")
        
        return sample_files
    
    def _load_sample(self, file_path: str) -> np.ndarray:
        """
        Load a single sample file
        
        Args:
            file_path (str): File path
            
        Returns:
            np.ndarray: Sample data with shape (C, H, W)
        """
        if file_path.endswith('.nc'):
            ds = xr.open_dataset(file_path)
            if 'data' in ds:
                data = ds['data'].values
            else:
                # Try to concatenate variables if 'data' not present
                # This assumes variables are thetao, so, uo, vo
                vars_list = ['thetao', 'so', 'uo', 'vo']
                present_vars = [v for v in vars_list if v in ds]
                if present_vars:
                    data_list = [ds[v].values for v in present_vars]
                    # Handle shapes (check if time dim exists)
                    processed_list = []
                    for d in data_list:
                        if d.ndim == 3: # (time, lat, lon) -> (lat, lon)
                             processed_list.append(d[0])
                        else:
                             processed_list.append(d)
                    data = np.stack(processed_list)
                else:
                     raise ValueError(f"Could not find valid data in .nc file: {file_path}")
            ds.close()
            return data

        data = np.load(file_path)
        
        # Handle .npz files
        if file_path.endswith('.npz'):
            # Directly use 'data' key as defined in engine.py
            if hasattr(data, 'files'):
                if 'data' in data.files:
                    data = data['data']
                else:
                    raise ValueError(f"Key 'data' not found in .npz file: {file_path}. Available keys: {data.files}")

        return data
    
    def _convert_to_xarray(self, 
                          data: np.ndarray, 
                          lat_range: Tuple[float, float] = (-80, 90),
                          lon_range: Tuple[float, float] = (-180, 180)) -> xr.Dataset:
        """
        Convert numpy array to xarray Dataset
        
        Args:
            data (np.ndarray): Data array with shape (C, H, W)
            lat_range (Tuple[float, float]): Latitude range (min, max)
            lon_range (Tuple[float, float]): Longitude range (min, max)
            
        Returns:
            xr.Dataset: xarray Dataset object
        """
        C, H, W = data.shape
        
        # Create coordinates
        latitudes = np.linspace(lat_range[0], lat_range[1], H)
        longitudes = np.linspace(lon_range[0], lon_range[1], W)
        time = np.array([datetime.now()], dtype='datetime64[ns]')
        
        # Create data variables dictionary
        data_vars = {}
        for i, var_name in enumerate(self.variables[:C]):
            data_vars[var_name] = (['time', 'latitude', 'longitude'], 
                                   data[i:i+1, :, :])
        
        # Create Dataset
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                'time': time,
                'latitude': latitudes,
                'longitude': longitudes
            }
        )
        
        return ds
    
    def random_sample_and_visualize(self,
                                   n_samples: int = 5,
                                   seed: Optional[int] = None,
                                   save_format: str = 'png',
                                   dpi: int = 300,
                                   figsize: Tuple[int, int] = (16, 12),
                                   export_netcdf: bool = True) -> List[str]:
        """
        Randomly sample and visualize samples with export
        
        Args:
            n_samples (int): Number of samples to draw, default 5
            seed (int, optional): Random seed for reproducibility
            save_format (str): Image save format, default 'png'
            dpi (int): Image resolution, default 300
            figsize (Tuple[int, int]): Figure size, default (16, 12)
            export_netcdf (bool): Whether to export as NetCDF format, default True
            
        Returns:
            List[str]: List of generated file paths
        """
        if len(self.sample_files) == 0:
            print("Error: No sample files available")
            return []
        
        # Set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Randomly sample files
        n_samples = min(n_samples, len(self.sample_files))
        selected_files = random.sample(self.sample_files, n_samples)
        
        print(f"\n{'='*60}")
        print(f"Processing {n_samples} randomly selected samples")
        print(f"{'='*60}\n")
        
        output_files = []
        
        # Create visualization subdirectory
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create data export subdirectory
        if export_netcdf:
            data_dir = os.path.join(self.output_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
        
        # Process each selected sample
        for idx, file_path in enumerate(selected_files, 1):
            file_name = os.path.basename(file_path)
            sample_id = os.path.splitext(file_name)[0]
            
            print(f"[{idx}/{n_samples}] Processing sample: {file_name}")
            
            try:
                # Load data
                data = self._load_sample(file_path)
                print(f"  Data shape: {data.shape}")
                print(f"  Value range: [{data.min():.4f}, {data.max():.4f}]")
                
                # Convert to xarray Dataset
                ds = self._convert_to_xarray(data)
                
                # Generate visualization
                output_image = os.path.join(viz_dir, f'sample_{sample_id}.{save_format}')
                plot_ocean_variables(
                    ds,
                    output_path=output_image,
                    figsize=figsize,
                    dpi=dpi
                )
                output_files.append(output_image)
                print(f"  ✓ Visualization saved: {output_image}")
                
                # Export as NetCDF (if enabled)
                if export_netcdf:
                    output_nc = os.path.join(data_dir, f'sample_{sample_id}.nc')
                    ds.to_netcdf(output_nc)
                    output_files.append(output_nc)
                    print(f"  ✓ NetCDF exported: {output_nc}")
                
                # Print statistics
                for var_name in self.variables[:data.shape[0]]:
                    var_data = ds[var_name].values
                    print(f"  {var_name}: mean={np.mean(var_data):.4f}, "
                          f"std={np.std(var_data):.4f}, "
                          f"min={np.min(var_data):.4f}, "
                          f"max={np.max(var_data):.4f}")
                
                print()
                
            except Exception as e:
                print(f"  ✗ Processing failed: {e}")
                continue
        
        print(f"{'='*60}")
        print(f"Processing complete! Generated {len(output_files)} files")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")
        
        # Generate processing summary
        self._generate_summary(selected_files, output_files)
        
        return output_files
    
    def random_sample_and_visualize_with_ground_truth(self,
                                                     ground_truth_dir: str = './data/glorys12/val',
                                                     n_samples: int = 5,
                                                     seed: Optional[int] = None,
                                                     save_format: str = 'jpg',
                                                     dpi: int = 300,
                                                     figsize: Tuple[int, int] = (21, 10)) -> List[str]:
        """
        Randomly sample and visualize samples, ground truth, and their difference.
        
        Args:
            ground_truth_dir (str): Directory containing ground truth files
            n_samples (int): Number of samples to draw, default 5
            seed (int, optional): Random seed for reproducibility
            save_format (str): Image save format, default 'jpg'
            dpi (int): Image resolution, default 300
            figsize (Tuple[int, int]): Figure size, default (16, 12)
            
        Returns:
            List[str]: List of generated file paths
        """
        if len(self.sample_files) == 0:
            print("Error: No sample files available")
            return []
            
        if not os.path.exists(ground_truth_dir):
            print(f"Error: Ground truth directory does not exist: {ground_truth_dir}")
            return []
        
        # Set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Randomly sample files
        n_samples = min(n_samples, len(self.sample_files))
        selected_files = random.sample(self.sample_files, n_samples)
        
        print(f"\n{'='*60}")
        print(f"Processing {n_samples} randomly selected samples with Ground Truth")
        print(f"{'='*60}\n")
        
        output_files = []
        
        # Create visualization subdirectory
        viz_dir = os.path.join(self.output_dir, 'visualizations_comparison')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Process each selected sample
        for idx, file_path in enumerate(tqdm(selected_files, desc="Processing samples"), 1):
            file_name = os.path.basename(file_path)
            sample_id = os.path.splitext(file_name)[0]
            
            # Construct ground truth path
            # Assume GT file has same basename but .nc extension
            gt_filename = sample_id + '.nc'
            gt_path = os.path.join(ground_truth_dir, gt_filename)
            
            # Try to handle common naming differences
            if not os.path.exists(gt_path):
                # 1. Try removing "sample_" prefix
                if sample_id.startswith('sample_'):
                    alt_id = sample_id.replace('sample_', '')
                    alt_path = os.path.join(ground_truth_dir, alt_id + '.nc')
                    if os.path.exists(alt_path):
                        gt_path = alt_path
                
                # 2. Try adding "sample_" prefix (unlikely for GT but possible)
                elif not sample_id.startswith('sample_'):
                     alt_path = os.path.join(ground_truth_dir, 'sample_' + gt_filename)
                     if os.path.exists(alt_path):
                         gt_path = alt_path

            if not os.path.exists(gt_path):
                print(f"Warning: Ground truth file not found: {gt_path}, skipping...")
                continue
            
            try:
                # Load data
                sample_data = self._load_sample(file_path)
                
                # Load GT data from NetCDF
                gt_data_list = []
                with xr.open_dataset(gt_path) as ds_gt_raw:
                    for var in self.variables:
                        if var not in ds_gt_raw:
                            raise ValueError(f"Variable {var} not found in GT file")
                        
                        # Extract data
                        var_data = ds_gt_raw[var].values
                        
                        # Handle dimensions: if (Time, Lat, Lon) -> (Lat, Lon)
                        if var_data.ndim == 3 and var_data.shape[0] == 1:
                            var_data = var_data[0]
                        
                        gt_data_list.append(var_data)
                
                gt_data = np.array(gt_data_list)
                
                # Check shapes and pad GT if necessary (Reference: modules/datasets/glorys12_dataset.py)
                if sample_data.shape != gt_data.shape:
                    # Check if mismatch is only in H dimension and GT is smaller
                    # Sample: (C, H_s, W), GT: (C, H_g, W)
                    if (sample_data.shape[0] == gt_data.shape[0] and 
                        sample_data.shape[2] == gt_data.shape[2] and 
                        sample_data.shape[1] > gt_data.shape[1]):
                        
                        pad_h = sample_data.shape[1] - gt_data.shape[1]
                        
                        # Pad with NaNs: ((0,0), (pad_h, 0), (0,0)) for (C, H, W)
                        # Corresponds to torch.nn.functional.pad(padding=(0,0, pad_h,0))
                        gt_data = np.pad(gt_data, ((0, 0), (pad_h, 0), (0, 0)), 
                                       mode='constant', constant_values=np.nan)
                
                # Check shapes again
                if sample_data.shape != gt_data.shape:
                    print(f"  Warning: Shape mismatch! Sample: {sample_data.shape}, GT: {gt_data.shape}")
                    continue
                
                # Calculate difference
                diff_data = sample_data - gt_data
                
                # Mask Land in Sample Data
                # Use GT NaN values to identify land
                # GT data already has NaNs where there is land (if loaded correctly)
                # Ensure we apply this mask to sample_data before visualization
                
                # Create land mask from GT data
                land_mask = np.isnan(gt_data)
                
                # Apply mask to sample data
                sample_data_masked = sample_data.copy()
                sample_data_masked[land_mask] = np.nan
                
                # Also apply to diff_data (should already be NaN if GT is NaN, but good to be explicit)
                diff_data[land_mask] = np.nan

                # Calculate vmin/vmax based on GT data
                vmin_vmax = {}
                for i, var in enumerate(self.variables):
                    var_data = gt_data[i]
                    # Handle all-NaN slice
                    if np.all(np.isnan(var_data)):
                         vmin, vmax = 0, 1
                    else:
                        if var == 'so': # Salinity
                            # Use 1st and 99th percentiles for robust scaling
                            vmin = np.nanpercentile(var_data, 0.5)
                            vmax = np.nanpercentile(var_data, 99.5)
                        elif var in ['uo', 'vo']: # Velocity
                            # Symmetric scale based on 99th percentile of absolute values
                            vmax = np.nanpercentile(np.abs(var_data), 99.5)
                            vmin = -vmax
                            if vmax == 0: vmin, vmax = -1, 1
                        else:
                            # Default for other variables (e.g., temperature)
                            vmin = np.nanmin(var_data)
                            vmax = np.nanmax(var_data)
                            if vmin == vmax: vmin, vmax = vmin - 0.5, vmax + 0.5
                    
                    vmin_vmax[var] = (vmin, vmax)
                
                # Convert to xarray Dataset (use same conversion for consistency)
                ds_sample = self._convert_to_xarray(sample_data_masked, lat_range=(-86, 90))
                ds_gt = self._convert_to_xarray(gt_data, lat_range=(-86, 90))
                ds_diff = self._convert_to_xarray(diff_data, lat_range=(-86, 90))
                
                # Extract time and depth info from sample_id
                # sample_id format: [sample_]YYYYMMDD_dXX (e.g. 20230101_d00)
                try:
                    clean_id = sample_id.replace('sample_', '')
                    parts = clean_id.split('_')
                    if len(parts) >= 2:
                        date_str = parts[0]
                        depth_idx = parts[1][1:]
                        # Format date nicely: 20230101 -> 2023-01-01
                        if len(date_str) == 8:
                            date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                        else:
                            date_formatted = date_str
                        
                        title_info = f"Time: {date_formatted}, Depth Index: {depth_idx}"
                        sample_title = f"Ocean Variables (Sample) - {title_info}"
                        gt_title = f"Ocean Variables (Ground Truth) - {title_info}"
                        diff_title = f"Ocean Variables Difference - {title_info}"
                    else:
                        sample_title = None
                        gt_title = None
                        diff_title = None
                except:
                    sample_title = None
                    gt_title = None
                    diff_title = None
                
                # Generate visualizations
                # 1. Sample (Pass vmin_vmax to align with GT)
                out_sample = os.path.join(viz_dir, f'{sample_id}_pred.{save_format}')
                plot_ocean_variables(ds_sample, output_path=out_sample, figsize=figsize, dpi=dpi, vmin_vmax=vmin_vmax, title=sample_title)
                output_files.append(out_sample)
                
                # 2. Ground Truth (Pass vmin_vmax to ensure consistency)
                out_gt = os.path.join(viz_dir, f'{sample_id}_gt.{save_format}')
                plot_ocean_variables(ds_gt, output_path=out_gt, figsize=figsize, dpi=dpi, vmin_vmax=vmin_vmax, title=gt_title)
                output_files.append(out_gt)
                
                # 3. Difference
                out_diff = os.path.join(viz_dir, f'{sample_id}_diff.{save_format}')
                plot_diff_variables(ds_diff, output_path=out_diff, figsize=figsize, dpi=dpi, title=diff_title)
                output_files.append(out_diff)
                
                print()
                
            except Exception as e:
                print(f"  ✗ Processing failed: {e}")
                continue
        
        print(f"{'='*60}")
        print(f"Comparison complete! Generated {len(output_files)} files")
        print(f"{'='*60}\n")
        
        return output_files
    
    def visualize_ensemble_with_ground_truth(self,
                                             ground_truth_dir: str = './data/glorys12/test',
                                             n_samples: int = 5,
                                             seed: Optional[int] = None,
                                             save_format: str = 'jpg',
                                             dpi: int = 300,
                                             figsize: Tuple[int, int] = (21, 10),
                                             plot_mode: str = 'all',
                                             region_extent: Optional[Union[Tuple[float, float, float, float], List[Tuple[float, float, float, float]]]] = None) -> List[str]:
        """
        Randomly sample and visualize ensemble mean (grouped by _rXX suffix), ground truth, and their difference.
        
        Args:
            ground_truth_dir (str): Directory containing ground truth files
            n_samples (int): Number of samples to draw, default 5
            seed (int, optional): Random seed for reproducibility
            save_format (str): Image save format, default 'jpg'
            dpi (int): Image resolution, default 300
            figsize (Tuple[int, int]): Figure size, default (21, 10)
            plot_mode (str): Plot mode, one of:
                - 'all': Generate all plots (individual + summary + spectral + regional if region_extent provided)
                - 'summary': Only generate global summary plot (4x4 grid)
                - 'region_summary': Generate all regional summary plots (polar + non-polar, requires region_extent)
                - 'regional': Only generate non-polar regional summary plots (requires region_extent)
                - 'polar': Only generate polar regional summary plots (Arctic + Antarctic, requires region_extent)
                - 'spectral': Only generate spectral power plot(s)
                Default is 'all'.
            region_extent (tuple or list of tuples, optional): Map extent for regional summary plot(s).
                                            Can be a single tuple (lon_min, lon_max, lat_min, lat_max) or a list of tuples
                                            for multiple regions. If provided, regional summary plot(s) will be generated.
            
        Returns:
            List[str]: List of generated file paths
        """
        # Normalize region_extent to a list for uniform processing
        if region_extent is not None:
            if isinstance(region_extent, tuple):
                region_extents = [region_extent]
            else:
                region_extents = region_extent
        else:
            region_extents = []
        if len(self.sample_files) == 0:
            print("Error: No sample files available")
            return []
            
        if not os.path.exists(ground_truth_dir):
            print(f"Error: Ground truth directory does not exist: {ground_truth_dir}")
            return []
        
        # Group files by base ID (removing _rXX suffix)
        ensemble_groups = self._get_ensemble_groups()
        base_ids = list(ensemble_groups.keys())
        
        if not base_ids:
            print("Error: No ensemble groups found")
            return []
        
        # Set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Randomly sample base IDs
        n_samples = min(n_samples, len(base_ids))
        selected_base_ids = random.sample(base_ids, n_samples)
        
        print(f"\n{'='*60}")
        print(f"Processing {n_samples} randomly selected ensemble groups")
        print(f"Total groups found: {len(base_ids)}")
        print(f"{'='*60}\n")
        
        output_files = []
        
        # Create visualization subdirectory
        viz_dir = os.path.join(self.output_dir, 'visualizations_ensemble_comparison')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Process each selected group
        for idx, base_id in enumerate(tqdm(selected_base_ids, desc="Processing ensemble groups"), 1):
            file_paths = ensemble_groups[base_id]
            # Sort to ensure deterministic order if needed
            file_paths.sort()
            
            print(f"[{idx}/{n_samples}] Processing group: {base_id} (Members: {len(file_paths)})")
            
            # Create subfolder for this sample
            sample_viz_dir = os.path.join(viz_dir, base_id)
            os.makedirs(sample_viz_dir, exist_ok=True)
            
            # Construct ground truth path
            gt_filename = base_id + '.nc'
            gt_path = os.path.join(ground_truth_dir, gt_filename)
            
            # Logic to handle sample_ prefix if missing/present in GT
            if not os.path.exists(gt_path):
                if base_id.startswith('sample_'):
                    alt_id = base_id.replace('sample_', '')
                    alt_path = os.path.join(ground_truth_dir, alt_id + '.nc')
                    if os.path.exists(alt_path):
                        gt_path = alt_path
                elif not base_id.startswith('sample_'):
                     alt_path = os.path.join(ground_truth_dir, 'sample_' + gt_filename)
                     if os.path.exists(alt_path):
                         gt_path = alt_path

            if not os.path.exists(gt_path):
                print(f"  Warning: Ground truth file not found for {base_id}, skipping...")
                continue
            
            try:
                # Load data from all ensemble members
                member_data_list = []
                for fp in file_paths:
                    data = self._load_sample(fp)
                    member_data_list.append(data)
                
                # Stack and compute mean
                ensemble_stack = np.stack(member_data_list, axis=0) # (N_members, C, H, W)
                ensemble_mean = np.mean(ensemble_stack, axis=0) # (C, H, W)
                
                # Load GT data
                gt_data_list = []
                with xr.open_dataset(gt_path) as ds_gt_raw:
                    for var in self.variables:
                        if var not in ds_gt_raw:
                            raise ValueError(f"Variable {var} not found in GT file")
                        var_data = ds_gt_raw[var].values
                        if var_data.ndim == 3 and var_data.shape[0] == 1:
                            var_data = var_data[0]
                        gt_data_list.append(var_data)
                
                gt_data = np.array(gt_data_list)
                
                # Check shapes and pad GT if necessary
                if ensemble_mean.shape != gt_data.shape:
                    if (ensemble_mean.shape[0] == gt_data.shape[0] and 
                        ensemble_mean.shape[2] == gt_data.shape[2] and 
                        ensemble_mean.shape[1] > gt_data.shape[1]):
                        
                        pad_h = ensemble_mean.shape[1] - gt_data.shape[1]
                        gt_data = np.pad(gt_data, ((0, 0), (pad_h, 0), (0, 0)), 
                                       mode='constant', constant_values=np.nan)
                
                if ensemble_mean.shape != gt_data.shape:
                    print(f"  Warning: Shape mismatch! Ensemble: {ensemble_mean.shape}, GT: {gt_data.shape}")
                    continue
                
                # Calculate difference
                diff_data = ensemble_mean - gt_data
                
                # Create land mask from GT data
                land_mask = np.isnan(gt_data)
                
                # Apply mask to ensemble mean
                ensemble_mean_masked = ensemble_mean.copy()
                ensemble_mean_masked[land_mask] = np.nan
                
                # Apply mask to diff_data
                diff_data[land_mask] = np.nan

                # Calculate vmin/vmax based on GT data
                vmin_vmax = {}
                for i, var in enumerate(self.variables):
                    var_data = gt_data[i]
                    if np.all(np.isnan(var_data)):
                         vmin, vmax = 0, 1
                    else:
                        if var == 'so':
                            vmin = np.nanpercentile(var_data, 0.5)
                            vmax = np.nanpercentile(var_data, 99.5)
                        elif var in ['uo', 'vo']:
                            vmax = np.nanpercentile(np.abs(var_data), 99.5)
                            vmin = -vmax
                            if vmax == 0: vmin, vmax = -1, 1
                        else:
                            vmin = np.nanmin(var_data)
                            vmax = np.nanmax(var_data)
                            if vmin == vmax: vmin, vmax = vmin - 0.5, vmax + 0.5
                    
                    vmin_vmax[var] = (vmin, vmax)
                
                # Convert to xarray Dataset
                ds_mean = self._convert_to_xarray(ensemble_mean_masked, lat_range=(-86, 90))
                ds_gt = self._convert_to_xarray(gt_data, lat_range=(-86, 90))
                ds_diff = self._convert_to_xarray(diff_data, lat_range=(-86, 90))
                
                # Extract time and depth info
                try:
                    clean_id = base_id.replace('sample_', '')
                    parts = clean_id.split('_')
                    if len(parts) >= 2:
                        date_str = parts[0]
                        depth_idx = parts[1][1:]
                        if len(date_str) == 8:
                            date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                        else:
                            date_formatted = date_str
                        
                        title_info = f"Time: {date_formatted}, Depth Index: {depth_idx}"
                        mean_title = f"Ensemble Mean (N={len(file_paths)}) - {title_info}"
                        gt_title = f"Ground Truth - {title_info}"
                        diff_title = f"Error (Mean - GT) - {title_info}"
                    else:
                        mean_title = None
                        gt_title = None
                        diff_title = None
                except:
                    mean_title = None
                    gt_title = None
                    diff_title = None
                
                # Calculate ensemble std (needed for both modes)
                ensemble_std = np.std(ensemble_stack, axis=0)
                ensemble_std_masked = ensemble_std.copy()
                ensemble_std_masked[land_mask] = np.nan
                ds_std = self._convert_to_xarray(ensemble_std_masked, lat_range=(-86, 90))
                
                # Generate individual visualizations only if plot_mode is 'all'
                if plot_mode == 'all':
                    # Generate all individual visualizations
                    # 1. Ensemble Mean
                    out_mean = os.path.join(sample_viz_dir, f'{base_id}_ensemble_mean.{save_format}')
                    if os.path.exists(out_mean):
                        print(f"  Skipping (exists): {out_mean}")
                    else:
                        plot_ocean_variables(ds_mean, output_path=out_mean, figsize=figsize, dpi=dpi, vmin_vmax=vmin_vmax, title=mean_title)
                        output_files.append(out_mean)
                    
                    # 2. Ground Truth
                    out_gt = os.path.join(sample_viz_dir, f'{base_id}_gt.{save_format}')
                    if os.path.exists(out_gt):
                        print(f"  Skipping (exists): {out_gt}")
                    else:
                        plot_ocean_variables(ds_gt, output_path=out_gt, figsize=figsize, dpi=dpi, vmin_vmax=vmin_vmax, title=gt_title)
                        output_files.append(out_gt)
                    
                    # 3. Difference (Error)
                    out_diff = os.path.join(sample_viz_dir, f'{base_id}_error.{save_format}')
                    if os.path.exists(out_diff):
                        print(f"  Skipping (exists): {out_diff}")
                    else:
                        plot_diff_variables(ds_diff, output_path=out_diff, figsize=figsize, dpi=dpi, title=diff_title)
                        output_files.append(out_diff)
                    
                    # 4. Ensemble Std
                    std_title = f"Ensemble Std (N={len(file_paths)}) - {title_info}" if mean_title else None
                    out_std = os.path.join(sample_viz_dir, f'{base_id}_ensemble_std.{save_format}')
                    if os.path.exists(out_std):
                        print(f"  Skipping (exists): {out_std}")
                    else:
                        plot_ocean_variables(ds_std, output_path=out_std, figsize=figsize, dpi=dpi, title=std_title)
                        output_files.append(out_std)
                    
                    # 5. Two Random Samples
                    n_members = len(member_data_list)
                    sample_indices = random.sample(range(n_members), min(2, n_members))
                    
                    for s_idx in sample_indices:
                        sample_data = member_data_list[s_idx]
                        sample_data_masked = sample_data.copy()
                        sample_data_masked[land_mask] = np.nan
                        
                        ds_sample = self._convert_to_xarray(sample_data_masked, lat_range=(-86, 90))
                        
                        member_filename = os.path.basename(file_paths[s_idx])
                        member_id = os.path.splitext(member_filename)[0]
                        
                        sample_title = f"Sample Member: {member_id} - {title_info}" if mean_title else None
                        out_sample = os.path.join(sample_viz_dir, f'{member_id}.{save_format}')
                        
                        if os.path.exists(out_sample):
                            print(f"  Skipping (exists): {out_sample}")
                        else:
                            plot_ocean_variables(ds_sample, output_path=out_sample, figsize=figsize, dpi=dpi, vmin_vmax=vmin_vmax, title=sample_title)
                            output_files.append(out_sample)
                
                # Get one sample for summary plot (in both modes)
                n_members = len(member_data_list)
                sample_idx = random.randint(0, n_members - 1)
                sample_data = member_data_list[sample_idx]
                sample_data_masked = sample_data.copy()
                sample_data_masked[land_mask] = np.nan
                first_sample_ds = self._convert_to_xarray(sample_data_masked, lat_range=(-86, 90))
                
                # Generate global summary plot only if plot_mode is 'all' or 'summary'
                if plot_mode in ('all', 'summary'):
                    # Summary Plot (4x4 grid: 4 variables x 4 columns)
                    summary_title = None
                    out_summary = os.path.join(sample_viz_dir, f'{base_id}_summary.{save_format}')
                    if os.path.exists(out_summary):
                        print(f"  Skipping (exists): {out_summary}")
                    else:
                        plot_ensemble_summary(
                            ds_gt=ds_gt,
                            ds_sample=first_sample_ds,
                            ds_mean=ds_mean,
                            ds_std=ds_std,
                            output_path=out_summary,
                            vmin_vmax=vmin_vmax,
                            title=summary_title
                        )
                        output_files.append(out_summary)
                
                # Generate spectral power plots only if plot_mode is 'all' or 'spectral'
                if plot_mode in ('all', 'spectral'):
                    # 1. Global Spectral Power Plot (always use PDF format)
                    out_spectral = os.path.join(sample_viz_dir, f'{base_id}_spectral_power.pdf')
                    if os.path.exists(out_spectral):
                        print(f"  Skipping (exists): {out_spectral}")
                    else:
                        spectral_title = f"Spectral Power - {title_info}" if title_info else None
                        plot_spectral_power(
                            gt_data=gt_data,
                            sample_data=sample_data_masked,
                            ensemble_mean_data=ensemble_mean_masked,
                            variables=self.variables,
                            output_path=out_spectral,
                            title=spectral_title,
                            figsize=(11, 2.5),
                        )
                        output_files.append(out_spectral)
                    
                    # 2. Regional Spectral Power Plots for all specified regions (always use PDF format)
                    for region_idx, extent in enumerate(region_extents):
                        lon_min, lon_max, lat_min, lat_max = extent
                        if len(region_extents) == 1:
                            region_suffix = "regional"
                        else:
                            region_suffix = f"regional_{region_idx+1}"
                        
                        out_spectral_region = os.path.join(sample_viz_dir, f'{base_id}_spectral_power_{region_suffix}.pdf')
                        if os.path.exists(out_spectral_region):
                            print(f"  Skipping (exists): {out_spectral_region}")
                        else:
                            spectral_region_title = f"Spectral Power ({region_suffix}) - {title_info}" if title_info else None
                            
                            # Get coordinates from ds_gt
                            lat_coords = ds_gt.latitude.values
                            lon_coords = ds_gt.longitude.values
                            
                            plot_spectral_power(
                                gt_data=gt_data,
                                sample_data=sample_data_masked,
                                ensemble_mean_data=ensemble_mean_masked,
                                variables=self.variables,
                                output_path=out_spectral_region,
                                title=spectral_region_title,
                                extent=extent,
                                lat_coords=lat_coords,
                                lon_coords=lon_coords
                            )
                            output_files.append(out_spectral_region)
                            print(f"  Generated regional spectral power ({region_idx+1}/{len(region_extents)}): {out_spectral_region}")
                
                # Generate regional summary plots for all specified regions
                if plot_mode in ('all', 'region_summary', 'regional', 'polar'):
                    for region_idx, extent in enumerate(region_extents):
                        lon_min, lon_max, lat_min, lat_max = extent
                        
                        # Determine region type (polar vs regular)
                        if lat_min >= 60:
                            region_type = "arctic"
                            is_polar_region = True
                        elif lat_max <= -60:
                            region_type = "antarctic"
                            is_polar_region = True
                        else:
                            region_type = "regional"
                            is_polar_region = False
                        
                        # Skip based on plot_mode
                        if plot_mode == 'regional' and is_polar_region:
                            continue
                        if plot_mode == 'polar' and not is_polar_region:
                            continue
                        
                        if len(region_extents) == 1:
                            region_suffix = region_type
                        else:
                            region_suffix = f"{region_type}_{region_idx+1}"
                        
                        # Calculate regional vmin/vmax based on cropped ground truth data
                        regional_vmin_vmax = {}
                        regional_std_vmin_vmax = {}
                        for i, var in enumerate(self.variables):
                            # Get the full variable data from ground truth
                            var_data_full = ds_gt[var].isel(time=0).values
                            lat_coords = ds_gt.latitude.values
                            lon_coords = ds_gt.longitude.values
                            
                            # Find indices for the region
                            lat_mask = (lat_coords >= lat_min) & (lat_coords <= lat_max)
                            lon_mask = (lon_coords >= lon_min) & (lon_coords <= lon_max)
                            
                            # Crop data to region
                            var_data_region = var_data_full[np.ix_(lat_mask, lon_mask)]
                            
                            if np.all(np.isnan(var_data_region)):
                                vmin, vmax = 0, 1
                            else:
                                if var == 'so':
                                    vmin = np.nanpercentile(var_data_region, 0.5)
                                    vmax = np.nanpercentile(var_data_region, 99.5)
                                elif var in ['uo', 'vo']:
                                    vmax = np.nanpercentile(np.abs(var_data_region), 99.5)
                                    vmin = -vmax
                                    if vmax == 0:
                                        vmin, vmax = -1, 1
                                else:
                                    vmin = np.nanmin(var_data_region)
                                    vmax = np.nanmax(var_data_region)
                                    if vmin == vmax:
                                        vmin, vmax = vmin - 0.5, vmax + 0.5
                            
                            regional_vmin_vmax[var] = (vmin, vmax)
                            
                            # Calculate regional vmin/vmax for std plot
                            std_data_full = ds_std[var].isel(time=0).values
                            std_data_region = std_data_full[np.ix_(lat_mask, lon_mask)]
                            
                            if np.all(np.isnan(std_data_region)):
                                std_vmin, std_vmax = 0, 1
                            else:
                                if var in ['uo', 'vo']:
                                    # For velocity variables: make colorbar symmetric around zero
                                    std_vmax = np.nanpercentile(np.abs(std_data_region), 99.5)
                                    if std_vmax == 0:
                                        std_vmax = 1
                                    std_vmin = -std_vmax
                                else:
                                    # For other variables: use 0 as minimum
                                    std_vmin = 0
                                    std_vmax = np.nanmax(std_data_region)
                                    if std_vmax == 0:
                                        std_vmax = 1
                            
                            regional_std_vmin_vmax[var] = (std_vmin, std_vmax)
                        
                        out_region_summary = os.path.join(sample_viz_dir, f'{base_id}_summary_{region_suffix}.{save_format}')
                        
                        if os.path.exists(out_region_summary):
                            print(f"  Skipping (exists): {out_region_summary}")
                        else:
                            # Check if this is a polar region
                            is_polar = False
                            pole = None
                            if lat_min >= 60:  # North polar region
                                is_polar = True
                                pole = 'north'
                            elif lat_max <= -60:  # South polar region
                                is_polar = True
                                pole = 'south'
                            
                            if is_polar:
                                # Use polar projection for polar regions
                                plot_polar_ensemble_summary(
                                    ds_gt=ds_gt,
                                    ds_sample=first_sample_ds,
                                    ds_mean=ds_mean,
                                    ds_std=ds_std,
                                    output_path=out_region_summary,
                                    vmin_vmax=regional_vmin_vmax,
                                    std_vmin_vmax=regional_std_vmin_vmax,
                                    extent=extent,
                                    figsize=(6, 5.25),
                                    cbar_pad=0.05,
                                    pole=pole
                                )
                            else:
                                # Use regular projection for non-polar regions
                                plot_ensemble_summary(
                                    ds_gt=ds_gt,
                                    ds_sample=first_sample_ds,
                                    ds_mean=ds_mean,
                                    ds_std=ds_std,
                                    output_path=out_region_summary,
                                    vmin_vmax=regional_vmin_vmax,
                                    std_vmin_vmax=regional_std_vmin_vmax,
                                    extent=extent,
                                    figsize=(6, 4.95),
                                    cbar_pad=0.05
                                )
                            output_files.append(out_region_summary)
                            print(f"  Generated regional summary ({region_idx+1}/{len(region_extents)}): {out_region_summary}")
                
                print()
                
            except Exception as e:
                print(f"  ✗ Processing failed for {base_id}: {e}")
                continue
        
        print(f"{'='*60}")
        print(f"Ensemble visualization complete! Generated {len(output_files)} files")
        print(f"{'='*60}\n")
        
        return output_files
    
    def _process_single_sample_rmse(self, file_path: str, ground_truth_dir: str) -> Optional[dict]:
        """
        Helper method to process a single sample for RMSE calculation
        """
        file_name = os.path.basename(file_path)
        sample_id = os.path.splitext(file_name)[0]
        
        # Construct ground truth path (reuse logic)
        gt_filename = sample_id + '.nc'
        gt_path = os.path.join(ground_truth_dir, gt_filename)
        
        # Logic to find GT file if name mismatch
        if not os.path.exists(gt_path):
            if sample_id.startswith('sample_'):
                alt_id = sample_id.replace('sample_', '')
                alt_path = os.path.join(ground_truth_dir, alt_id + '.nc')
                if os.path.exists(alt_path):
                    gt_path = alt_path
            elif not sample_id.startswith('sample_'):
                 alt_path = os.path.join(ground_truth_dir, 'sample_' + gt_filename)
                 if os.path.exists(alt_path):
                     gt_path = alt_path

        if not os.path.exists(gt_path):
            # Silent skip if GT missing
            return None
            
        try:
            # Load sample
            sample_data = self._load_sample(file_path)
            
            # Load GT
            gt_data_list = []
            with xr.open_dataset(gt_path) as ds_gt_raw:
                for var in self.variables:
                    if var not in ds_gt_raw:
                        raise ValueError(f"Variable {var} not found in GT file")
                    var_data = ds_gt_raw[var].values
                    if var_data.ndim == 3 and var_data.shape[0] == 1:
                        var_data = var_data[0]
                    gt_data_list.append(var_data)
            
            gt_data = np.array(gt_data_list)
            
            # Align shapes (padding)
            if sample_data.shape != gt_data.shape:
                if (sample_data.shape[0] == gt_data.shape[0] and 
                    sample_data.shape[2] == gt_data.shape[2] and 
                    sample_data.shape[1] > gt_data.shape[1]):
                    pad_h = sample_data.shape[1] - gt_data.shape[1]
                    gt_data = np.pad(gt_data, ((0, 0), (pad_h, 0), (0, 0)), 
                                   mode='constant', constant_values=np.nan)
            
            if sample_data.shape != gt_data.shape:
                print(f"  Warning: Shape mismatch for {sample_id}, skipping.")
                return None
                
            # Calculate RMSE per variable
            # Apply land mask (where GT is NaN)
            land_mask = np.isnan(gt_data)
            
            # Squared difference
            diff_sq = (sample_data - gt_data) ** 2
            
            # Set land points to NaN in diff_sq so they are ignored in mean
            diff_sq[land_mask] = np.nan
            
            # Calculate RMSE for each channel
            # axis=(1, 2) means spatial dimensions (C, H, W) -> (C,)
            mse_per_var = np.nanmean(diff_sq, axis=(1, 2))
            rmse_per_var = np.sqrt(mse_per_var)
            
            return {
                'sample_id': sample_id,
                'rmse_per_var': rmse_per_var,
                'row': [sample_id] + rmse_per_var.tolist()
            }
                
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            return None

    def calculate_all_samples_rmse(self, 
                                 ground_truth_dir: str = './data/glorys12/val',
                                 output_csv: str = None,
                                 num_workers: int = 32) -> None:
        """
        Calculate RMSE for all samples against ground truth for each variable
        and export to CSV.
        
        Args:
            ground_truth_dir (str): Directory containing ground truth files
            output_csv (str): Output CSV file name (saved to current working directory)
            num_workers (int): Number of threads for parallel processing
        """
        if not os.path.exists(ground_truth_dir):
            print(f"Error: Ground truth directory does not exist: {ground_truth_dir}")
            return
        
        if output_csv is None:
            output_csv = os.path.join(self.output_dir, 'rmse_results.csv')

        print(f"Calculating RMSE for all {len(self.sample_files)} samples with {num_workers} threads...")
        
        results = []
        header = ['sample_id'] + self.variables
        
        # Track sums and counts for average calculation
        all_rmses = [] 
        
        # Use ProcessPoolExecutor instead of ThreadPoolExecutor to avoid Segmentation Faults
        # caused by non-thread-safe underlying libraries (like NetCDF4/HDF5)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._process_single_sample_rmse, fp, ground_truth_dir) for fp in self.sample_files]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Calculating RMSE"):
                result = future.result()
                if result is not None:
                    results.append(result['row'])
                    all_rmses.append(result['rmse_per_var'])

        # Sort results by sample_id
        results.sort(key=lambda x: x[0])
        
        # Export to CSV
        output_path = os.path.join(os.getcwd(), output_csv)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(results)
            
        print(f"\nRMSE results exported to: {output_path}")
        
        # Print Average RMSE
        if all_rmses:
            all_rmses_arr = np.array(all_rmses)
            avg_rmse = np.mean(all_rmses_arr, axis=0)
            print("\nAverage RMSE per variable:")
            for var, rmse in zip(self.variables, avg_rmse):
                print(f"  {var}: {rmse:.6f}")
        else:
            print("No valid samples processed.")

    def _get_ensemble_groups(self) -> dict:
        groups = {}
        for file_path in self.sample_files:
            file_name = os.path.basename(file_path)
            sample_id = os.path.splitext(file_name)[0]
            match = re.search(r'_r\d+$', sample_id)
            if match:
                base_id = sample_id[:match.start()]
            else:
                base_id = sample_id
            groups.setdefault(base_id, []).append(file_path)
        return groups

    def _process_single_ensemble_rmse(self, base_id: str, file_paths: List[str], ground_truth_dir: str, num_ensemble_members: Optional[int] = None) -> Optional[dict]:
        gt_filename = base_id + '.nc'
        gt_path = os.path.join(ground_truth_dir, gt_filename)

        try:
            selected_paths = file_paths
            if num_ensemble_members is not None:
                if num_ensemble_members <= 0:
                    print(f"  Warning: num_ensemble_members <= 0 for {base_id}, skipping.")
                    return None
                selected_paths = sorted(file_paths)[:num_ensemble_members]
            if not selected_paths:
                print(f"  Warning: No ensemble members for {base_id}, skipping.")
                return None

            sample_list = [self._load_sample(fp) for fp in selected_paths]
            sample_stack = np.stack(sample_list, axis=0)

            nan_mask_ref = np.isnan(sample_stack[0])
            if not np.all(np.isnan(sample_stack) == nan_mask_ref):
                raise ValueError(f"NaN mask mismatch in ensemble members for {base_id}")

            valid_count = np.sum(~np.isnan(sample_stack), axis=0)
            sample_sum = np.nansum(sample_stack, axis=0)
            ensemble_mean = np.divide(
                sample_sum,
                valid_count,
                out=np.full_like(sample_sum, np.nan),
                where=valid_count > 0
            )
            
            gt_data_list = []
            with xr.open_dataset(gt_path) as ds_gt_raw:
                for var in self.variables:
                    if var not in ds_gt_raw:
                        raise ValueError(f"Variable {var} not found in GT file")
                    var_data = ds_gt_raw[var].values
                    if var_data.ndim == 3 and var_data.shape[0] == 1:
                        var_data = var_data[0]
                    gt_data_list.append(var_data)
            
            gt_data = np.array(gt_data_list)
            
            if ensemble_mean.shape != gt_data.shape:
                if (ensemble_mean.shape[0] == gt_data.shape[0] and 
                    ensemble_mean.shape[2] == gt_data.shape[2] and 
                    ensemble_mean.shape[1] > gt_data.shape[1]):
                    pad_h = ensemble_mean.shape[1] - gt_data.shape[1]
                    gt_data = np.pad(gt_data, ((0, 0), (pad_h, 0), (0, 0)), 
                                   mode='constant', constant_values=np.nan)
            
            if ensemble_mean.shape != gt_data.shape:
                print(f"  Warning: Shape mismatch for {base_id}, skipping.")
                return None
                
            land_mask = np.isnan(gt_data)
            diff_sq = (ensemble_mean - gt_data) ** 2
            diff_sq[land_mask] = np.nan
            mse_per_var = np.nanmean(diff_sq, axis=(1, 2))
            rmse_per_var = np.sqrt(mse_per_var)
            
            return {
                'sample_id': base_id,
                'rmse_per_var': rmse_per_var,
                'row': [base_id] + rmse_per_var.tolist()
            }
                
        except Exception as e:
            print(f"Error processing {base_id}: {e}")
            return None

    def calculate_ensemble_samples_rmse(self, 
                                        ground_truth_dir: str = './data/glorys12/test',
                                        output_csv_name: str = 'rmse_results_ensemble.csv',
                                        num_workers: int = 32,
                                        num_ensemble_members: Optional[int] = None,
                                        sample_ratio: float = 1.0) -> None:
        if not os.path.exists(ground_truth_dir):
            print(f"Error: Ground truth directory does not exist: {ground_truth_dir}")
            return

        output_csv = os.path.join(self.output_dir, output_csv_name)

        ensemble_groups = self._get_ensemble_groups()
        base_ids = sorted(ensemble_groups.keys())
        if sample_ratio <= 0:
            print("Error: sample_ratio must be greater than 0.")
            return
        if sample_ratio > 1:
            sample_ratio = 1.0
        if sample_ratio < 1:
            sample_count = max(1, int(len(base_ids) * sample_ratio))
            rng = np.random.default_rng()
            base_ids = rng.choice(base_ids, size=sample_count, replace=False).tolist()
        print(f"Calculating ensemble RMSE for {len(base_ids)} groups with {num_workers} threads...")
        
        results = []
        header = ['sample_id'] + self.variables
        all_rmses = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(self._process_single_ensemble_rmse, base_id, ensemble_groups[base_id], ground_truth_dir, num_ensemble_members)
                for base_id in base_ids
            ]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Calculating Ensemble RMSE"):
                result = future.result()
                if result is not None:
                    results.append(result['row'])
                    all_rmses.append(result['rmse_per_var'])

        results.sort(key=lambda x: x[0])
        output_path = os.path.join(os.getcwd(), output_csv)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(results)
            
        print(f"\nEnsemble RMSE results exported to: {output_path}")
        
        if all_rmses:
            all_rmses_arr = np.array(all_rmses)
            avg_rmse = np.mean(all_rmses_arr, axis=0)
            print("\nAverage Ensemble RMSE per variable:")
            for var, rmse in zip(self.variables, avg_rmse):
                print(f"  {var}: {rmse:.6f}")
        else:
            print("No valid ensemble groups processed.")

    def compute_spatial_ensemble_rmse(self,
                                       ground_truth_dir: str = './data/glorys12/test',
                                       output_file: str = None,
                                       num_workers: int = None,
                                       sample_ratio: float = 1.0) -> dict:
        """
        Compute spatial RMSE for each position (latitude, longitude) using ensemble mean.
        
        For each variable (thetao, so, uo, vo), collect ensemble mean predictions and 
        ground truth at each position across all depth layers and time points, 
        then compute RMSE at each spatial location.
        
        Args:
            ground_truth_dir: Directory containing ground truth NetCDF files
            output_file: Output npy file path, default {output_dir}/spatial_ensemble_rmse.npy
            num_workers: Number of worker processes for parallel processing,
                        default None (use all CPU cores - 1)
            sample_ratio: Ratio of ensemble groups to use (0.0 to 1.0), default 1.0 (use all)
            
        Returns:
            dict: Dictionary containing:
                - 'rmse': dict of {variable: rmse_array} with shape (n_lat, n_lon)
                - 'count': dict of {variable: count_array} with shape (n_lat, n_lon)
                - 'latitude': latitude coordinates array
                - 'longitude': longitude coordinates array
                - 'variables': list of variable names
        """
        if output_file is None:
            output_file = os.path.join(self.output_dir, 'spatial_ensemble_rmse.npy')
        
        if not os.path.exists(ground_truth_dir):
            raise FileNotFoundError(f"Ground truth directory not found: {ground_truth_dir}")
        
        # Get ensemble groups
        ensemble_groups = self._get_ensemble_groups()
        base_ids = sorted(ensemble_groups.keys())
        
        if not base_ids:
            raise ValueError("No ensemble groups found")
        
        # Sample ensemble groups if sample_ratio < 1.0
        if sample_ratio < 1.0:
            n_total = len(base_ids)
            n_sample = max(1, int(n_total * sample_ratio))
            # Use fixed seed for reproducibility
            random.seed(42)
            base_ids = random.sample(base_ids, n_sample)
            random.seed()  # Reset random seed
            print(f"Sampling {n_sample}/{n_total} ensemble groups ({sample_ratio*100:.1f}%)")
        
        # Get spatial dimensions from first available ground truth file
        sample_gt_file = None
        for base_id in base_ids[:10]:  # Try first 10 groups
            gt_filename = base_id + '.nc'
            gt_path = os.path.join(ground_truth_dir, gt_filename)
            if not os.path.exists(gt_path) and base_id.startswith('sample_'):
                gt_path = os.path.join(ground_truth_dir, base_id.replace('sample_', '') + '.nc')
            if os.path.exists(gt_path):
                sample_gt_file = gt_path
                break
        
        if sample_gt_file is None:
            raise FileNotFoundError(f"No ground truth files found in {ground_truth_dir}")
        
        with xr.open_dataset(sample_gt_file) as ds:
            lat_coords = ds.latitude.values
            lon_coords = ds.longitude.values
        
        n_lat, n_lon = len(lat_coords), len(lon_coords)
        
        # Set default num_workers
        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)
        
        print(f"Computing spatial ensemble RMSE...")
        print(f"Spatial grid: {n_lat} x {n_lon} = {n_lat * n_lon} positions")
        print(f"Ensemble groups: {len(base_ids)}")
        print(f"Using {num_workers} worker processes")
        
        # Prepare arguments for parallel processing
        args_list = [
            (base_id, ensemble_groups[base_id], ground_truth_dir, self.variables)
            for base_id in base_ids
        ]
        
        # Process ensemble groups in parallel
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(_process_ensemble_group_for_spatial_rmse, args_list),
                total=len(args_list),
                desc="Processing ensemble groups"
            ))
        
        # Collect valid results and errors
        valid_results = []
        errors = []
        for r in results:
            if r is None:
                errors.append('None result')
            elif 'error' in r:
                errors.append(r['error'])
            else:
                valid_results.append(r)
        
        if not valid_results:
            print(f"All {len(results)} groups failed. Sample errors:")
            for err in errors[:10]:
                print(f"  - {err}")
            raise ValueError("No valid ensemble groups processed")
        
        print(f"Successfully processed {len(valid_results)} groups, {len(errors)} failed")
        
        # Initialize accumulators for each variable
        sum_sq_error = {}
        count = {}
        for var in self.variables:
            sum_sq_error[var] = np.zeros((n_lat, n_lon), dtype=np.float64)
            count[var] = np.zeros((n_lat, n_lon), dtype=np.int64)
        
        # Accumulate results from all groups
        for group_result in valid_results:
            for var in self.variables:
                if var in group_result:
                    sum_sq_error[var] += group_result[var]['sum_sq_error']
                    count[var] += group_result[var]['count']
        
        # Compute RMSE for each variable
        rmse = {}
        for var in self.variables:
            rmse[var] = np.zeros((n_lat, n_lon), dtype=np.float64)
            valid_mask = count[var] > 0
            rmse[var][valid_mask] = np.sqrt(sum_sq_error[var][valid_mask] / count[var][valid_mask])
        
        # Prepare output dictionary
        result_dict = {
            'rmse': rmse,
            'count': count,
            'latitude': lat_coords,
            'longitude': lon_coords,
            'variables': self.variables
        }
        
        # Print summary statistics
        print(f"\nSpatial ensemble RMSE statistics:")
        for var in self.variables:
            var_rmse = rmse[var]
            valid_mask = count[var] > 0
            if np.any(valid_mask):
                print(f"  {var}: mean_rmse={np.nanmean(var_rmse):.4f}, "
                      f"max_rmse={np.nanmax(var_rmse):.4f}, "
                      f"valid_positions={np.sum(valid_mask)}")
        
        # Save to npy file
        np.save(output_file, result_dict)
        print(f"\nResults saved to: {output_file}")
        
        return result_dict

    def plot_spatial_rmse_and_srmse(self,
                                     spatial_rmse_file: str = None,
                                     spatial_std_csv: str = './data/glorys12/test_spatial_std.csv',
                                     output_dir: str = None,
                                     dpi: int = 600,
                                     figsize: tuple = (8, 7)) -> List[str]:
        """
        Load spatial ensemble RMSE and plot both RMSE and SRMSE distributions.
        
        Creates two 2x2 subplot figures:
        1. RMSE: Raw RMSE values for each variable
        2. SRMSE: RMSE normalized by test set spatial standard deviation
        
        SRMSE = RMSE / spatial_std
        
        Args:
            spatial_rmse_file: Path to spatial_ensemble_rmse.npy file.
                              If None, will look for 'spatial_ensemble_rmse.npy' in self.output_dir
            spatial_std_csv: Path to spatial standard deviation CSV file
            output_dir: Directory to save plots. If None, uses self.output_dir/spatial_rmse_maps
            dpi: Image resolution
            figsize: Figure size (width, height) in inches, default is (8, 7)

        Returns:
            List of saved plot file paths [rmse_path, srmse_path]
        """
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import pandas as pd
        
        # Set Times New Roman font
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Find spatial RMSE file
        if spatial_rmse_file is None:
            spatial_rmse_file = os.path.join(self.output_dir, 'spatial_ensemble_rmse.npy')
        
        if not os.path.exists(spatial_rmse_file):
            raise FileNotFoundError(f"Spatial RMSE file not found: {spatial_rmse_file}")
        
        # Load RMSE data
        data = np.load(spatial_rmse_file, allow_pickle=True).item()
        rmse = data['rmse']
        lat_coords = data['latitude']
        lon_coords = data['longitude']
        variables = data['variables']
        
        # Load spatial std data from CSV
        print(f"\nLoading spatial standard deviation from: {spatial_std_csv}")
        std_df = pd.read_csv(spatial_std_csv)
        
        # Create std arrays for each variable
        n_lat, n_lon = len(lat_coords), len(lon_coords)
        spatial_std = {}
        for var in variables:
            spatial_std[var] = np.full((n_lat, n_lon), np.nan)
        
        # Build latitude/longitude to index mapping
        lat_to_idx = {lat: idx for idx, lat in enumerate(lat_coords)}
        lon_to_idx = {lon: idx for idx, lon in enumerate(lon_coords)}
        
        # Fill std arrays
        for _, row in std_df.iterrows():
            var = row['variable']
            lat = row['latitude']
            lon = row['longitude']
            std_val = row['std']
            
            if var in spatial_std and lat in lat_to_idx and lon in lon_to_idx:
                lat_idx = lat_to_idx[lat]
                lon_idx = lon_to_idx[lon]
                spatial_std[var][lat_idx, lon_idx] = std_val
        
        # Calculate SRMSE
        srmse = {}
        for var in variables:
            std_mask = spatial_std[var] > 0
            srmse[var] = np.full_like(rmse[var], np.nan)
            srmse[var][std_mask] = rmse[var][std_mask] / spatial_std[var][std_mask]
        
        # Set output directory
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, 'spatial_rmse_maps')
        os.makedirs(output_dir, exist_ok=True)
        
        # Variable labels, units and colormaps
        var_labels = {
            'thetao': 'Temperature',
            'so': 'Salinity',
            'uo': 'Eastward Velocity',
            'vo': 'Northward Velocity'
        }
        var_units = {
            'thetao': '°C',
            'so': 'psu',
            'uo': 'm/s',
            'vo': 'm/s'
        }
        var_cmaps = {
            'thetao': 'YlOrRd',
            'so': 'YlGnBu',
            'uo': 'Greys',
            'vo': 'PuBu'
        }
        
        saved_files = []
        lon_mesh, lat_mesh = np.meshgrid(lon_coords, lat_coords)
        
        # Plot 1: RMSE
        print(f"\nPlotting spatial RMSE for {len(variables)} variables...")
        fig1, axes1 = plt.subplots(2, 2, figsize=figsize,
                                   subplot_kw={'projection': ccrs.PlateCarree()},
                                   gridspec_kw={'wspace': 0, 'hspace': 0.2})
        axes1 = axes1.flatten()
        
        for idx, var in enumerate(variables):
            if idx >= len(axes1):
                break
            ax = axes1[idx]
            var_rmse = rmse[var]

            # 陆地填充改为白色
            ax.add_feature(cfeature.LAND, facecolor='white', edgecolor='black', linewidth=0.5)
            ax.coastlines(resolution='110m', linewidth=0.5)

            # 坐标标签控制：左边一列显示左边坐标，下面一行显示下面坐标
            # idx 0, 2 是左边一列；idx 2, 3 是下面一行
            draw_left = idx in [0, 2]
            draw_bottom = idx in [2, 3]
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.left_labels = draw_left
            gl.right_labels = False
            gl.top_labels = False
            gl.bottom_labels = draw_bottom

            im = ax.pcolormesh(lon_mesh, lat_mesh, var_rmse,
                              transform=ccrs.PlateCarree(),
                              cmap=var_cmaps.get(var, 'viridis'),
                              shading='auto', vmin=0)

            cbar = plt.colorbar(im, ax=ax, pad=0.03, aspect=25)
            cbar.set_label(f"RMSE ({var_units.get(var, '')})", fontsize=9)
            ax.set_title(var_labels.get(var, var), fontsize=13)

        rmse_path = os.path.join(output_dir, 'spatial_rmse_all_variables.jpg')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(rmse_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        saved_files.append(rmse_path)
        print(f"  Saved: {rmse_path}")
        
        # Plot 2: SRMSE
        print(f"\nPlotting spatial SRMSE for {len(variables)} variables...")
        fig2, axes2 = plt.subplots(2, 2, figsize=figsize,
                                   subplot_kw={'projection': ccrs.PlateCarree()},
                                   gridspec_kw={'wspace': 0, 'hspace': 0.2})
        axes2 = axes2.flatten()
        
        for idx, var in enumerate(variables):
            if idx >= len(axes2):
                break
            ax = axes2[idx]
            var_srmse = srmse[var]

            # 陆地填充改为白色
            ax.add_feature(cfeature.LAND, facecolor='white', edgecolor='black', linewidth=0.5)
            ax.coastlines(resolution='110m', linewidth=0.5)

            # 坐标标签控制：左边一列显示左边坐标，下面一行显示下面坐标
            draw_left = idx in [0, 2]
            draw_bottom = idx in [2, 3]
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.left_labels = draw_left
            gl.right_labels = False
            gl.top_labels = False
            gl.bottom_labels = draw_bottom

            # Calculate 99.9th percentile as upper bound
            vmax = np.nanpercentile(var_srmse, 99.9)
            if vmax <= 0 or np.isnan(vmax):
                vmax = 1.0

            im = ax.pcolormesh(lon_mesh, lat_mesh, var_srmse,
                              transform=ccrs.PlateCarree(),
                              cmap=var_cmaps.get(var, 'viridis'),
                              shading='auto', vmin=0, vmax=vmax)

            cbar = plt.colorbar(im, ax=ax, pad=0.03, aspect=25, extend='max')
            cbar.set_label("SRMSE", fontsize=9)
            ax.set_title(var_labels.get(var, var), fontsize=13)

        srmse_path = os.path.join(output_dir, 'spatial_srmse_all_variables.jpg')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(srmse_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        saved_files.append(srmse_path)
        print(f"  Saved: {srmse_path}")
        
        return saved_files

    def _read_rmse_csv(self, file_path: str) -> Tuple[List[str], dict]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"RMSE file does not exist: {file_path}")

        with open(file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)

            if not header or len(header) < 2:
                return [], {}

            variables = header[1:]
            data = {var: [] for var in variables}

            for row in reader:
                if len(row) < len(header):
                    continue
                for i, var in enumerate(variables, start=1):
                    try:
                        val = float(row[i])
                    except (ValueError, TypeError):
                        continue
                    if np.isfinite(val):
                        data[var].append(val)

        return variables, data

    def _read_rmse_csv_with_depth(self, file_path: str) -> Tuple[List[str], dict]:
        """
        Read RMSE CSV file and organize data by depth layer.
        
        Args:
            file_path: Path to RMSE CSV file
            
        Returns:
            Tuple of (variables list, dict with depth_index as keys containing RMSE data)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"RMSE file does not exist: {file_path}")

        with open(file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)

            if not header or len(header) < 2:
                return [], {}

            variables = header[1:]
            # Initialize data structure: depth_index -> {var: [values]}
            depth_data = {}

            for row in reader:
                if len(row) < len(header):
                    continue
                
                sample_id = row[0]
                # Extract depth index from sample_id (format: YYYYMMDD_d{depth:02d})
                depth_idx = None
                if '_d' in sample_id:
                    try:
                        depth_str = sample_id.split('_d')[1]
                        depth_idx = int(depth_str[:2])  # Get first 2 digits
                    except (ValueError, IndexError):
                        continue
                
                if depth_idx is None:
                    continue
                
                if depth_idx not in depth_data:
                    depth_data[depth_idx] = {var: [] for var in variables}
                
                for i, var in enumerate(variables, start=1):
                    try:
                        val = float(row[i])
                    except (ValueError, TypeError):
                        continue
                    if np.isfinite(val):
                        depth_data[depth_idx][var].append(val)

        return variables, depth_data

    def _read_test_std_statistics(self, std_csv_path: str) -> dict:
        """
        Read test set standard deviations from CSV file.
        
        Args:
            std_csv_path: Path to test_std_statistics.csv file
            
        Returns:
            dict: Nested dict with structure {depth_index: {variable: std_mean}}
        """
        if not os.path.exists(std_csv_path):
            return {}
        
        stds = {}
        with open(std_csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    depth_idx = int(row['depth_index'])
                    var = row['variable']
                    std_mean = float(row['spatial_std_mean'])
                    
                    if depth_idx not in stds:
                        stds[depth_idx] = {}
                    stds[depth_idx][var] = std_mean
                except (ValueError, KeyError):
                    continue
        
        return stds

    def _plot_rmse_boxplot_internal(self,
                                    variables: List[str],
                                    depth_data: dict,
                                    depth_stds: dict,
                                    output_path: str,
                                    figsize: Tuple[int, int],
                                    dpi: int,
                                    normalized: bool = True) -> str:
        """
        Internal method to plot RMSE boxplot (normalized or raw).
        Each variable is plotted in a separate subplot.
        
        Args:
            variables: List of variable names
            depth_data: RMSE data organized by depth
            depth_stds: Standard deviations for normalization (can be empty for raw RMSE)
            output_path: Path to save the plot
            figsize: Figure size
            dpi: Image resolution
            normalized: If True, normalize by test set std; if False, plot raw RMSE
            
        Returns:
            Path to saved plot file
        """
        import matplotlib.pyplot as plt
        from matplotlib import font_manager as fm
        
        # Set Times New Roman font
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Variable labels and colors
        var_labels = {
            'thetao': 'Temperature',
            'so': 'Salinity',
            'uo': 'Eastward Velocity',
            'vo': 'Northward Velocity'
        }
        var_colors = {
            'thetao': '#ed8687',
            'so': '#ffd19d',
            'uo': '#bed2c6',
            'vo': '#c9e0e5'
        }
        
        # Prepare data for boxplot
        sorted_depths = sorted(depth_data.keys())
        num_depths = len(sorted_depths)
        num_vars = len(variables)
        
        # Create figure with subplots for each variable (1x4 horizontal layout)
        fig, axes = plt.subplots(1, 4, figsize=(7, 5))
        axes = axes.flatten()
        
        # Set labels and title
        if normalized:
            xlabel = 'SRMSE'
        else:
            xlabel = 'RMSE'
        
        for var_idx, var in enumerate(variables):
            ax = axes[var_idx]
            var_color = var_colors.get(var, '#3498DB')
            
            # Collect boxplot data for this variable
            boxplot_data = []
            positions = []
            
            for depth_idx in sorted_depths:
                rmse_values = depth_data[depth_idx].get(var, [])
                
                if rmse_values:
                    if normalized:
                        # Normalize by test set std for this specific depth and variable
                        std_val = depth_stds.get(depth_idx, {}).get(var) if depth_stds else None
                        if std_val is not None and std_val > 0:
                            plot_values = [v / std_val for v in rmse_values]
                        else:
                            plot_values = rmse_values
                    else:
                        # Use raw RMSE values
                        plot_values = rmse_values
                    
                    boxplot_data.append(plot_values)
                else:
                    boxplot_data.append([])
                
                positions.append(depth_idx)
            
            # Create horizontal boxplot for this variable (vert=0 means horizontal)
            bp = ax.boxplot(boxplot_data, positions=positions, patch_artist=True,
                           showmeans=True, meanline=True, widths=1.0,
                           flierprops=dict(marker='.', markersize=2, alpha=0.3),
                           vert=False)
            
            # Color the boxes
            for patch in bp['boxes']:
                patch.set_facecolor(var_color)
                patch.set_alpha(1.0)
            
            # Style the mean line
            for line in bp['means']:
                line.set_color('red')
                line.set_linewidth(1.5)
            
            # Set labels for each subplot (XY swapped)
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_title(var_labels.get(var, var), fontsize=13)
            ax.grid(True, alpha=0.3, axis='both')
            
            # Invert y-axis so smaller depth values are at the top
            ax.invert_yaxis()
            
            # Only show y-axis label and tick labels for the first subplot
            if var_idx == 0:
                ax.set_ylabel('Depth Layer', fontsize=10)
                ax.set_yticks(positions[::max(1, len(positions)//10)])  # Show ~10 ticks
                ax.set_yticklabels([str(d) for d in positions[::max(1, len(positions)//10)]])
            else:
                ax.set_ylabel('')
                # Keep yticks for grid lines but hide labels
                ax.set_yticks(positions[::max(1, len(positions)//10)])
                ax.set_yticklabels([])
            
            # Remove top and right spines for all subplots
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Remove left spine for subplots other than the first one
            if var_idx != 0:
                ax.spines['left'].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Boxplot saved to: {output_path}")
        return output_path

    def plot_srmse_boxplot_by_depth(self,
                                    rmse_csv_path: str = None,
                                    output_path_normalized: str = None,
                                    output_path_raw: str = None,
                                    test_std_csv_path: str = './data/glorys12/test_std_statistics.csv',
                                    figsize: Tuple[int, int] = (16, 10),
                                    dpi: int = 300) -> List[str]:
        """
        Plot RMSE boxplot for all depth layers in one figure.
        
        Generates two versions:
        1. Normalized version (SRMSE): RMSE normalized by test set standard deviation
        2. Raw version: Original RMSE values without normalization
        
        Args:
            rmse_csv_path: Path to RMSE CSV file generated by calculate_ensemble_samples_rmse.
                          If None, will look for 'rmse_results_ensemble_10.csv' in self.output_dir
            output_path_normalized: Path to save normalized plot. 
                                   If None, uses self.output_dir/srmse_boxplot_by_depth.png
            output_path_raw: Path to save raw RMSE plot.
                            If None, uses self.output_dir/rmse_boxplot_by_depth.png
            test_std_csv_path: Path to test set standard deviation CSV file (test_std_statistics.csv)
            figsize: Figure size
            dpi: Image resolution
            
        Returns:
            List of saved plot file paths [normalized_path, raw_path]
        """
        # Set RMSE CSV path
        if rmse_csv_path is None:
            rmse_csv_path = os.path.join(self.output_dir, 'rmse_results_ensemble_10.csv')
        
        if not os.path.exists(rmse_csv_path):
            raise FileNotFoundError(f"RMSE CSV file not found: {rmse_csv_path}")
        
        # Set output paths
        if output_path_normalized is None:
            output_path_normalized = os.path.join(self.output_dir, 'srmse_boxplot_by_depth.pdf')
        if output_path_raw is None:
            output_path_raw = os.path.join(self.output_dir, 'rmse_boxplot_by_depth.pdf')
        
        # Ensure output directories exist
        for path in [output_path_normalized, output_path_raw]:
            output_dir = os.path.dirname(path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
        
        # Read RMSE data organized by depth
        variables, depth_data = self._read_rmse_csv_with_depth(rmse_csv_path)
        
        if not depth_data:
            print("Warning: No valid depth data found in RMSE file")
            return []
        
        # Read test set standard deviations for normalization (per depth, per variable)
        depth_stds = self._read_test_std_statistics(test_std_csv_path)
        if not depth_stds:
            print(f"Warning: Could not load test std statistics from {test_std_csv_path}")
        
        saved_files = []
        
        # Plot 1: Normalized version (SRMSE)
        path_norm = self._plot_rmse_boxplot_internal(
            variables=variables,
            depth_data=depth_data,
            depth_stds=depth_stds,
            output_path=output_path_normalized,
            figsize=figsize,
            dpi=dpi,
            normalized=True
        )
        saved_files.append(path_norm)
        
        # Plot 2: Raw RMSE version
        path_raw = self._plot_rmse_boxplot_internal(
            variables=variables,
            depth_data=depth_data,
            depth_stds={},  # Empty dict means no normalization
            output_path=output_path_raw,
            figsize=figsize,
            dpi=dpi,
            normalized=False
        )
        saved_files.append(path_raw)
        
        print(f"\nTotal {len(saved_files)} boxplot plots saved:")
        print(f"  1. Normalized (SRMSE): {path_norm}")
        print(f"  2. Raw (RMSE): {path_raw}")
        
        return saved_files

    def _read_statistics_std(self, statistics_path: str) -> dict:
        if not statistics_path or not os.path.exists(statistics_path):
            return {}
        with open(statistics_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        stds = {}
        for var, values in stats.items():
            if isinstance(values, dict) and 'std' in values:
                stds[var] = values['std']
        return stds

    def plot_rmse_boxplot_comparison(self,
                                     rmse_files: Union[List[str], dict],
                                     output_path: str = None,
                                     figsize: Tuple[int, int] = (12, 12),
                                     dpi: int = 300,
                                     title: Optional[str] = None,
                                     statistics_train_path: Optional[str] = './data/glorys12/statistics_train.json',
                                     show_srmse_summary: bool = True) -> Optional[str]:
        if not rmse_files:
            print("Warning: rmse_files is empty")
            return None

        if isinstance(rmse_files, dict):
            model_to_file = rmse_files
        else:
            model_to_file = {Path(fp).stem: fp for fp in rmse_files}

        rmse_data_by_model = {}
        for model_name, file_path in model_to_file.items():
            if not os.path.exists(file_path):
                print(f"Warning: RMSE file not found: {file_path}")
                continue
            _, data = self._read_rmse_csv(file_path)
            for var in self.variables:
                data.setdefault(var, [])
            rmse_data_by_model[model_name] = data

        if not rmse_data_by_model:
            print("Warning: No valid RMSE data found")
            return None

        if output_path is None:
            output_path = os.path.join(self.output_dir, 'rmse_boxplot_comparison.png')

        stds = self._read_statistics_std(statistics_train_path)
        srmse_data_by_model = {}
        if stds:
            for model_name, data in rmse_data_by_model.items():
                srmse_values = {var: [] for var in self.variables}
                for var in self.variables:
                    std_val = stds.get(var)
                    if std_val is None or std_val == 0:
                        continue
                    for val in data.get(var, []):
                        if np.isfinite(val):
                            srmse_values[var].append(val / std_val)
                srmse_data_by_model[model_name] = srmse_values
        else:
            print(f"Warning: statistics_train file not found or invalid: {statistics_train_path}")

        plot_rmse_boxplots(
            rmse_data_by_model=rmse_data_by_model,
            variables=self.variables,
            output_path=output_path,
            figsize=figsize,
            dpi=dpi,
            title=title,
            srmse_data_by_model=srmse_data_by_model if stds else None,
            show_srmse_summary=show_srmse_summary
        )

        return output_path
    
    def _generate_summary(self, 
                         selected_files: List[str], 
                         output_files: List[str]) -> None:
        """
        Generate processing summary file
        
        Args:
            selected_files (List[str]): List of selected sample files
            output_files (List[str]): List of generated output files
        """
        summary_path = os.path.join(self.output_dir, 'processing_summary.txt')
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Sample Post-Processing Summary\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Processing time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Sample directory: {self.sample_dir}\n")
            f.write(f"Output directory: {self.output_dir}\n")
            f.write(f"Total samples: {len(self.sample_files)}\n")
            f.write(f"Processed samples: {len(selected_files)}\n")
            f.write(f"Generated files: {len(output_files)}\n\n")
            
            f.write("Processed samples:\n")
            for i, file_path in enumerate(selected_files, 1):
                f.write(f"  {i}. {os.path.basename(file_path)}\n")
            
            f.write("\nGenerated files:\n")
            for i, file_path in enumerate(output_files, 1):
                f.write(f"  {i}. {file_path}\n")
        
        print(f"Processing summary saved: {summary_path}")
    
    def visualize_all_samples(self,
                             max_samples: Optional[int] = None,
                             **kwargs) -> List[str]:
        """
        Visualize all samples (or specified number of samples)
        
        Args:
            max_samples (int, optional): Maximum number of samples to process. None means process all samples
            **kwargs: Additional parameters to pass to random_sample_and_visualize
            
        Returns:
            List[str]: List of generated file paths
        """
        n_samples = len(self.sample_files)
        if max_samples is not None:
            n_samples = min(n_samples, max_samples)
        
        # Process sequentially without random sampling
        return self.random_sample_and_visualize(n_samples=n_samples, **kwargs)
    
    def get_statistics(self) -> dict:
        """
        Get statistics for all samples
        
        Returns:
            dict: Statistics dictionary
        """
        print("Computing statistics for all samples...")
        
        stats = {
            'total_samples': len(self.sample_files),
            'variables': {}
        }
        
        # Initialize variable statistics
        for var_name in self.variables:
            stats['variables'][var_name] = {
                'mean': [],
                'std': [],
                'min': [],
                'max': []
            }
        
        # Iterate through all samples
        for file_path in self.sample_files:
            try:
                data = self._load_sample(file_path)
                
                for i, var_name in enumerate(self.variables[:data.shape[0]]):
                    var_data = data[i]
                    stats['variables'][var_name]['mean'].append(np.mean(var_data))
                    stats['variables'][var_name]['std'].append(np.std(var_data))
                    stats['variables'][var_name]['min'].append(np.min(var_data))
                    stats['variables'][var_name]['max'].append(np.max(var_data))
                    
            except Exception as e:
                print(f"Warning: Unable to process file {file_path}: {e}")
                continue
        
        # Calculate summary statistics
        for var_name in stats['variables']:
            var_stats = stats['variables'][var_name]
            if len(var_stats['mean']) > 0:
                stats['variables'][var_name]['avg_mean'] = np.mean(var_stats['mean'])
                stats['variables'][var_name]['avg_std'] = np.mean(var_stats['std'])
                stats['variables'][var_name]['global_min'] = np.min(var_stats['min'])
                stats['variables'][var_name]['global_max'] = np.max(var_stats['max'])
        
        # Print statistics summary
        print("\nStatistics Summary:")
        print(f"Total samples: {stats['total_samples']}")
        for var_name in self.variables:
            if var_name in stats['variables']:
                var_stats = stats['variables'][var_name]
                if 'avg_mean' in var_stats:
                    print(f"\n{var_name}:")
                    print(f"  Average mean: {var_stats['avg_mean']:.4f}")
                    print(f"  Average std: {var_stats['avg_std']:.4f}")
                    print(f"  Global min: {var_stats['global_min']:.4f}")
                    print(f"  Global max: {var_stats['global_max']:.4f}")
        
        return stats

    def visualize_vertical_profiles(self,
                                    date_str: str,
                                    ground_truth_dir: str = './data/glorys12/test',
                                    save_format: str = 'png',
                                    dpi: int = 300,
                                    figsize: Tuple[int, int] = (20, 12),
                                    lon_step: float = 20.0,
                                    max_depth_levels: int = 40) -> str:
        """
        Visualize vertical cross-sections (north-south profiles) for a specific date.
        
        For each longitude slice (every lon_step degrees), shows 4 types of plots:
        1. Ground Truth
        2. Ensemble Mean
        3. Ensemble Std
        4. Error (Mean - GT)
        
        All slices are displayed in a 3D layout showing their spatial positions.
        
        Args:
            date_str (str): Date string in format 'YYYYMMDD' (e.g., '20230101')
            ground_truth_dir (str): Directory containing ground truth files
            save_format (str): Image save format, default 'png'
            dpi (int): Image resolution, default 300
            figsize (Tuple[int, int]): Figure size, default (20, 12)
            lon_step (float): Longitude interval for cross-sections, default 20 degrees
            max_depth_levels (int): Maximum number of depth levels to process
            
        Returns:
            str: Path to the saved visualization file
        """
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Group ensemble files by date and depth
        ensemble_groups = self._get_ensemble_groups()
        
        # Filter groups for the specified date
        date_groups = {}
        for base_id, file_paths in ensemble_groups.items():
            # Extract date from base_id (format: [sample_]YYYYMMDD_dXX)
            clean_id = base_id.replace('sample_', '')
            parts = clean_id.split('_')
            if len(parts) >= 2:
                file_date = parts[0]
                depth_idx = int(parts[1][1:])  # Remove 'd' prefix
                if file_date == date_str and depth_idx < max_depth_levels:
                    date_groups[depth_idx] = file_paths
        
        if not date_groups:
            print(f"Error: No samples found for date {date_str}")
            return ""
        
        # Sort by depth index
        sorted_depths = sorted(date_groups.keys())
        n_depths = len(sorted_depths)
        
        print(f"\n{'='*60}")
        print(f"Processing vertical profiles for date: {date_str}")
        print(f"Number of depth levels: {n_depths}")
        print(f"Depth indices: {sorted_depths}")
        print(f"{'='*60}\n")
        
        # Load ground truth for all depth levels
        gt_3d = {}  # {var: (depth, lat, lon)}
        sample_shape = None
        depth_values = []
        
        for var in self.variables:
            gt_3d[var] = []
        
        # Also prepare ensemble data structures
        ensemble_mean_3d = {var: [] for var in self.variables}
        ensemble_std_3d = {var: [] for var in self.variables}
        
        # Load GT file to check existence and get shape
        test_depth_idx = sorted_depths[0]
        test_base_id = f"{date_str}_d{test_depth_idx:02d}"
        gt_path = os.path.join(ground_truth_dir, test_base_id + '.nc')
        if not os.path.exists(gt_path):
            alt_path = os.path.join(ground_truth_dir, 'sample_' + test_base_id + '.nc')
            if os.path.exists(alt_path):
                gt_path = alt_path
        
        if not os.path.exists(gt_path):
            print(f"Error: Ground truth file not found for {test_base_id}")
            return ""
        
        # Process each depth level
        for depth_idx in tqdm(sorted_depths, desc="Loading depth levels"):
            base_id = f"{date_str}_d{depth_idx:02d}"
            file_paths = date_groups[depth_idx]
            
            # Load ensemble members
            member_data_list = []
            for fp in file_paths:
                data = self._load_sample(fp)
                member_data_list.append(data)
            
            if sample_shape is None:
                sample_shape = member_data_list[0].shape
                C, H, W = sample_shape
            
            # Calculate ensemble statistics
            ensemble_stack = np.stack(member_data_list, axis=0)
            ensemble_mean = np.mean(ensemble_stack, axis=0)
            ensemble_std = np.std(ensemble_stack, axis=0)
            
            # Load ground truth
            gt_path = os.path.join(ground_truth_dir, base_id + '.nc')
            if not os.path.exists(gt_path):
                alt_path = os.path.join(ground_truth_dir, 'sample_' + base_id + '.nc')
                if os.path.exists(alt_path):
                    gt_path = alt_path
            
            if not os.path.exists(gt_path):
                print(f"Warning: GT not found for {base_id}, skipping...")
                continue
            
            gt_data_list = []
            with xr.open_dataset(gt_path) as ds_gt_raw:
                # Extract depth value
                if 'depth' in ds_gt_raw:
                    d_val = ds_gt_raw['depth'].values
                    # Handle scalar or array
                    if d_val.ndim == 0:
                        depth_values.append(float(d_val))
                    elif d_val.size > 0:
                        depth_values.append(float(d_val.flatten()[0]))
                    else:
                        depth_values.append(float(depth_idx))
                else:
                    depth_values.append(float(depth_idx))

                for var in self.variables:
                    var_data = ds_gt_raw[var].values
                    if var_data.ndim == 3 and var_data.shape[0] == 1:
                        var_data = var_data[0]
                    gt_data_list.append(var_data)
            gt_data = np.array(gt_data_list)
            
            # Pad GT if necessary
            if gt_data.shape != sample_shape:
                if (gt_data.shape[0] == sample_shape[0] and 
                    gt_data.shape[2] == sample_shape[2] and 
                    sample_shape[1] > gt_data.shape[1]):
                    pad_h = sample_shape[1] - gt_data.shape[1]
                    gt_data = np.pad(gt_data, ((0, 0), (pad_h, 0), (0, 0)), 
                                   mode='constant', constant_values=np.nan)
            
            # Create land mask from GT
            land_mask = np.isnan(gt_data)
            
            # Apply mask to ensemble data
            ensemble_mean_masked = ensemble_mean.copy()
            ensemble_mean_masked[land_mask] = np.nan
            ensemble_std_masked = ensemble_std.copy()
            ensemble_std_masked[land_mask] = np.nan
            
            # Store data for each variable
            for i, var in enumerate(self.variables):
                gt_3d[var].append(gt_data[i])
                ensemble_mean_3d[var].append(ensemble_mean_masked[i])
                ensemble_std_3d[var].append(ensemble_std_masked[i])
        
        # Convert lists to arrays
        for var in self.variables:
            gt_3d[var] = np.array(gt_3d[var])  # (depth, lat, lon)
            ensemble_mean_3d[var] = np.array(ensemble_mean_3d[var])
            ensemble_std_3d[var] = np.array(ensemble_std_3d[var])
        
        # Calculate error
        error_3d = {}
        for var in self.variables:
            error_3d[var] = ensemble_mean_3d[var] - gt_3d[var]
        
        # Get coordinates
        lat_range = (-86, 90)
        lon_range = (-180, 180)
        lat_coords = np.linspace(lat_range[0], lat_range[1], sample_shape[1])
        lon_coords = np.linspace(lon_range[0], lon_range[1], sample_shape[2])
        
        # Create depth coordinates (negative for proper 3D visualization)
        depth_coords = -np.array(depth_values)
        
        # Create output directory
        viz_dir = os.path.join(self.output_dir, 'visualizations_vertical_profiles')
        os.makedirs(viz_dir, exist_ok=True)
        
        output_path = os.path.join(viz_dir, f'vertical_profiles_{date_str}.{save_format}')
        
        print(f"\nGenerating 3D visualization with longitude slices every {lon_step}°...")
        
        # Call the plot function from plot.py
        plot_vertical_profiles_3d(
            gt_3d=gt_3d,
            ensemble_mean_3d=ensemble_mean_3d,
            ensemble_std_3d=ensemble_std_3d,
            error_3d=error_3d,
            lat_coords=lat_coords,
            lon_coords=lon_coords,
            depth_coords=depth_coords,
            variables=self.variables,
            output_path=output_path,
            date_str=date_str,
            lon_step=lon_step,
            figsize=figsize,
            dpi=dpi
        )
        
        print(f"\nVertical profile visualization saved to: {output_path}")
        print(f"{'='*60}\n")
        
        return output_path


def main():
    """
    Usage example
    """
    # Example: Create post-processor and randomly sample for visualization
    sample_dir = './output/results/test/sample_20231223_120000/heun-steps50-image1000-res352x720'
    
    processor = SamplePostProcessor(sample_dir=sample_dir)
    
    # Randomly sample 5 samples and visualize
    output_files = processor.random_sample_and_visualize(
        n_samples=5,
        seed=42,
        export_netcdf=True
    )
    
    # Get statistics
    stats = processor.get_statistics()


def _load_sample_for_worker(file_path: str) -> np.ndarray:
    """
    Load a single sample file (module-level version for multiprocessing).
    Duplicates the logic from SamplePostProcessor._load_sample.
    """
    if file_path.endswith('.nc'):
        ds = xr.open_dataset(file_path)
        if 'data' in ds:
            data = ds['data'].values
        else:
            # Try to concatenate variables if 'data' not present
            vars_list = ['thetao', 'so', 'uo', 'vo']
            present_vars = [v for v in vars_list if v in ds]
            if present_vars:
                data_list = [ds[v].values for v in present_vars]
                # Handle shapes (check if time dim exists)
                processed_list = []
                for d in data_list:
                    if d.ndim == 3:  # (time, lat, lon) -> (lat, lon)
                        processed_list.append(d[0])
                    else:
                        processed_list.append(d)
                data = np.stack(processed_list)
            else:
                raise ValueError(f"Could not find valid data in .nc file: {file_path}")
        ds.close()
        return data

    data = np.load(file_path)
    
    # Handle .npz files
    if file_path.endswith('.npz'):
        if hasattr(data, 'files'):
            if 'data' in data.files:
                data = data['data']
            else:
                raise ValueError(f"Key 'data' not found in .npz file: {file_path}")
    
    return data


def _process_ensemble_group_for_spatial_rmse(args):
    """
    Worker function to process a single ensemble group for spatial RMSE calculation.
    
    Args:
        args: Tuple of (base_id, file_paths, ground_truth_dir, variables)
        
    Returns:
        dict: Dictionary with sum of squared errors and count for each variable,
              or None if processing fails
    """
    base_id, file_paths, ground_truth_dir, variables = args
    
    try:
        # Sort to ensure deterministic order
        file_paths = sorted(file_paths)
        
        # Load all ensemble members
        member_data_list = []
        for fp in file_paths:
            data = _load_sample_for_worker(fp)
            member_data_list.append(data)
        
        # Compute ensemble mean
        ensemble_stack = np.stack(member_data_list, axis=0)  # (N_members, C, H, W)
        
        # Compute ensemble mean (handle NaN values)
        valid_count = np.sum(~np.isnan(ensemble_stack), axis=0)
        sample_sum = np.nansum(ensemble_stack, axis=0)
        ensemble_mean = np.divide(
            sample_sum,
            valid_count,
            out=np.full_like(sample_sum, np.nan),
            where=valid_count > 0
        )
        
        # Load ground truth
        gt_filename = base_id + '.nc'
        gt_path = os.path.join(ground_truth_dir, gt_filename)
        
        if not os.path.exists(gt_path) and base_id.startswith('sample_'):
            gt_path = os.path.join(ground_truth_dir, base_id.replace('sample_', '') + '.nc')
        
        if not os.path.exists(gt_path):
            return {'error': f'GT file not found: {gt_path}'}
        
        gt_data_list = []
        with xr.open_dataset(gt_path) as ds_gt_raw:
            for var in variables:
                if var not in ds_gt_raw:
                    return {'error': f'Variable {var} not in GT: {gt_path}'}
                var_data = ds_gt_raw[var].values
                if var_data.ndim == 3 and var_data.shape[0] == 1:
                    var_data = var_data[0]
                gt_data_list.append(var_data)
        
        gt_data = np.array(gt_data_list)
        
        # Handle shape mismatch: crop ensemble_mean to match GT shape
        if ensemble_mean.shape != gt_data.shape:
            if (ensemble_mean.shape[0] == gt_data.shape[0] and 
                ensemble_mean.shape[2] == gt_data.shape[2] and 
                ensemble_mean.shape[1] > gt_data.shape[1]):
                # Crop ensemble_mean from the beginning (South Pole side)
                crop_h = ensemble_mean.shape[1] - gt_data.shape[1]
                ensemble_mean = ensemble_mean[:, crop_h:, :]
            else:
                return {'error': f'Shape mismatch: ensemble {ensemble_mean.shape} vs GT {gt_data.shape}'}
        
        # Compute squared error at each position
        land_mask = np.isnan(gt_data)
        diff_sq = (ensemble_mean - gt_data) ** 2
        diff_sq[land_mask] = 0  # Set land to 0 for accumulation
        
        # Prepare result
        result = {}
        for i, var in enumerate(variables):
            # Count valid (non-land) positions
            valid_mask = ~land_mask[i]
            result[var] = {
                'sum_sq_error': diff_sq[i],
                'count': valid_mask.astype(np.int64)
            }
        
        return result
        
    except Exception as e:
        return {'error': f'Exception: {str(e)}'}


if __name__ == '__main__':
    main()
