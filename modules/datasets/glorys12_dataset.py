import os
import glob
import time
import pandas as pd
import torch
import xarray as xr
import numpy as np
import netCDF4
from torch.utils.data import Dataset
from typing import Optional, List, Tuple, Union
from datetime import datetime, timedelta


def glorys_argo_collate_fn(batch):
    """
    Custom collate function to handle variable number of Argo profiles.
    
    Args:
        batch: List of tuples (glorys_data, argo_data, depth_index, mask, time_encoding)
               time_encoding is optional and only present if use_time_embed is True
        
    Returns:
        batched_glorys: (B, C, H, W)
        batched_argo: dict containing padded tensors
        batched_depth: (B,)
        batched_mask: (B, 1, H, W)
        batched_time: (B, 2) or None - Cyclical time encoding [sin, cos]
    """
    glorys_list = []
    argo_list = []
    depth_list = []
    mask_list = []
    time_list = []
    
    for item in batch:
        glorys_list.append(item[0])
        argo_list.append(item[1])
        depth_list.append(item[2])
        if len(item) > 3:
            mask_list.append(item[3])
        # Check if time_encoding is present (5th element)
        if len(item) > 4:
            time_list.append(item[4])
        
    batched_glorys = torch.stack(glorys_list)
    batched_depth = torch.tensor(depth_list)
    batched_mask = torch.stack(mask_list) if mask_list else None
    
    # Stack time encodings if present
    batched_time = torch.stack(time_list) if time_list else None
    
    # Handle Argo data
    # Check if we have any Argo data
    has_argo = any(len(a) > 0 for a in argo_list if a is not None)
    
    if not has_argo:
        return batched_glorys, None, batched_depth, batched_mask, batched_time
        
    # Find max number of profiles
    max_profiles = 0
    for argo in argo_list:
        if argo and 'temperature' in argo:
            max_profiles = max(max_profiles, argo['temperature'].shape[0])
            
    if max_profiles == 0:
        return batched_glorys, None, batched_depth, batched_mask, batched_time
        
    # Prepare batched Argo tensors
    # We need to know the depth dimension for profiles
    # Find first non-empty argo data to get depth dim
    n_depth_levels = 0
    for argo in argo_list:
        if argo and 'temperature' in argo and argo['temperature'].shape[0] > 0:
            n_depth_levels = argo['temperature'].shape[1]
            break
            
    batched_argo = {
        'temperature': torch.zeros(len(batch), max_profiles, n_depth_levels),
        'salinity': torch.zeros(len(batch), max_profiles, n_depth_levels),
        'temperature_mask': torch.zeros(len(batch), max_profiles, n_depth_levels),
        'salinity_mask': torch.zeros(len(batch), max_profiles, n_depth_levels),
        'latitude': torch.zeros(len(batch), max_profiles),
        'longitude': torch.zeros(len(batch), max_profiles),
        'relative_time': torch.zeros(len(batch), max_profiles),
        'mask': torch.zeros(len(batch), max_profiles, dtype=torch.bool) # Mask for valid profiles
    }
    
    for i, argo in enumerate(argo_list):
        if argo and 'temperature' in argo and argo['temperature'].shape[0] > 0:
            n_profiles = argo['temperature'].shape[0]
            batched_argo['temperature'][i, :n_profiles] = argo['temperature']
            batched_argo['salinity'][i, :n_profiles] = argo['salinity']
            batched_argo['temperature_mask'][i, :n_profiles] = argo['temperature_mask']
            batched_argo['salinity_mask'][i, :n_profiles] = argo['salinity_mask']
            batched_argo['latitude'][i, :n_profiles] = argo['latitude']
            batched_argo['longitude'][i, :n_profiles] = argo['longitude']
            batched_argo['relative_time'][i, :n_profiles] = argo['relative_time']
            batched_argo['mask'][i, :n_profiles] = True
            
    return batched_glorys, batched_argo, batched_depth, batched_mask, batched_time


class GLORYS12Dataset(Dataset):
    """
    PyTorch Dataset class for GLORYS12 ocean data.
    Each .nc file represents a single sample.
    
    Args:
        data_dir (str): Root directory containing the GLORYS12 data
        dataset_type (str): Type of dataset - 'train', 'cal', or 'test'
        metadata_file (str, optional): Path to metadata CSV file. If None, will look for 
                                       'metadata_{dataset_type}.csv' in data_dir
        variables (List[str]): List of variables to load from nc files. 
                               Default: ['thetao', 'so', 'uo', 'vo']
        transform (callable, optional): Optional transform to be applied on a sample
        normalize (bool): Whether to normalize the data
        statistics_file (str, optional): Path to statistics file for normalization
        
    Returns:
        torch.Tensor: Shape (C, H, W) where C is number of variables
    """
    
    def __init__(
        self,
        data_dir: str,
        dataset_type: str = 'train',
        variables: List[str] = ['thetao', 'so', 'uo', 'vo'],
        transform: Optional[callable] = None,
        normalize: bool = False,
        statistics_file: Optional[str] = None,
        argo_file: Optional[str] = None,
        argo_days: int = 1,
        max_argo_profiles_per_day: Optional[int] = None,
        normalization_method: str = 'minmax',
        velocity_weight: Optional[float] = 1.0,
        use_time_embed: bool = False
    ):
        super(GLORYS12Dataset, self).__init__()
        
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.variables = variables
        self.transform = transform
        self.normalize = normalize
        self.argo_file = argo_file
        
        # Validate argo_days parameter: must be a positive odd number
        if argo_days <= 0:
            raise ValueError(f"argo_days must be a positive number, got {argo_days}")
        if argo_days % 2 == 0:
            raise ValueError(f"argo_days must be an odd number (1, 3, 5, 7, ...), got {argo_days}")
        self.argo_days = argo_days
        self.max_argo_profiles_per_day = max_argo_profiles_per_day
        
        self.normalization_method = normalization_method
        self.velocity_weight = velocity_weight
        self.use_time_embed = use_time_embed
        self.argo_is_daily_dir = bool(self.argo_file and os.path.isdir(self.argo_file))
        self._argo_nc = None
        
        # Build file paths for each sample
        self.file_paths = self._build_file_paths()
        
        # Load statistics for normalization if needed
        self.statistics = None
        if self.normalize:
            self._load_statistics(statistics_file)
            
        # Initialize Argo data mapping if file provided
        self.argo_indices = None
        if self.argo_file:
            self._init_argo()

    def _init_argo(self):
        """
        Initialize Argo data mapping from date to indices and load all data into memory.
        """
        if not os.path.exists(self.argo_file):
            raise FileNotFoundError(f"Argo file not found: {self.argo_file}")
        
        if self.argo_is_daily_dir:
            print(f"Loading daily Argo profiles from {self.argo_file}...")
            daily_files = sorted(glob.glob(os.path.join(self.argo_file, '*.nc')))
            if not daily_files:
                raise FileNotFoundError(f"No daily Argo NetCDF files found in: {self.argo_file}")
            
            self.cached_argo_data = {}
            self.argo_indices = {}
            first_valid_file = None
            
            for file_path in daily_files:
                name = os.path.splitext(os.path.basename(file_path))[0]
                date_key = None
                for part in reversed(name.split('_')):
                    try:
                        date_obj = datetime.strptime(part, '%Y-%m-%d')
                        date_key = date_obj.strftime('%Y%m%d')
                        break
                    except ValueError:
                        continue
                if not date_key:
                    continue
                self.argo_indices[date_key] = file_path
                if first_valid_file is None:
                    first_valid_file = file_path
            
            if not self.argo_indices:
                raise ValueError(f"No valid daily Argo files with date in filename found in: {self.argo_file}")
            
            if first_valid_file is None:
                first_valid_file = next(iter(self.argo_indices.values()))
            
            ds = xr.open_dataset(first_valid_file)
            if 'depth' not in ds.coords:
                ds.close()
                raise ValueError("Daily Argo NetCDF must have 'depth' coordinate")
            self.cached_argo_data['depth'] = torch.from_numpy(ds['depth'].values).float().share_memory_()
            ds.close()
            
            print(f"Indexed {len(self.argo_indices)} daily Argo files")
            return
        
        print(f"Loading Argo profiles from {self.argo_file} into memory...")
        try:
            ds = xr.open_dataset(self.argo_file)
            if 'time' not in ds.coords:
                raise ValueError("Argo NetCDF must have 'time' coordinate")
            
            self.cached_argo_data = {}
            
            self.cached_argo_data['latitude'] = torch.from_numpy(ds['latitude'].values).float().share_memory_()
            self.cached_argo_data['longitude'] = torch.from_numpy(ds['longitude'].values).float().share_memory_()
            self.cached_argo_data['depth'] = torch.from_numpy(ds['depth'].values).float().share_memory_()
            self.cached_argo_data['time'] = ds['time'].values
            
            times = self.cached_argo_data['time']
            ds.close()
            
            from collections import defaultdict
            self.argo_indices = defaultdict(list)
            
            for idx, t in enumerate(times):
                ts = pd.Timestamp(t)
                if pd.isna(ts):
                    continue
                date_str = ts.strftime('%Y%m%d')
                self.argo_indices[date_str].append(idx)
                
            print(f"Loaded and indexed {len(times)} Argo profiles across {len(self.argo_indices)} days")
            
        except Exception as e:
            print(f"Error initializing Argo data: {e}")
            raise

    def _load_argo_sample(self, indices: Union[List[int], List[str], str, None]) -> dict:
        """
        Load Argo profiles for specific indices from cached memory.
        
        Args:
            indices (List[int] or List[str]): List of profile indices or file paths to load
            
        Returns:
            dict: Dictionary containing Argo data tensors
        """
        if not self.argo_file or not indices:
            return None
        
        if self.argo_is_daily_dir:
            # Handle list of file paths
            if isinstance(indices, str):
                indices = [indices]
            
            all_data = []
            
            for file_path in indices:
                try:
                    ds = xr.open_dataset(file_path)
                    t_vals = ds['temperature'].values
                    s_vals = ds['salinity'].values
                    lat_vals = ds['latitude'].values
                    lon_vals = ds['longitude'].values
                    depth_vals = ds['depth'].values
                    time_vals = ds['time'].values
                    ds.close()
                    
                    temp = torch.from_numpy(t_vals).float()
                    salt = torch.from_numpy(s_vals).float()
                    
                    temp_mask = (~torch.isnan(temp)).float()
                    salt_mask = (~torch.isnan(salt)).float()
                    
                    data = {
                        'temperature': temp,
                        'salinity': salt,
                        'temperature_mask': temp_mask,
                        'salinity_mask': salt_mask,
                        'latitude': torch.from_numpy(lat_vals).float(),
                        'longitude': torch.from_numpy(lon_vals).float(),
                        'depth': torch.from_numpy(depth_vals).float(),
                        'time': time_vals
                    }
                    all_data.append(data)
                except Exception as e:
                    print(f"Warning: Error loading Argo file {file_path}: {e}")
                    continue
            
            if not all_data:
                return None
            
            # Concatenate all data
            final_data = {}
            # Use the first valid data to get depth
            final_data['depth'] = all_data[0]['depth']
            
            # Concatenate other fields
            for key in ['temperature', 'salinity', 'temperature_mask', 'salinity_mask', 'latitude', 'longitude']:
                final_data[key] = torch.cat([d[key] for d in all_data], dim=0)
            
            # Concatenate time (numpy array)
            final_data['time'] = np.concatenate([d['time'] for d in all_data], axis=0)

            return self._normalize_argo_data(final_data)
            
        # If no indices, return empty tensors
            
        try:
            if self._argo_nc is None:
                self._argo_nc = netCDF4.Dataset(self.argo_file, 'r')

            nc = self._argo_nc
            t_vals = nc.variables['temperature'][indices]
            s_vals = nc.variables['salinity'][indices]
            
            if hasattr(t_vals, 'filled'):
                t_vals = t_vals.filled(np.nan)
            if hasattr(s_vals, 'filled'):
                s_vals = s_vals.filled(np.nan)
                
            temp = torch.from_numpy(t_vals).float()
            salt = torch.from_numpy(s_vals).float()

            # Compute masks
            temp_mask = (~torch.isnan(temp)).float()
            salt_mask = (~torch.isnan(salt)).float()

            data = {
                'temperature': temp,
                'salinity': salt,
                'temperature_mask': temp_mask,
                'salinity_mask': salt_mask,
                'latitude': self.cached_argo_data['latitude'][indices].clone(),
                'longitude': self.cached_argo_data['longitude'][indices].clone(),
                'depth': self.cached_argo_data['depth'].clone(),
                'time': self.cached_argo_data['time'][indices]
            }

            return self._normalize_argo_data(data)
            
        except Exception as e:
            print(f"Error loading Argo sample: {e}")
            raise

    def _get_argo_data(self, timestamp: datetime) -> Optional[dict]:
        """
        Get and process Argo data for a specific timestamp and time window.
        
        Args:
            timestamp (datetime): Reference timestamp
            
        Returns:
            dict: Processed Argo data or None
        """
        if not self.argo_file or timestamp is None:
            return None
            
        argo_indices = []
        argo_paths = []
        
        # Determine target dates based on argo_days
        # Support any odd number of days centered on current date
        target_dates = []
        half_window = self.argo_days // 2
        for i in range(-half_window, half_window + 1):
            target_dates.append(timestamp + timedelta(days=i))
        
        for date_obj in target_dates:
            date_str = date_obj.strftime('%Y%m%d')
            if self.argo_is_daily_dir:
                path = self.argo_indices.get(date_str)
                if path:
                    argo_paths.append(path)
            else:
                indices = self.argo_indices.get(date_str, [])
                argo_indices.extend(indices)
        
        # Load data
        if self.argo_is_daily_dir:
            argo_data = self._load_argo_sample(argo_paths)
        else:
            argo_data = self._load_argo_sample(argo_indices)
        
        if argo_data:
            # Subsample Argo profiles if exceeding total max profiles (per_day * argo_days)
            n_profiles = argo_data['temperature'].shape[0]
            if self.max_argo_profiles_per_day is not None:
                total_max = self.max_argo_profiles_per_day * self.argo_days
                if n_profiles > total_max:
                    # Randomly sample indices
                    indices = torch.randperm(n_profiles)[:total_max]
                    np_indices = indices.numpy()
                    
                    for key in ['temperature', 'salinity', 'temperature_mask', 'salinity_mask', 'latitude', 'longitude', 'time']:
                        if key in argo_data:
                            if isinstance(argo_data[key], torch.Tensor):
                                argo_data[key] = argo_data[key][indices]
                            else:
                                argo_data[key] = argo_data[key][np_indices]
            
            times = pd.to_datetime(argo_data['time'])
            argo_data['relative_time'] = self._compute_relative_time(times, timestamp)
            
            # Remove raw time array as it's not needed for training
            del argo_data['time']
            
            return argo_data
            
        return None

    def _normalize_argo_data(self, data: dict) -> dict:
        """
        Normalize Argo temperature and salinity in-place based on configured statistics.
        """
        if not self.normalize or not self.statistics:
            return data

        if 'thetao' in self.statistics:
            mask = ~torch.isnan(data['temperature'])
            if self.normalization_method in ('minmax', 'quantile'):
                if self.normalization_method == 'minmax':
                    t_min = self.statistics['thetao']['min']
                    t_max = self.statistics['thetao']['max']
                else:
                    t_min = self.statistics['thetao']['q0.1']
                    t_max = self.statistics['thetao']['q99.9']
                data['temperature'][mask] = (
                    2.0 * (data['temperature'][mask] - t_min) / (t_max - t_min) - 1.0
                )
            elif self.normalization_method == 'zscore':
                t_mean = self.statistics['thetao']['mean']
                t_std = self.statistics['thetao']['std']
                if t_std == 0:
                    data['temperature'][mask] = 0.0
                else:
                    data['temperature'][mask] = (
                        (data['temperature'][mask] - t_mean) / t_std
                    )
            else:
                raise ValueError(f"Unknown normalization method: {self.normalization_method}")
            data['temperature'][~mask] = 0.0

        if 'so' in self.statistics:
            mask = ~torch.isnan(data['salinity'])
            if self.normalization_method in ('minmax', 'quantile'):
                if self.normalization_method == 'minmax':
                    s_min = self.statistics['so']['min']
                    s_max = self.statistics['so']['max']
                else:
                    s_min = self.statistics['so']['q0.1']
                    s_max = self.statistics['so']['q99.9']
                data['salinity'][mask] = (
                    2.0 * (data['salinity'][mask] - s_min) / (s_max - s_min) - 1.0
                )
            elif self.normalization_method == 'zscore':
                s_mean = self.statistics['so']['mean']
                s_std = self.statistics['so']['std']
                if s_std == 0:
                    data['salinity'][mask] = 0.0
                else:
                    data['salinity'][mask] = (
                        (data['salinity'][mask] - s_mean) / s_std
                    )
            else:
                raise ValueError(f"Unknown normalization method: {self.normalization_method}")
            data['salinity'][~mask] = 0.0

        return data
    
    def _load_statistics(self, statistics_file: Optional[str] = None) -> None:
        """
        Load statistics from JSON file for normalization.
        
        Args:
            statistics_file (str, optional): Path to statistics JSON file.
                                            If None, will look for 'statistics_train.json' in data_dir
        """
        import json
        
        if statistics_file is None:
            # Default to train statistics for normalization
            statistics_file = os.path.join(self.data_dir, 'statistics_train.json')
        
        if not os.path.exists(statistics_file):
            raise FileNotFoundError(
                f"Statistics file not found: {statistics_file}. "
                f"Please run compute_statistics() on the training dataset first."
            )
        
        with open(statistics_file, 'r') as f:
            self.statistics = json.load(f)
        
        # Verify that all required variables have statistics
        for var in self.variables:
            if var not in self.statistics:
                raise ValueError(f"Statistics for variable '{var}' not found in {statistics_file}")
        
        print(f"Loaded statistics from: {statistics_file}")
    
    def _build_file_paths(self) -> List[str]:
        """
        Build the full file path for each sample based on files in the directory.
        
        Returns:
            List[str]: List of file paths corresponding to each sample
        """
        # Determine subdirectory: train uses 'train', cal uses 'cal', test uses 'test'
        if self.dataset_type == 'test':
            subdir = 'test'
        elif self.dataset_type == 'cal':
            subdir = 'cal'
        else:
            subdir = 'train'
        search_dir = os.path.join(self.data_dir, subdir)
        
        if not os.path.exists(search_dir):
            raise FileNotFoundError(f"Data directory not found: {search_dir}")
            
        file_paths = glob.glob(os.path.join(search_dir, "*.nc"))
        file_paths.sort()  # Ensure deterministic order
        
        # Filter by year for train (2012-2021) and cal (2022)
        if self.dataset_type in ['train', 'cal']:
            filtered_paths = []
            for fp in file_paths:
                filename = os.path.basename(fp)
                try:
                    # Parse date from filename, assuming format YYYYMMDD_*.nc
                    date_str = filename.split('_')[0]
                    year = int(date_str[:4])
                    
                    if self.dataset_type == 'train':
                        if 2012 <= year <= 2021:
                            filtered_paths.append(fp)
                    elif self.dataset_type == 'cal':
                        if year == 2022:
                            filtered_paths.append(fp)
                except (ValueError, IndexError):
                    # If filename format doesn't match, we skip it
                    continue
            
            print(f"Filtered {self.dataset_type} set: {len(filtered_paths)} samples (original in {subdir}: {len(file_paths)})")
            file_paths = filtered_paths
        
        if not file_paths:
            print(f"Warning: No .nc files found in {search_dir} for type {self.dataset_type}")
            
        return file_paths
    

    def _normalize_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize data to [-1, 1] range using min-max normalization.
        Only normalizes non-NaN elements; NaN values are set to 0.
        
        Formula: normalized = 2 * (x - min) / (max - min) - 1
        
        Args:
            data (torch.Tensor): Input data with shape (C, H, W)
            
        Returns:
            torch.Tensor: Normalized data in [-1, 1] range with NaN values set to 0
        """
        if self.statistics is None:
            raise RuntimeError("Statistics not loaded. Cannot normalize data.")
        
        normalized_data = data.clone()
        
        for var_idx, var_name in enumerate(self.variables):
            var_data = data[var_idx]
            valid_mask = ~torch.isnan(var_data)

            if self.normalization_method == 'minmax':
                var_min = self.statistics[var_name]['min']
                var_max = self.statistics[var_name]['max']
                normalized_data[var_idx][valid_mask] = (
                    2.0 * (var_data[valid_mask] - var_min) / (var_max - var_min) - 1.0
                )
            elif self.normalization_method == 'quantile':
                var_min = self.statistics[var_name]['q0.1']
                var_max = self.statistics[var_name]['q99.9']
                normalized_data[var_idx][valid_mask] = (
                    2.0 * (var_data[valid_mask] - var_min) / (var_max - var_min) - 1.0
                )
            elif self.normalization_method == 'zscore':
                var_mean = self.statistics[var_name]['mean']
                var_std = self.statistics[var_name]['std']
                if var_std == 0:
                    normalized_data[var_idx][valid_mask] = 0.0
                else:
                    if var_name in ('uo', 'vo'):
                        mean_to_use = 0.0
                        # Apply velocity weight if configured
                        weight = self.velocity_weight if self.velocity_weight is not None else 1.0
                    else:
                        mean_to_use = var_mean
                        weight = 1.0
                        
                    normalized_data[var_idx][valid_mask] = (
                        (var_data[valid_mask] - mean_to_use) / var_std
                    ) * weight
            else:
                raise ValueError(f"Unknown normalization method: {self.normalization_method}")
            
            normalized_data[var_idx][~valid_mask] = 0.0
        
        return normalized_data
    
    def _denormalize_data(self, normalized_data: torch.Tensor) -> torch.Tensor:
        """
        Denormalize data from [-1, 1] range back to original range.
        
        Formula: x = (normalized + 1) / 2 * (max - min) + min
        
        Args:
            normalized_data (torch.Tensor): Normalized data with shape (C, H, W) or (B, C, H, W)
            
        Returns:
            torch.Tensor: Denormalized data in original range with same shape as input
        """
        if self.statistics is None:
            raise RuntimeError("Statistics not loaded. Cannot denormalize data.")
        
        if normalized_data.dim() not in (3, 4):
            raise ValueError(f"Expected normalized_data to have 3 or 4 dimensions, but got shape {normalized_data.shape}")
        
        denormalized_data = normalized_data.clone()
        
        if self.normalization_method in ('minmax', 'quantile'):
            if self.normalization_method == 'minmax':
                min_key = 'min'
                max_key = 'max'
            else:
                min_key = 'q0.1'
                max_key = 'q99.9'

            var_mins = torch.tensor(
                [self.statistics[var_name][min_key] for var_name in self.variables],
                dtype=normalized_data.dtype,
                device=normalized_data.device,
            )
            var_maxs = torch.tensor(
                [self.statistics[var_name][max_key] for var_name in self.variables],
                dtype=normalized_data.dtype,
                device=normalized_data.device,
            )
            
            if normalized_data.dim() == 3:
                mins = var_mins.view(-1, 1, 1)
                maxs = var_maxs.view(-1, 1, 1)
            else:
                mins = var_mins.view(1, -1, 1, 1)
                maxs = var_maxs.view(1, -1, 1, 1)
            
            denormalized_data = (denormalized_data + 1.0) / 2.0 * (maxs - mins) + mins
        elif self.normalization_method == 'zscore':
            var_means = torch.tensor(
                [self.statistics[var_name]['mean'] for var_name in self.variables],
                dtype=normalized_data.dtype,
                device=normalized_data.device,
            )
            
            var_stds = torch.tensor(
                [self.statistics[var_name]['std'] for var_name in self.variables],
                dtype=normalized_data.dtype,
                device=normalized_data.device,
            )
            
            for idx, var_name in enumerate(self.variables):
                if var_name in ('uo', 'vo'):
                    var_means[idx] = 0.0
                    # Adjust std for velocity weight during denormalization
                    # x = z_weighted * (std / weight) + mean
                    if self.velocity_weight is not None:
                         var_stds[idx] = var_stds[idx] / self.velocity_weight

            var_stds = torch.where(var_stds == 0, torch.ones_like(var_stds), var_stds)

            if normalized_data.dim() == 3:
                means = var_means.view(-1, 1, 1)
                stds = var_stds.view(-1, 1, 1)
            else:
                means = var_means.view(1, -1, 1, 1)
                stds = var_stds.view(1, -1, 1, 1)

            denormalized_data = denormalized_data * stds + means
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")
        
        return denormalized_data
    

    def _load_sample(
        self, 
        idx: int, 
        normalize: bool = None,
        return_mask: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Load a single sample from file.
        
        Args:
            idx (int): Sample index
            normalize (bool): Whether to normalize. If None, uses self.normalize
            return_mask (bool): Whether to return the ocean mask
            
        Returns:
            torch.Tensor: Data tensor with shape (C, H, W)
            (Optional) torch.Tensor: Ocean mask with shape (1, H, W) if return_mask is True
        """
            
        file_path = self.file_paths[idx]
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load NetCDF file
        ds = xr.open_dataset(file_path)

        # Extract variables and stack them
        var_arrays = []
        for var in self.variables:
            if var not in ds:
                raise ValueError(f"Variable '{var}' not found in {file_path}")
            
            var_data = ds[var].values  # Shape: (lat, lon) or (time, lat, lon)

            # Handle different shapes - ensure we get (H, W)
            if var_data.ndim == 3:
                var_data = var_data[0]  # Take first time step if present
            elif var_data.ndim != 2:
                raise ValueError(f"Unexpected shape for variable '{var}': {var_data.shape}")
            
            var_arrays.append(var_data)
        
        # Stack variables along channel dimension: (C, H, W)
        data = np.stack(var_arrays, axis=0).astype(np.float32)
        
        # Convert to torch tensor
        data_tensor = torch.from_numpy(data)
        
        ds.close()
        
        # Pad the second dimension from 341 to 352 (add 19 zeros at the beginning)
        # IMPORTANT: Padding must be done BEFORE normalization
        # because padding adds zeros, and we want to normalize after padding
        # so that padded zeros can be properly handled during normalization
        pad_size = 352 - data_tensor.shape[1]
        if pad_size > 0:
            padding = (0, 0, pad_size, 0)  # (left, right, top, bottom) for (H, W)
            # Use NaN for padding instead of 0, so normalization can handle it correctly
            data_tensor = torch.nn.functional.pad(data_tensor, padding, mode='constant', value=float('nan'))
        
        # Compute ocean mask (1 for ocean, 0 for land)
        # glorys12 original data's empty value (NaN) is land
        mask = (~torch.isnan(data_tensor[0:1])).float()

        # Apply normalization if requested
        # This must be done AFTER padding so that padded NaN values are correctly set to 0
        should_normalize = normalize if normalize is not None else self.normalize
        if should_normalize:
            data_tensor = self._normalize_data(data_tensor)
        
        if return_mask:
            return data_tensor, mask
        return data_tensor
    
    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, dict, int, torch.Tensor, torch.Tensor]]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            torch.Tensor: Sample data with shape (C, H, W)
            (Optional) dict: Argo data corresponding to the sample date, if argo_file is provided.
                             Includes relative time for each profile.
            (Optional) int: Depth index of the sample
            (Optional) torch.Tensor: Ocean mask with shape (1, H, W)
            (Optional) torch.Tensor: Cyclical time encoding with shape (2,) if use_time_embed is True
        """
        data, mask = self._load_sample(idx, return_mask=True)
        
        # Apply transform if provided
        if self.transform is not None:
            data = self.transform(data)

        # Get sample info for depth and timestamp
        info = self.get_sample_info(idx)
        depth_index = info['depth_index'] if info['depth_index'] is not None else 0
        depth_index = depth_index / 39.0
        
        # Compute cyclical time encoding if enabled
        time_encoding = None
        if self.use_time_embed and info['timestamp'] is not None:
            time_encoding = self._compute_cyclical_time_encoding(info['timestamp'])

        if self.argo_file:
            argo_data = self._get_argo_data(info['timestamp'])
            if argo_data:
                if time_encoding is not None:
                    return data, argo_data, depth_index, mask, time_encoding
                return data, argo_data, depth_index, mask
            
        if time_encoding is not None:
            return data, {}, depth_index, mask, time_encoding
        return data, {}, depth_index, mask

    def _compute_relative_time(self, times: pd.DatetimeIndex, reference_date: datetime) -> torch.Tensor:
        """
        Compute relative time with respect to noon of the reference date.
        Result is normalized such that:
        - (argo_days * 12) hours = 1.0 unit
        
        For example:
        - If argo_days = 1: 12 hours = 1.0 unit
        - If argo_days = 3: 36 hours = 1.0 unit
        - If argo_days = 5: 60 hours = 1.0 unit
        - If argo_days = 7: 84 hours = 1.0 unit
        
        Args:
            times (pd.DatetimeIndex): Timestamps of Argo profiles
            reference_date (datetime): Reference date (usually the GLORYS sample date)
            
        Returns:
            torch.Tensor: Relative time values
        """
        # Set reference to noon of the reference date
        ref_noon = reference_date.replace(hour=12, minute=0, second=0, microsecond=0)
        ref_noon_ts = pd.Timestamp(ref_noon)
        
        # Calculate difference in seconds
        delta_seconds = (times - ref_noon_ts).total_seconds().values
        
        # Normalize
        # If argo_days=1, normalization factor is 12 hours
        # If argo_days=3, normalization factor is 36 hours (3 * 12)
        norm_hours = 12.0 * self.argo_days
        
        rel_time = delta_seconds / (norm_hours * 3600.0)
        
        return torch.from_numpy(rel_time).float()
    
    def _compute_cyclical_time_encoding(self, timestamp: datetime) -> torch.Tensor:
        """
        Compute cyclical time encoding for a given timestamp using day of year.
        
        Encodes the date as sin and cos components to capture seasonal periodicity:
        - x_sin = sin(2 * pi * day_of_year / days_in_year)
        - x_cos = cos(2 * pi * day_of_year / days_in_year)
        
        Args:
            timestamp (datetime): The date to encode
            
        Returns:
            torch.Tensor: Tensor of shape (2,) containing [sin_val, cos_val]
        """
        import calendar
        
        # Get day of year (1-365 or 1-366 for leap years)
        day_of_year = timestamp.timetuple().tm_yday
        
        # Get total days in year
        days_in_year = 366 if calendar.isleap(timestamp.year) else 365
        
        # Compute cyclical encoding
        angle = 2.0 * np.pi * day_of_year / days_in_year
        sin_val = np.sin(angle)
        cos_val = np.cos(angle)
        
        return torch.tensor([sin_val, cos_val], dtype=torch.float32)
    
    def get_sample_info(self, idx: int) -> dict:
        """
        Get metadata information for a specific sample.
        
        Args:
            idx (int): Sample index
            
        Returns:
            dict: Sample metadata including timestamp, depth_index, and file_path
        """
        file_path = self.file_paths[idx]
        filename = os.path.basename(file_path)
        
        # Expected format: YYYYMMDD_dDD.nc (e.g., 20130101_d00.nc)
        try:
            name_parts = filename.replace('.nc', '').split('_')
            date_str = name_parts[0]
            depth_str = name_parts[1]
            
            timestamp = datetime.strptime(date_str, '%Y%m%d')
            # Extract depth index (remove 'd' prefix)
            depth_index = int(depth_str.replace('d', ''))
            
        except Exception as e:
            print(f"Warning: Failed to parse filename {filename}: {e}")
            timestamp = None
            depth_index = None
            
        return {
            'index': idx,
            'timestamp': timestamp,
            'depth_index': depth_index,
            'file_path': file_path
        }
    
    def compute_statistics(self, 
                           output_file: Optional[str] = None, 
                           num_samples: Optional[int] = None,
                           compute_extreme_quantiles: bool = False) -> dict:
        """
        Compute min, max, mean, standard deviation, and optionally percentiles (0.1%, 99.9%) values for each variable across the dataset.
        
        Args:
            output_file (str, optional): Path to save statistics as JSON file. 
                                        If None, will save to '{data_dir}/statistics_{dataset_type}.json'
            num_samples (int, optional): Number of samples to use for statistics computation.
                                        If None, uses all samples in the dataset.
            compute_extreme_quantiles (bool): Whether to compute 0.1% and 99.9% quantiles.
                                              Default: True
        
        Returns:
            dict: Dictionary containing statistics for each variable
                  Format: {
                      'variable_name': {
                      'min': float,
                      'max': float,
                      'mean': float,
                      'std': float,
                      'q0.1': float,
                      'q99.9': float
                      },
                      ...
                  }
        """
        import json
        from tqdm import tqdm
        
        print(f"Computing statistics for {self.dataset_type} dataset...")
        
        # Determine number of samples to process
        total_samples = len(self)
        samples_to_process = num_samples if num_samples is not None else total_samples
        samples_to_process = min(samples_to_process, total_samples)
        
        # Initialize storage for all values to compute percentiles
        all_values = {var: [] for var in self.variables}
        
        # Track exact min/max across all data
        exact_stats = {var: {'min': float('inf'), 'max': float('-inf')} for var in self.variables}
        
        # Limit memory usage by subsampling if necessary
        # Target roughly 10M pixels total per variable for robust quantile estimation
        # This avoids "quantile() input tensor is too large" error (limit is often around 16M on some backends)
        MAX_TOTAL_PIXELS = 1_000_000_000
        pixels_per_sample = max(100, MAX_TOTAL_PIXELS // samples_to_process)
        
        # Iterate through samples
        for idx in tqdm(range(samples_to_process), desc="Processing samples"):
            try:
                data = self._load_sample(idx)
                
                # Collect data for each variable
                for var_idx, var_name in enumerate(self.variables):
                    var_data = data[var_idx]  # Shape: (H, W)
                    
                    # Filter out NaN values
                    valid_data = var_data[~torch.isnan(var_data)]
                    
                    if len(valid_data) > 0:
                        # Update exact min/max
                        var_min = valid_data.min().item()
                        var_max = valid_data.max().item()
                        exact_stats[var_name]['min'] = min(exact_stats[var_name]['min'], var_min)
                        exact_stats[var_name]['max'] = max(exact_stats[var_name]['max'], var_max)
                        
                        # Subsample for quantile calculation if too large
                        if len(valid_data) > pixels_per_sample:
                            # Random subsampling
                            perm = torch.randperm(len(valid_data))
                            sampled_data = valid_data[perm[:pixels_per_sample]]
                            all_values[var_name].append(sampled_data)
                        else:
                            all_values[var_name].append(valid_data)
                        
            except Exception as e:
                print(f"Warning: Failed to process sample {idx}: {e}")
                continue
        
        # Compute statistics
        stats = {}
        print("\nComputing percentiles and final statistics...")
        
        for var_name in self.variables:
            if all_values[var_name]:
                # Concatenate all data for this variable
                var_data_all = torch.cat(all_values[var_name])
                
                # Compute stats
                # Use exact min/max from tracking, quantiles from subsampled data
                var_min = exact_stats[var_name]['min']
                var_max = exact_stats[var_name]['max']
                var_mean = var_data_all.mean().item()
                var_std = var_data_all.std().item()
                if compute_extreme_quantiles:
                    var_q001 = torch.quantile(var_data_all, 0.001).item()
                    var_q999 = torch.quantile(var_data_all, 0.999).item()
                else:
                    var_q001 = None
                    var_q999 = None
                
                stats[var_name] = {
                    'min': var_min,
                    'max': var_max,
                    'mean': var_mean,
                    'std': var_std,
                    'q0.1': var_q001,
                    'q99.9': var_q999
                }
            else:
                print(f"Warning: No valid data found for variable {var_name}")
                stats[var_name] = {
                    'min': float('inf'),
                    'max': float('-inf'),
                    'mean': float('inf'),
                    'std': float('inf'),
                    'q0.1': float('inf'),
                    'q99.9': float('-inf')
                }
        
        # Prepare output file path
        if output_file is None:
            output_file = os.path.join(self.data_dir, f'statistics_{self.dataset_type}.json')
        
        # Save statistics to JSON file
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=4)
        
        print(f"\nStatistics computed successfully!")
        print(f"Results saved to: {output_file}")
        print("\nSummary:")
        for var_name, var_stats in stats.items():
            if compute_extreme_quantiles and var_stats['q0.1'] is not None and var_stats['q99.9'] is not None:
                print(
                    f"  {var_name:10s}: "
                    f"min = {var_stats['min']:12.6f}, "
                    f"max = {var_stats['max']:12.6f}, "
                    f"mean = {var_stats['mean']:12.6f}, "
                    f"std = {var_stats['std']:12.6f}, "
                    f"q0.1 = {var_stats['q0.1']:12.6f}, "
                    f"q99.9 = {var_stats['q99.9']:12.6f}"
                )
            else:
                print(
                    f"  {var_name:10s}: "
                    f"min = {var_stats['min']:12.6f}, "
                    f"max = {var_stats['max']:12.6f}, "
                    f"mean = {var_stats['mean']:12.6f}, "
                    f"std = {var_stats['std']:12.6f}"
                )
        
        return stats

    def visualize_distribution(self, 
                              output_file: Optional[str] = None,
                              num_samples: Optional[int] = None,
                              bins: int = 50,
                              figsize: tuple = (16, 10),
                              dpi: int = 300,
                              log_scale: bool = True) -> None:
        """
        Visualize the distribution of values for each variable in the dataset.
        Creates histograms showing the value distribution with statistical information.
        
        Args:
            output_file (str, optional): Path to save the plot. 
                                        If None, will save to '{data_dir}/distribution_{dataset_type}.png'
            num_samples (int, optional): Number of samples to use for visualization.
                                        If None, uses all samples in the dataset.
            bins (int): Number of bins for histograms. Default: 50
            figsize (tuple): Figure size (width, height) in inches. Default: (16, 10)
            dpi (int): Resolution of the output image. Default: 300
            log_scale (bool): Whether to use log scale for y-axis. Default: True
        
        Returns:
            None: Saves the plot to the specified output path
        """
        from tqdm import tqdm
        from modules.plot import plot_variable_distribution
        
        print(f"Collecting data for distribution visualization from {self.dataset_type} dataset...")
        
        # Determine number of samples to process
        total_samples = len(self)
        samples_to_process = num_samples if num_samples is not None else total_samples
        samples_to_process = min(samples_to_process, total_samples)
        
        # Initialize data collectors for each variable
        var_data = {var: [] for var in self.variables}
        
        # Collect data from samples
        for idx in tqdm(range(samples_to_process), desc="Loading samples"):
            try:
                data = self._load_sample(idx)
                
                # Collect data for each variable
                for var_idx, var_name in enumerate(self.variables):
                    var_array = data[var_idx]  # Shape: (H, W)
                    var_data[var_name].append(var_array)
                    
            except Exception as e:
                print(f"Warning: Failed to process sample {idx}: {e}")
                continue
        
        # Concatenate all data for each variable
        print("Combining data...")
        data_dict = {}
        for var_name, data_list in var_data.items():
            if len(data_list) > 0:
                # Stack all samples and flatten
                data_dict[var_name] = torch.cat([d.flatten() for d in data_list])
            else:
                print(f"Warning: No valid data collected for variable '{var_name}'")
        
        # Prepare output file path
        if output_file is None:
            output_file = os.path.join(self.data_dir, f'distribution_{self.dataset_type}.png')
        
        # Call the plotting function
        print(f"Generating distribution plot...")
        plot_variable_distribution(
            data_dict=data_dict,
            output_path=output_file,
            figsize=figsize,
            dpi=dpi,
            bins=bins,
            log_scale=log_scale
        )
        
        print(f"Distribution visualization completed!")
        print(f"Plot saved to: {output_file}")


    def visualize_random_samples(self, 
                                num_samples: int = 3, 
                                output_dir: str = './output/visualization',
                                verbose: bool = True,
                                argo_days: Optional[int] = None) -> None:
        """
        Randomly select and visualize samples from the dataset.
        Visualizes both GLORYS12 data and corresponding Argo data (if available).
        
        Args:
            num_samples (int): Number of random samples to visualize
            output_dir (str): Directory to save visualization plots
            verbose (bool): Whether to print progress information
            argo_days (int, optional): Number of days for Argo time window (must be odd).
                If None, uses self.argo_days. The time window is centered on the GLORYS date.
        """
        import random
        from datetime import timedelta
        from modules.plot.plot import plot_ocean_variables, plot_argo_3d
        
        # Set global font to Times New Roman
        import matplotlib.pyplot as plt
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Determine argo_days to use
        if argo_days is None:
            argo_days = self.argo_days
        else:
            # Validate argo_days parameter
            if argo_days <= 0:
                raise ValueError(f"argo_days must be a positive number, got {argo_days}")
            if argo_days % 2 == 0:
                raise ValueError(f"argo_days must be an odd number (1, 3, 5, 7, ...), got {argo_days}")
        
        if verbose:
            print(f"Visualizing {num_samples} random samples from {self.dataset_type} dataset...")
            print(f"Argo time window: {argo_days} day(s) centered on GLORYS date")
            
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Select random indices
        total_samples = len(self)
        indices = random.sample(range(total_samples), min(num_samples, total_samples))
        
        for idx in indices:
            # Load sample info
            info = self.get_sample_info(idx)
            timestamp = info['timestamp']
            date_str = timestamp.strftime('%Y-%m-%d') if timestamp else f"idx_{idx}"
            file_path = info['file_path']
            
            if verbose:
                print(f"Processing sample {idx} (Date: {date_str})...")
            
            # Load data using internal methods to get xarray dataset for GLORYS plotting
            # We use xarray directly for GLORYS visualization as plot_ocean_variables expects xr.Dataset
            try:
                # 1. Visualize GLORYS data
                ds = xr.open_dataset(file_path)
                
                # Calculate vmin/vmax based on GLORYS reanalysis data using percentiles
                vmin_vmax = {}
                variables = ['thetao', 'so', 'uo', 'vo']
                ds_plot = ds.isel(time=0)
                for var in variables:
                    if var in ds_plot:
                        var_data = ds_plot[var].values
                        if np.all(np.isnan(var_data)):
                            vmin, vmax = 0, 1
                        else:
                            if var == 'so':
                                # Salinity: use 0.5% and 99.5% percentiles
                                vmin = np.nanpercentile(var_data, 0.5)
                                vmax = np.nanpercentile(var_data, 99.5)
                            elif var in ['uo', 'vo']:
                                # Velocity: symmetric around zero using 99.5% of absolute values
                                vmax = np.nanpercentile(np.abs(var_data), 99.5)
                                vmin = -vmax
                                if vmax == 0:
                                    vmin, vmax = -1, 1
                            else:
                                # Temperature: use min/max
                                vmin = np.nanmin(var_data)
                                vmax = np.nanmax(var_data)
                                if vmin == vmax:
                                    vmin, vmax = vmin - 0.5, vmax + 0.5
                        vmin_vmax[var] = (vmin, vmax)
                
                glorys_output_path = os.path.join(output_dir, f"glorys_{date_str}.png")
                plot_ocean_variables(ds, output_path=glorys_output_path, vmin_vmax=vmin_vmax, dpi=900)
                ds.close()
                
                # 2. Visualize Argo data if available
                if self.argo_file and timestamp:
                    # Collect Argo data from time window centered on the GLORYS date
                    argo_indices_list = []
                    argo_paths = []
                    half_window = argo_days // 2
                    
                    # Generate target dates for the time window
                    target_dates = []
                    for i in range(-half_window, half_window + 1):
                        target_dates.append(timestamp + timedelta(days=i))
                    
                    # Collect Argo data from all target dates
                    for date_obj in target_dates:
                        date_key = date_obj.strftime('%Y%m%d')
                        if self.argo_is_daily_dir:
                            path = self.argo_indices.get(date_key)
                            if path:
                                argo_paths.append(path)
                        else:
                            day_indices = self.argo_indices.get(date_key, [])
                            argo_indices_list.extend(day_indices)
                    
                    # Load Argo data
                    if self.argo_is_daily_dir:
                        if argo_paths:
                            argo_data = self._load_argo_sample(argo_paths)
                        else:
                            argo_data = None
                    else:
                        if argo_indices_list:
                            argo_data = self._load_argo_sample(argo_indices_list)
                        else:
                            argo_data = None
                    
                    # Visualize Argo data
                    if argo_data and argo_data['temperature'].shape[0] > 0:
                        n_profiles = argo_data['temperature'].shape[0]
                        start_date = target_dates[0].strftime('%Y-%m-%d')
                        end_date = target_dates[-1].strftime('%Y-%m-%d')
                        if verbose:
                            print(f"  Found {n_profiles} Argo profiles for time window [{start_date} to {end_date}]")
                        argo_output_path = os.path.join(output_dir, f"argo_3d_{date_str}.png")
                        
                        # Calculate vmin/vmax based on Argo data itself (not GLORYS)
                        argo_vmin_vmax = {}
                        temp_data = argo_data['temperature']
                        sal_data = argo_data['salinity']
                        
                        # Convert to numpy if tensor
                        if torch.is_tensor(temp_data):
                            temp_data = temp_data.cpu().numpy()
                        if torch.is_tensor(sal_data):
                            sal_data = sal_data.cpu().numpy()
                        
                        # Temperature: use min/max
                        if np.all(np.isnan(temp_data)):
                            argo_vmin_vmax['thetao'] = (0, 1)
                        else:
                            t_min = np.nanmin(temp_data)
                            t_max = np.nanmax(temp_data)
                            if t_min == t_max:
                                t_min, t_max = t_min - 0.5, t_max + 0.5
                            argo_vmin_vmax['thetao'] = (t_min, t_max)
                        
                        # Salinity: use 0.5% and 99.5% percentiles
                        if np.all(np.isnan(sal_data)):
                            argo_vmin_vmax['so'] = (0, 1)
                        else:
                            s_min = np.nanpercentile(sal_data, 0.5)
                            s_max = np.nanpercentile(sal_data, 99.5)
                            argo_vmin_vmax['so'] = (s_min, s_max)
                        
                        plot_argo_3d(argo_data, output_path=argo_output_path, vmin_vmax=argo_vmin_vmax)
                    elif verbose:
                        print(f"  No Argo profiles found for time window")
            
            except Exception as e:
                print(f"Error visualizing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                
        if verbose:
            print(f"Visualization completed. Results saved to {output_dir}")

    def visualize_argo_spatial_distribution(self,
                                            num_samples: int = 3,
                                            output_dir: str = './output/visualization',
                                            verbose: bool = True,
                                            figsize: tuple = (10, 6),
                                            dpi: int = 300,
                                            marker_size: int = 15,
                                            marker_color: str = 'red',
                                            salinity_vmin: float = None,
                                            salinity_vmax: float = None,
                                            depth_index: int = None,
                                            show_background: bool = True) -> None:
        """
        Randomly select and visualize Argo profiles' spatial distribution overlaid on GLORYS12 salinity field.
        
        Creates scatter plots showing Argo profile locations on top of GLORYS12 sea water salinity (so) field
        as background, using Cartopy for geographic projection. Argo profiles are collected from a time window
        centered on the GLORYS sample date, with window size determined by argo_days parameter.
        
        Args:
            num_samples (int): Number of random samples to visualize
            output_dir (str): Directory to save visualization plots
            verbose (bool): Whether to print progress information
            figsize (tuple): Figure size (width, height) in inches
            dpi (int): Resolution of the output image
            marker_size (int): Size of Argo profile scatter markers
            marker_color (str): Color of Argo profile scatter markers
            salinity_vmin (float, optional): Minimum value for salinity colorbar
            salinity_vmax (float, optional): Maximum value for salinity colorbar
            depth_index (int, optional): Depth layer index to use for salinity background. 
                                        If None, uses the same depth as the sample.
            show_background (bool): Whether to show salinity background field and legend.
                                   If False, only Argo profile points are shown without background.
        """
        import random
        import matplotlib
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        
        # Set Times New Roman font for all text elements
        matplotlib.rcParams['font.family'] = 'serif'
        matplotlib.rcParams['font.serif'] = ['Times New Roman']
        matplotlib.rcParams['axes.labelsize'] = 10
        matplotlib.rcParams['axes.titlesize'] = 11
        matplotlib.rcParams['xtick.labelsize'] = 10
        matplotlib.rcParams['ytick.labelsize'] = 10
        
        if verbose:
            print(f"Visualizing Argo spatial distribution for {num_samples} random samples...")
            print(f"  Time window: {self.argo_days} days (centered on GLORYS date)")
            
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if Argo data is available
        if not self.argo_file:
            print("Warning: No Argo file provided. Cannot visualize Argo profiles.")
            return
        
        # Select random indices
        total_samples = len(self)
        indices = random.sample(range(total_samples), min(num_samples, total_samples))
        
        for idx in indices:
            # Load sample info
            info = self.get_sample_info(idx)
            timestamp = info['timestamp']
            date_str = timestamp.strftime('%Y-%m-%d') if timestamp else f"idx_{idx}"
            file_path = info['file_path']
            
            if verbose:
                print(f"Processing sample {idx} (Date: {date_str})...")
            
            try:
                # Determine which depth layer to use for salinity background
                sample_depth_index = info['depth_index']
                target_depth_index = depth_index if depth_index is not None else sample_depth_index
                
                # Load GLORYS12 data for salinity background
                if target_depth_index == sample_depth_index:
                    # Use the same file as the sample
                    ds = xr.open_dataset(file_path)
                else:
                    # Find the file for the target depth layer
                    # Construct filename: YYYYMMDD_d{target_depth_index:02d}.nc
                    date_str = timestamp.strftime('%Y%m%d')
                    target_filename = f"{date_str}_d{target_depth_index:02d}.nc"
                    target_file_path = os.path.join(os.path.dirname(file_path), target_filename)
                    
                    if not os.path.exists(target_file_path):
                        print(f"  Warning: Target depth file not found: {target_filename}, using sample depth instead")
                        ds = xr.open_dataset(file_path)
                        target_depth_index = sample_depth_index
                    else:
                        ds = xr.open_dataset(target_file_path)
                
                # Extract salinity data (first time step if multiple times exist)
                if 'time' in ds.dims:
                    salinity = ds['so'].isel(time=0)
                else:
                    salinity = ds['so']
                
                # Get coordinate arrays
                lons = ds.longitude.values
                lats = ds.latitude.values
                
                # Create figure with Cartopy projection
                fig, ax = plt.subplots(figsize=figsize, 
                                       subplot_kw={'projection': ccrs.PlateCarree()})
                
                # Determine colorbar limits for salinity
                if show_background and (salinity_vmin is None or salinity_vmax is None):
                    salinity_vals = salinity.values
                    valid_mask = ~np.isnan(salinity_vals)
                    if valid_mask.any():
                        if salinity_vmin is None:
                            salinity_vmin = np.nanpercentile(salinity_vals, 1)
                        if salinity_vmax is None:
                            salinity_vmax = np.nanpercentile(salinity_vals, 99)
                
                # Plot salinity field as background (only if show_background is True)
                if show_background:
                    im = ax.pcolormesh(lons, lats, salinity.values,
                                       cmap='viridis',
                                       shading='auto',
                                       vmin=salinity_vmin,
                                       vmax=salinity_vmax,
                                       transform=ccrs.PlateCarree())
                    
                    # Add colorbar for salinity (smaller size)
                    cbar = plt.colorbar(im, ax=ax, shrink=0.5, extend='both', pad=0.02)
                    cbar.set_label('Salinity (psu)', fontsize=10)
                
                # Load Argo profiles from time window centered on current date
                # Time window size is determined by self.argo_days
                half_window = self.argo_days // 2
                all_argo_lats = []
                all_argo_lons = []
                all_argo_dates = []
                
                for day_offset in range(-half_window, half_window + 1):
                    target_date = timestamp + timedelta(days=day_offset)
                    target_date_key = target_date.strftime('%Y%m%d')
                    target_date_str = target_date.strftime('%Y-%m-%d')
                    
                    if self.argo_is_daily_dir:
                        # Handle daily directory structure
                        argo_file_path = self.argo_indices.get(target_date_key)
                        if argo_file_path:
                            argo_data = self._load_argo_sample(argo_file_path)
                            if argo_data and argo_data['temperature'].shape[0] > 0:
                                all_argo_lats.extend(argo_data['latitude'].numpy().tolist())
                                all_argo_lons.extend(argo_data['longitude'].numpy().tolist())
                                all_argo_dates.extend([target_date_str] * argo_data['temperature'].shape[0])
                    else:
                        # Handle single file structure
                        argo_indices = self.argo_indices.get(target_date_key, [])
                        if argo_indices:
                            argo_data = self._load_argo_sample(argo_indices)
                            if argo_data and argo_data['temperature'].shape[0] > 0:
                                all_argo_lats.extend(argo_data['latitude'].numpy().tolist())
                                all_argo_lons.extend(argo_data['longitude'].numpy().tolist())
                                all_argo_dates.extend([target_date_str] * argo_data['temperature'].shape[0])
                
                # Convert to numpy arrays
                argo_lats = np.array(all_argo_lats)
                argo_lons = np.array(all_argo_lons)
                
                if verbose:
                    print(f"  Found {len(argo_lats)} Argo profiles in {self.argo_days}-day window")
                
                # Plot Argo profile locations as scatter points
                if len(argo_lats) > 0:
                    scatter = ax.scatter(argo_lons, argo_lats,
                                        c=marker_color,
                                        s=marker_size,
                                        marker='o',
                                        edgecolors='black',
                                        linewidths=0.5,
                                        alpha=0.8,
                                        transform=ccrs.PlateCarree(),
                                        zorder=5,
                                        label=f'Argo Profiles (n={len(argo_lats)})')
                    # JASA style: legend at lower right (only if show_background is True)
                    if show_background:
                        ax.legend(loc='lower right', fontsize=9, frameon=True, fancybox=False, edgecolor='black')
                
                # Add map features
                ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
                ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
                gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
                # Hide top and right labels
                gl.top_labels = False
                gl.right_labels = False
                
                # JASA style: remove top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Set title with time window and depth information (JASA: normal weight)
                half_window = self.argo_days // 2
                window_start = (timestamp - timedelta(days=half_window)).strftime('%Y-%m-%d')
                window_end = (timestamp + timedelta(days=half_window)).strftime('%Y-%m-%d')
                glorys_date_formatted = timestamp.strftime('%Y-%m-%d')
                ax.set_title(f'Argo Profiles Spatial Distribution on Salinity Field\n'
                            f'GLORYS12 Date: {glorys_date_formatted} | '
                            f'Argo Window: {window_start} to {window_end}', 
                            fontsize=11, fontweight='normal')
                
                # Save figure
                output_path = os.path.join(output_dir, f"argo_spatial_{date_str}_d{target_depth_index:02d}.png")
                plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                
                if verbose:
                    print(f"  Saved: {output_path}")
                
                # Close dataset
                ds.close()
                
            except Exception as e:
                print(f"Error visualizing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
        
        if verbose:
            print(f"Argo spatial distribution visualization completed. Results saved to {output_dir}")


def test_glorys12_dataset(data_dir: str = './data/glorys12', dataset_type: str = 'train'):
    """
    Test function for GLORYS12Dataset class.
    
    This function performs comprehensive testing of the dataset including:
    - Dataset initialization
    - Sample loading
    - Data shape verification
    - Coordinate information extraction
    - DataLoader integration
    - Normalize functionality
    
    Args:
        data_dir (str): Root directory containing GLORYS12 data
        dataset_type (str): Dataset type to test ('train', 'cal', or 'test')
    """
    from torch.utils.data import DataLoader
    
    print("=" * 80)
    print(f"Testing GLORYS12Dataset with dataset_type='{dataset_type}'")
    print("=" * 80)
    
    # Test 1: Basic initialization without normalization
    print("\n[Test 1] Basic Dataset Initialization")
    try:
        dataset = GLORYS12Dataset(
            data_dir=data_dir,
            dataset_type=dataset_type,
            variables=['thetao', 'so', 'uo', 'vo']
        )
        print(f"✓ Dataset initialized successfully")
        print(f"  - Total samples: {len(dataset)}")
        print(f"  - Variables: {dataset.variables}")
    except Exception as e:
        print(f"✗ Failed to initialize dataset: {e}")
        return
    
    # Test 2: Load single sample
    print("\n[Test 2] Single Sample Loading")
    try:
        sample = dataset[0]
        print(f"✓ Sample loaded successfully")
        print(f"  - Sample shape: {sample.shape}")
        print(f"  - Sample dtype: {sample.dtype}")
        print(f"  - Sample min/max: {sample.min().item():.4f} / {sample.max().item():.4f}")
        print(f"  - Contains NaN: {torch.isnan(sample).any().item()}")
    except Exception as e:
        print(f"✗ Failed to load sample: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: Get sample metadata
    print("\n[Test 3] Sample Metadata")
    try:
        sample_info = dataset.get_sample_info(0)
        print(f"✓ Metadata retrieved successfully")
        print(f"  - Index: {sample_info['index']}")
        print(f"  - Timestamp: {sample_info['timestamp']}")
        print(f"  - Depth index: {sample_info['depth_index']}")
        print(f"  - File path: {sample_info['file_path']}")
    except Exception as e:
        print(f"✗ Failed to get sample info: {e}")
    
    # Test 5: DataLoader integration
    print("\n[Test 5] DataLoader Integration")
    try:
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch = next(iter(dataloader))
        print(f"✓ DataLoader works successfully")
        print(f"  - Batch shape: {batch.shape}")
    except Exception as e:
        print(f"✗ Failed to use DataLoader: {e}")

    # Test 6: Argo Data Integration
    print("\n[Test 6] Argo Data Integration")
    argo_file = './data/argo/argo_profiles.nc'
    if os.path.exists(argo_file):
        try:
            dataset_argo = GLORYS12Dataset(
                data_dir=data_dir,
                dataset_type=dataset_type,
                variables=['thetao', 'so', 'uo', 'vo'],
                argo_file=argo_file
            )
            print(f"✓ Dataset with Argo initialized successfully")
            
            # Load sample
            sample, argo_data = dataset_argo[0]
            print(f"✓ Sample with Argo loaded successfully")
            print(f"  - GLORYS shape: {sample.shape}")
            print(f"  - Argo keys: {list(argo_data.keys())}")
            print(f"  - Argo Temperature shape: {argo_data['temperature'].shape}")
            print(f"  - Argo Salinity shape: {argo_data['salinity'].shape}")
            print(f"  - Argo Latitude shape: {argo_data['latitude'].shape}")
            
            # Check date match
            info = dataset_argo.get_sample_info(0)
            print(f"  - Sample date: {info['timestamp']}")
            
        except Exception as e:
            print(f"✗ Failed to load Argo data: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠ Argo file not found at {argo_file}, skipping Test 6")
    
    # Test 6: Multiple samples verification
    print("\n[Test 6] Multiple Samples Verification")
    try:
        num_samples_to_test = min(5, len(dataset))
        print(f"Testing {num_samples_to_test} samples...")
        for i in range(num_samples_to_test):
            sample = dataset[i]
            if sample.shape[0] != len(dataset.variables):
                print(f"✗ Sample {i} has incorrect number of channels: {sample.shape[0]}")
                break
        else:
            print(f"✓ All {num_samples_to_test} samples have consistent shape")
    except Exception as e:
        print(f"✗ Multiple samples test failed: {e}")
    
    # Test 7: Custom variables
    print("\n[Test 7] Custom Variables Selection")
    try:
        custom_dataset = GLORYS12Dataset(
            data_dir=data_dir,
            dataset_type=dataset_type,
            variables=['thetao', 'so']  # Only 2 variables
        )
        sample = custom_dataset[0]
        print(f"✓ Custom variables work successfully")
        print(f"  - Requested variables: {custom_dataset.variables}")
        print(f"  - Sample channels: {sample.shape[0]}")
        assert sample.shape[0] == 2, "Channel count mismatch"
    except Exception as e:
        print(f"✗ Custom variables test failed: {e}")
    
    # Test 8: Normalization functionality
    print("\n[Test 8] Normalization Functionality")
    try:
        # First compute statistics if not exists
        stats_file = os.path.join(data_dir, 'statistics_train.json')
        if not os.path.exists(stats_file):
            print("  Computing statistics first...")
            dataset.compute_statistics(num_samples=min(10, len(dataset)))
        
        # Create normalized dataset
        normalized_dataset = GLORYS12Dataset(
            data_dir=data_dir,
            dataset_type=dataset_type,
            variables=['thetao', 'so', 'uo', 'vo'],
            normalize=True
        )
        
        # Load samples
        norm_sample = normalized_dataset[0]
        original_sample = dataset[0]
        
        print(f"✓ Normalization works successfully")
        print(f"  - Sample shape: {norm_sample.shape}")
        
        # Test each variable separately
        print("\n  Testing each variable:")
        for var_idx, var_name in enumerate(normalized_dataset.variables):
            norm_var = norm_sample[var_idx]
            orig_var = original_sample[var_idx]
            
            # Check for NaN values
            has_nan_orig = torch.isnan(orig_var).any().item()
            has_nan_norm = torch.isnan(norm_var).any().item()
            
            # Get valid (non-NaN) values
            valid_mask = ~torch.isnan(orig_var)
            norm_valid = norm_var[valid_mask]
            
            # Get NaN positions and check if they're set to 0
            nan_mask = torch.isnan(orig_var)
            nan_values = norm_var[nan_mask]
            all_nan_are_zero = (nan_values == 0.0).all().item() if nan_mask.any() else True
            
            print(f"\n    [{var_name}]")
            print(f"      Original - min: {orig_var[valid_mask].min().item():.4f}, max: {orig_var[valid_mask].max().item():.4f}")
            print(f"      Normalized - min: {norm_valid.min().item():.4f}, max: {norm_valid.max().item():.4f}")
            print(f"      Has NaN in original: {has_nan_orig}")
            print(f"      Has NaN in normalized: {has_nan_norm}")
            if has_nan_orig:
                print(f"      NaN count: {nan_mask.sum().item()}")
                print(f"      All NaN->0: {all_nan_are_zero}")
            
            # Verify normalization range for valid values
            if len(norm_valid) > 0:
                if norm_valid.min() < -1.01 or norm_valid.max() > 1.01:
                    print(f"      ⚠ Warning: Values outside [-1, 1] range!")
        
        # Test denormalization
        print("\n  Testing denormalization:")
        denorm_sample = normalized_dataset._denormalize_data(norm_sample)
                
        # Print debug info
        print(f"\n  Debug Info:")
        print(f"    Original sample shape: {original_sample.shape}")
        print(f"    Normalized sample shape: {norm_sample.shape}")
        print(f"    Denormalized sample shape: {denorm_sample.shape}")
        print(f"    Statistics loaded: {normalized_dataset.statistics}")
                
        for var_idx, var_name in enumerate(normalized_dataset.variables):
            orig_var = original_sample[var_idx]
            denorm_var = denorm_sample[var_idx]
                    
            # Only compare valid (non-NaN) values
            valid_mask = ~torch.isnan(orig_var)
                    
            if valid_mask.any():
                orig_valid = orig_var[valid_mask]
                denorm_valid = denorm_var[valid_mask]
                        
                # Calculate max difference
                max_diff = (orig_valid - denorm_valid).abs().max().item()
                        
                # Debug: show actual values
                print(f"\n    [{var_name}]")
                print(f"      Original (valid) - min: {orig_valid.min().item():.6f}, max: {orig_valid.max().item():.6f}, mean: {orig_valid.mean().item():.6f}")
                print(f"      Denormalized (valid) - min: {denorm_valid.min().item():.6f}, max: {denorm_valid.max().item():.6f}, mean: {denorm_valid.mean().item():.6f}")
                print(f"      Max reconstruction error: {max_diff:.6f}")
                        
                if max_diff > 1e-4:
                    print(f"      ⚠ Warning: Large reconstruction error!")
        
    except Exception as e:
        print(f"✗ Normalization test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 9: Visualize variable distribution (optional, commented out by default)
    print("\n[Test 9] Variable Distribution Visualization")
    try:
        print("Note: This test is optional and may take some time.")
        print("Uncomment the following lines to test distribution visualization:")
        # dataset.visualize_distribution(
        #     num_samples=10,  # Use only 10 samples for quick testing
        #     bins=30,
        #     output_file='./test_distribution.png'
        # )
        print("✓ Visualization method available (test skipped)")
    except Exception as e:
        print(f"✗ Distribution visualization test failed: {e}")
    
    # Test 10: Random Samples Visualization
    print("\n[Test 10] Random Samples Visualization")
    try:
        # We need a dataset with Argo for this test
        if 'dataset_argo' in locals():
            print("Visualizing random samples with Argo data...")
            dataset_argo.visualize_random_samples(
                num_samples=2,
                output_dir='./output/test_visualization',
                verbose=True
            )
            print("✓ Random samples visualization completed")
        else:
            print("⚠ Argo dataset not available, skipping visualization test")
    except Exception as e:
        print(f"✗ Random samples visualization test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Testing completed!")
    print("=" * 80)


if __name__ == '__main__':
    """
    Run test function when script is executed directly
    """
    import sys
    # Add project root to sys.path to allow importing modules
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import argparse
    
    parser = argparse.ArgumentParser(description='Test GLORYS12Dataset class')
    parser.add_argument('--data-dir', type=str, default='./data/glorys12',
                        help='Root directory containing GLORYS12 data')
    parser.add_argument('--dataset-type', type=str, default='train',
                        choices=['train', 'cal', 'test'],
                        help='Dataset type to test')
    
    args = parser.parse_args()
    
    # Run the test
    test_glorys12_dataset(
        data_dir=args.data_dir,
        dataset_type=args.dataset_type
    )
