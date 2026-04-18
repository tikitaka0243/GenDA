from datetime import datetime, timedelta
import os
import shutil
from typing import Optional, Union
import copernicusmarine
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from modules.utils import get_random_files
from modules.plot.plot import plot_ocean_variables
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


class GLORYS12Processor():

    def __init__(self, username: str, password: str,
                 base_path: str = "GLOBAL_MULTIYEAR_PHY_001_030",
                 data_dir='./data/glorys12'):
        """
        Initialize the GLORYS12 processor with authentication and paths.
        """
        self.username = username
        self.password = password
        self.base_path = base_path

        self.data_dir = data_dir

    def get_datasetid(self, year, month):
        year = int(year)
        month = int(month)

        if (year == 2021 and month >= 7) or (year >= 2022):
            dataset_id = 'cmems_mod_glo_phy_myint_0.083deg_P1D-m'
        else:
            dataset_id = 'cmems_mod_glo_phy_my_0.083deg_P1D-m'

        # return dataset_id
        return 'cmems_mod_glo_phy_my_0.083deg_P1D-m'

    def download_data(self, year, month, day=None, filter_pattern: str = None) -> None:
        """
        Download GLORYS12 data for specified time period.
        
        Args:
            year: Target year (YYYY)
            month: Target month (MM)
            filter_pattern: Optional filter pattern for files
        """
        year = int(year)
        month = int(month)

        if filter_pattern is None:
            if day is not None:
                day = int(day)
                filter_pattern = f"*_{year}{month:02d}{day:02d}_*"
            else:
                filter_pattern = f"*_{year}{month:02d}*"

        copernicusmarine.get(
            username=self.username,
            password=self.password,
            dataset_id=self.get_datasetid(year, month),
            filter=filter_pattern,
            skip_existing=True
        )

    def load_data(
        self, 
        year: Union[str, int], 
        month: Union[str, int], 
        day: Union[str, int], 
        depth_idx: int = None,
        variables: list = ["thetao", "so", "uo", "vo"]
    ) -> xr.Dataset:
        """
        Load ocean data for specified date, depth level and variables.
        
        Args:
            year: Target year (YYYY) as string or integer
            month: Target month (MM) as string or integer  
            day: Target day (DD) as string or integer
            depth_idx: Depth level index (optional)
            variables: List of variables to extract (default: ["thetao", "so", "uo", "vo"])
            
        Returns:
            xarray.Dataset containing requested variables at specified depth
        """
        
        # Convert year, month, day to string with zero-padding if needed
        year_str = str(year).zfill(4)  # Ensure 4-digit year
        month_str = str(month).zfill(2)  # Ensure 2-digit month
        day_str = str(day).zfill(2)  # Ensure 2-digit day
        
        folder_path = os.path.join(self.base_path, self.get_datasetid(year_str, month_str) + '_202311', year_str, month_str)
        file_path = self._find_nc_file_by_time(folder_path, year_str + month_str + day_str)
        
        if file_path is None:
            raise FileNotFoundError(f"No matching .nc file found for {year_str}-{month_str}-{day_str}")
            
        full_path = os.path.join(folder_path, file_path)
        ds = xr.open_dataset(full_path)

        # Create new Dataset with selected variables at specified depth
        result = xr.Dataset()
        for var in variables:
            if var in ds:
                if depth_idx is not None:
                    result[var] = ds[var].isel(depth=depth_idx)
                else:
                    result[var] = ds[var]
            else:
                print(f"Warning: Variable '{var}' not found in dataset")

        # Convert float64 to float32 to save memory
        result_float32 = result.copy()
        for var in result.data_vars:
            if result[var].dtype == 'float64':
                result_float32[var] = result[var].astype('float32')
        
        return result_float32

    def gen_all_metadata(self):
        self.gen_metadata(dataset_type="train", sample_size=100000)
        self.gen_metadata(dataset_type="cal", sample_size=10000)
        self.gen_metadata(dataset_type="test", sample_size=10000)

    def gen_metadata(self, sample_size=100000, dataset_type="train"):
        """
        Generate metadata for samples with timestamps and depth indices based on dataset type.
        Ensures unique (timestamp, depth_index) combinations and sorts by timestamp then depth_index.
        
        Parameters:
        -----------
        sample_size : int
            Number of samples to generate
        dataset_type : str
            Type of dataset: "train", "cal", or "test"
            - train: 2012-2021
            - cal: 2022
            - test: 2023
            
        Returns:
        --------
        pandas.DataFrame
            Generated metadata with timestamp and depth_index columns
        """
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created directory: {self.data_dir}")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Define date range based on dataset type
        if dataset_type == "train":
            # Training data: 2012-2021
            start_date = datetime(2012, 1, 1)
            end_date = datetime(2021, 12, 31)
            date_range_desc = "2012-2021"
        elif dataset_type == "cal":
            # Calibration data: 2022
            start_date = datetime(2022, 1, 1)
            end_date = datetime(2022, 12, 31)
            date_range_desc = "2022"
        elif dataset_type == "test":
            # Test data: 2023
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 12, 31)
            date_range_desc = "2023"
        else:
            raise ValueError("dataset_type must be 'train', 'cal', or 'test'")
        
        # Calculate total number of days in the range
        total_days = (end_date - start_date).days + 1
        
        # Calculate maximum possible unique samples
        # Total unique combinations = days × depth indices (50 depth indices)
        max_unique_samples = total_days * 50
        
        if sample_size > max_unique_samples:
            raise ValueError(f"Requested sample_size ({sample_size}) exceeds maximum unique samples "
                            f"({max_unique_samples}) for {date_range_desc}")
        
        print(f"Generating {sample_size} unique samples from {max_unique_samples} possible combinations...")
        
        # Use a set to track unique combinations for efficient duplicate detection
        unique_combinations = set()
        timestamps_list = []
        depth_indices_list = []
        
        # Generate unique samples
        while len(timestamps_list) < sample_size:
            # Generate random day offset and depth index
            day_offset = np.random.randint(0, total_days)
            depth_idx = np.random.randint(0, 50)
            
            # Create unique key for the combination
            combination_key = (day_offset, depth_idx)
            
            # Check if combination is unique
            if combination_key not in unique_combinations:
                unique_combinations.add(combination_key)
                
                # Create timestamp from day offset
                random_timestamp = start_date + timedelta(days=day_offset)
                random_timestamp = random_timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
                
                timestamps_list.append(random_timestamp)
                depth_indices_list.append(depth_idx)
        
        # Create DataFrame from generated data
        metadata_df = pd.DataFrame({
            'timestamp': timestamps_list,
            'depth_index': depth_indices_list
        })
        
        # Sort by timestamp (ascending) and then by depth_index (ascending)
        metadata_df = metadata_df.sort_values(['timestamp', 'depth_index']).reset_index(drop=True)
        
        # Define output file path with dataset type in filename
        output_file = os.path.join(self.data_dir, f"metadata_{dataset_type}.csv")
        
        # Save to CSV file
        metadata_df.to_csv(output_file, index=False)
        
        print(f"Metadata generated successfully!")
        print(f"Dataset type: {dataset_type}")
        print(f"Time range: {date_range_desc}")
        print(f"Sample size: {sample_size} unique samples")
        print(f"Actual time range: {metadata_df['timestamp'].min()} to {metadata_df['timestamp'].max()}")
        print(f"Depth index range: {metadata_df['depth_index'].min()} to {metadata_df['depth_index'].max()}")
        print(f"File saved to: {output_file}")
        
        # Verify uniqueness
        unique_count = len(metadata_df.drop_duplicates(['timestamp', 'depth_index']))
        if unique_count == sample_size:
            print(f"✓ All samples are unique (verified: {unique_count} unique entries)")
        else:
            print(f"⚠ Warning: Found duplicates! Expected {sample_size}, got {unique_count} unique entries")
        
        return metadata_df

    def dowload_and_process(self, year_range=range(2012, 2022)):

        for year in year_range:

            if year in range(2012, 2022):
                data_type = "train"
            elif year == 2022:
                data_type = "cal"
            elif year >= 2023:
                data_type = "test"

            metadata = pd.read_csv(os.path.join(self.data_dir, f"metadata_{data_type}.csv"))
            metadata['timestamp'] = pd.to_datetime(metadata['timestamp'])

            for month in range(1, 13):
                
                filtered_metadata = metadata[(metadata['timestamp'].dt.year == year) & 
                         (metadata['timestamp'].dt.month == month)]

                for day in tqdm(range(1, 32)):
                    filtered_metadata_2 = filtered_metadata[(filtered_metadata['timestamp'].dt.day == day)]

                    download = True

                    if len(filtered_metadata_2) > 0:
                        for index, row in tqdm(filtered_metadata_2.iterrows()):
                            depth_index = row['depth_index']

                            output_path = os.path.join(self.data_dir, data_type, f"{year}{month:02d}{day:02d}_d{depth_index:02d}.nc")

                            if os.path.exists(output_path):
                                print(f"File already exists: {output_path}")
                                continue

                            if download:
                                self.download_data(year, month, day)
                                download = False

                            ds = self.load_data(year, month, day, depth_idx=depth_index)

                            # Define target grid for interpolation (0.5° resolution)
                            target_lon = np.arange(-180, 180, 0.5)
                            target_lat = np.arange(-80, 90.5, 0.5)

                            # Perform interpolation to 0.5° resolution
                            ds_interp = ds.interp(
                                longitude=target_lon,
                                latitude=target_lat,
                                method='linear'
                            )

                            # Save with compression using the encapsulated function
                            self.save_compressed_netcdf(ds_interp, output_path)
                            
                            # Close the datasets to free memory
                            ds.close()
                            ds_interp.close()
            
                path_to_delete = os.path.join(self.base_path, self.get_datasetid(year, month) + '_202311', str(year), f"{month:02d}")
                if os.path.exists(path_to_delete):
                    shutil.rmtree(path_to_delete)
                    print(f"Deleted {path_to_delete}")

    def dowload_and_process_first40_daily(self, year_range=range(2013, 2025), depth_range=range(0, 40)):

        for year in year_range:

            if year in range(2011, 2023):
                data_type = "train"
            elif year in range(2023, 2024):
                data_type = "val"
            elif year in range(2024, 2025):
                data_type = "test"
            else:
                continue

            for month in tqdm(range(1, 13), desc=f"Processing year {year}"):

                if month == 12:
                    next_month_start = datetime(int(year) + 1, 1, 1)
                else:
                    next_month_start = datetime(int(year), int(month) + 1, 1)
                days_in_month = (next_month_start - timedelta(days=1)).day

                for day in tqdm(range(1, days_in_month + 1)):

                    download = True

                    for depth_index in depth_range:

                        output_path = os.path.join(self.data_dir, data_type, f"{year}{month:02d}{day:02d}_d{depth_index:02d}.nc")

                        if os.path.exists(output_path):
                            continue

                        if download:
                            self.download_data(year, month, day)
                            download = False

                        try:
                            ds = self.load_data(year, month, day, depth_idx=int(depth_index))
                        except FileNotFoundError as e:
                            print(e)
                            break

                        target_lon = np.arange(-180, 180, 0.5)
                        target_lat = np.arange(-80, 90.5, 0.5)

                        ds_interp = ds.interp(
                            longitude=target_lon,
                            latitude=target_lat,
                            method='linear'
                        )

                        self.save_compressed_netcdf(ds_interp, output_path)

                        ds.close()
                        ds_interp.close()

                path_to_delete = os.path.join(self.base_path, self.get_datasetid(year, month) + '_202311', str(year), f"{month:02d}")
                if os.path.exists(path_to_delete):
                    shutil.rmtree(path_to_delete)
                    print(f"Deleted {path_to_delete}")

    def plot_random_samples(self, data_folder_path='./data/glorys12/train'):
        """
        Plot random samples from the processed data.
        
        Args:
            data_folder_path (str): Path to the folder containing processed data files.
        """
        
        samples_path = get_random_files(data_folder_path, n=5)
        
        for i, sample_path in enumerate(samples_path):
            ds = xr.open_dataset(sample_path)
            plot_ocean_variables(ds, output_path=f"./output/plot/data/glorys12/random_select_samples/{i}.png")
                
    def save_compressed_netcdf(self, dataset, output_path, compression_settings=None):
        """
        Save xarray Dataset to NetCDF file with lossless compression.
        
        Args:
            dataset (xr.Dataset): Dataset to be saved
            output_path (str): Full path for output file
            compression_settings (dict, optional): Custom compression settings for specific variables
            
        Returns:
            str: Path of the saved file
        """
        # Default compression settings
        default_compression = {
            'zlib': True,           # Enable DEFLATE compression
            'complevel': 6,         # Compression level (1-9, 6 is good balance)
            'shuffle': True,        # Enable byte-shuffle for better compression
            'dtype': 'float32'      # Ensure float32 type
        }
        
        # Variable-specific compression settings (optional)
        if compression_settings is None:
            compression_settings = {
                'thetao': {'complevel': 6},  # Temperature
                'so': {'complevel': 6},       # Salinity  
                'uo': {'complevel': 6},       # Zonal current
                'vo': {'complevel': 6}        # Meridional current
            }
        
        # Build encoding dictionary
        encoding = {}
        for var_name in dataset.data_vars:
            # Start with default settings
            var_encoding = default_compression.copy()
            
            # Apply variable-specific settings if available
            if var_name in compression_settings:
                var_encoding.update(compression_settings[var_name])
            
            encoding[var_name] = var_encoding
        
        # Add encoding for coordinates to prevent compression issues
        for coord_name in dataset.coords:
            encoding[coord_name] = {'_FillValue': None}
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        
        # Save dataset with compression
        dataset.to_netcdf(
            output_path,
            encoding=encoding,
            engine='netcdf4',          # Use netCDF4 format for compression
            format='NETCDF4'
        )
                    

    def visualize_all_depths(self, year: int, month: int, day: int, 
                             output_dir: str = './output/visualization/all_depths',
                             data_type: str = 'train',
                             dpi: int = 600,
                             clean_mode: bool = False,
                             depth: Optional[int] = None,
                             add_noise: bool = False,
                             noise_level: float = 0.05) -> None:
        """
        Visualize reanalysis data for depth layers of four variables
        at a specified time. Each variable at each depth is a separate plot.
        
        Args:
            year: Target year
            month: Target month
            day: Target day
            output_dir: Output directory path
            data_type: Data type ('train', 'val', 'test')
            dpi: Image resolution (dots per inch)
            clean_mode: If True, output clean ocean field only without borders,
                       coordinates, title, colorbar, and axis labels
            depth: Specific depth layer to visualize (0-39). If None, visualize all depths.
            add_noise: If True, add Gaussian noise to the data fields
            noise_level: Relative noise level (standard deviation as fraction of data range)
        
        Returns:
            None: Saves images to the specified directory
        """
        # Set global font to Times New Roman
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Variable definitions
        variables = ['thetao', 'so', 'uo', 'vo']
        var_titles = {
            'thetao': 'Sea Water Potential Temperature (°C)',
            'so': 'Sea Water Salinity (psu)',
            'uo': 'Eastward Sea Water Velocity (m/s)',
            'vo': 'Northward Sea Water Velocity (m/s)'
        }
        var_cmaps = {
            'thetao': 'plasma',
            'so': 'viridis',
            'uo': 'RdBu_r',
            'vo': 'RdGy_r'
        }
        
        # Create date string
        date_str = f"{year}{month:02d}{day:02d}"
        
        # Create output directory with suffix for clean mode
        if clean_mode:
            date_output_dir = os.path.join(output_dir, f"{date_str}_clean")
        else:
            date_output_dir = os.path.join(output_dir, date_str)
        os.makedirs(date_output_dir, exist_ok=True)
        
        # Determine depth layers to visualize
        if depth is None:
            depth_layers = range(40)
            print(f"Starting visualization for {date_str} across all depth layers...")
        else:
            depth_layers = [depth]
            print(f"Starting visualization for {date_str} at depth layer {depth}...")
        print(f"Output directory: {date_output_dir}")
        
        # Iterate through depth layers
        for depth_idx in depth_layers:
            # Build data file path
            data_file = os.path.join(self.data_dir, data_type, f"{date_str}_d{depth_idx:02d}.nc")
            
            # Check if file exists
            if not os.path.exists(data_file):
                print(f"Warning: File does not exist {data_file}, skipping depth layer {depth_idx}")
                continue
            
            # Load data
            try:
                ds = xr.open_dataset(data_file)
            except Exception as e:
                print(f"Error: Unable to load file {data_file}: {e}")
                continue
            
            # Create separate plots for each variable
            for var in variables:
                if var not in ds:
                    print(f"Warning: Variable {var} not in dataset, skipping")
                    continue
                
                # Extract data
                data = ds[var]
                if 'time' in data.dims:
                    data = data.isel(time=0)
                
                # Calculate color range (using quantiles)
                data_values = data.values.copy()
                
                # Add Gaussian noise if requested
                if add_noise:
                    valid_mask = ~np.isnan(data_values)
                    valid_data = data_values[valid_mask]
                    if len(valid_data) > 0:
                        data_range = np.nanmax(data_values) - np.nanmin(data_values)
                        noise_std = noise_level * data_range
                        noise = np.random.normal(0, noise_std, data_values.shape)
                        data_values[valid_mask] += noise[valid_mask]
                
                valid_data = data_values[~np.isnan(data_values)]
                
                if len(valid_data) == 0:
                    print(f"Warning: {var} at depth layer {depth_idx} has no valid data")
                    plt.close()
                    continue
                
                # Set color range based on variable type
                if var == 'thetao':
                    # Temperature: use global min/max
                    vmin = np.nanmin(data_values)
                    vmax = np.nanmax(data_values)
                elif var == 'so':
                    # Salinity: use 0.5% and 99.5% quantiles
                    vmin = np.percentile(valid_data, 0.5)
                    vmax = np.percentile(valid_data, 99.5)
                else:  # uo, vo
                    # Velocity: symmetric 99.5% quantile
                    vmax = np.percentile(np.abs(valid_data), 99.5)
                    vmin = -vmax
                
                if clean_mode:
                    # Clean mode: output ocean field with coastline and land, but no labels/titles/colorbar
                    fig = plt.figure(figsize=(10, 5))
                    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                    
                    # Create map
                    im = ax.pcolormesh(data.longitude, data.latitude, data_values,
                                      cmap=var_cmaps[var], shading='auto',
                                      vmin=vmin, vmax=vmax,
                                      transform=ccrs.PlateCarree())
                    
                    # Add coastline and land features (kept in clean mode)
                    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
                    
                    # Remove all axes, borders, and labels
                    ax.set_frame_on(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    # Set map extent
                    ax.set_extent([-180, 180, -80, 90], crs=ccrs.PlateCarree())
                    
                    # Save image without any decorations
                    output_filename = f"{date_str}_{var}_d{depth_idx:02d}_clean.png"
                    output_path = os.path.join(date_output_dir, output_filename)
                    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                               facecolor='white', pad_inches=0)
                    plt.close()
                else:
                    # Normal mode: full visualization with all elements
                    fig = plt.figure(figsize=(6, 4))
                    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                    
                    # Create map
                    im = ax.pcolormesh(data.longitude, data.latitude, data_values,
                                      cmap=var_cmaps[var], shading='auto',
                                      vmin=vmin, vmax=vmax,
                                      transform=ccrs.PlateCarree())
                    
                    # Add coastline and land features
                    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax, shrink=0.6, extend='both', pad=0.03)
                    # Extract units
                    title_text = var_titles[var]
                    if '(' in title_text and ')' in title_text:
                        units = title_text.split('(')[-1].split(')')[0]
                    else:
                        units = var
                    cbar.set_label(units, rotation=90)
                    
                    # Set title
                    ax.set_title(f"{var_titles[var]}\nDepth Layer {depth_idx}", 
                                fontsize=12, fontweight='bold')
                    
                    # Add gridlines
                    gl = ax.gridlines(draw_labels=True, alpha=0.3, linestyle='--')
                    gl.top_labels = False
                    gl.right_labels = False
                    gl.xlabel_style = {'size': 10}
                    gl.ylabel_style = {'size': 10}
                    
                    # Set map extent
                    ax.set_extent([-180, 180, -80, 90], crs=ccrs.PlateCarree())
                    
                    # Save image
                    output_filename = f"{date_str}_{var}_d{depth_idx:02d}.png"
                    output_path = os.path.join(date_output_dir, output_filename)
                    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
                    plt.close()
                
                print(f"Saved: {output_filename}")
            
            # Close dataset
            ds.close()
        
        if depth is None:
            print(f"\nVisualization complete! Generated 160 image files saved to: {date_output_dir}")
            print(f"File naming format: {date_str}_{{variable}}_d{{depth:02d}}.png")
            print(f"  - Variables: thetao (temperature), so (salinity), uo (zonal velocity), vo (meridional velocity)")
            print(f"  - Depth layers: 00-39 (40 layers total)")
        else:
            print(f"\nVisualization complete! Generated 4 image files saved to: {date_output_dir}")
            print(f"File naming format: {date_str}_{{variable}}_d{depth:02d}.png")
            print(f"  - Variables: thetao (temperature), so (salinity), uo (zonal velocity), vo (meridional velocity)")
            print(f"  - Depth layer: {depth}")

    def compute_spatial_std_statistics(self, data_type: str = 'test', 
                                        output_file: str = None,
                                        batch_size: int = 10,
                                        num_workers: int = None) -> pd.DataFrame:
        """
        Compute spatial standard deviation for each position (latitude, longitude) 
        in the test dataset.
        
        For each variable (thetao, so, uo, vo), collect data at each position across
        all depth layers (0-39) and all time points, then compute the overall
        standard deviation.
        
        Uses Welford's online algorithm for memory-efficient incremental computation:
        1. Group files by depth layer
        2. Process files incrementally, updating running statistics (count, mean, M2)
        3. Compute standard deviation: sqrt(M2 / (n-1))
        
        Args:
            data_type: Dataset type ('train', 'val', 'test'), default 'test'
            output_file: Output CSV file path, default ./data/glorys12/{data_type}_spatial_std.csv
            batch_size: Number of depth layers to process per batch (for memory control)
            num_workers: Number of worker processes for parallel file processing, 
                        default None (use all CPU cores)
            
        Returns:
            pd.DataFrame: Statistics for each variable at each position
                columns: variable, latitude, longitude, std, mean, count
        """
        import re
        from collections import defaultdict
        
        if output_file is None:
            output_file = os.path.join(self.data_dir, f'{data_type}_spatial_std.csv')
        
        data_folder = os.path.join(self.data_dir, data_type)
        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"Data folder not found: {data_folder}")
        
        # Get all files and group by depth
        all_files = [f for f in os.listdir(data_folder) if f.endswith('.nc')]
        
        # Group files by depth index
        depth_files = defaultdict(list)
        for filename in all_files:
            match = re.search(r'_d(\d{2})\.nc$', filename)
            if match:
                depth_idx = int(match.group(1))
                depth_files[depth_idx].append(os.path.join(data_folder, filename))
        
        if not depth_files:
            raise ValueError(f"No valid data files found in {data_folder}")
        
        variables = ['thetao', 'so', 'uo', 'vo']
        
        # Get spatial dimensions from first file
        sample_file = list(depth_files.values())[0][0]
        with xr.open_dataset(sample_file) as ds:
            lat_coords = ds.latitude.values
            lon_coords = ds.longitude.values
        
        n_lat, n_lon = len(lat_coords), len(lon_coords)
        
        # Set default num_workers
        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)
        
        print(f"Computing spatial standard deviation for {data_type} dataset...")
        print(f"Spatial grid: {n_lat} x {n_lon} = {n_lat * n_lon} positions")
        print(f"Depth layers: {len(depth_files)} (0-{max(depth_files.keys())})")
        print(f"Total files: {sum(len(files) for files in depth_files.values())}")
        print(f"Using {num_workers} worker processes for parallel processing")
        
        # Process by depth layers with progress bar
        sorted_depths = sorted(depth_files.keys())
        
        # Initialize global statistics
        stats = {}
        for var in variables:
            stats[var] = {
                'count': np.zeros((n_lat, n_lon), dtype=np.int64),
                'mean': np.zeros((n_lat, n_lon), dtype=np.float64),
                'M2': np.zeros((n_lat, n_lon), dtype=np.float64),
            }
        
        for depth_idx in tqdm(sorted_depths, desc="Processing depth layers"):
            files = depth_files[depth_idx]
            
            # Prepare arguments for parallel processing
            args_list = [(fp, variables, n_lat, n_lon) for fp in files]
            
            # Process files in parallel using multiprocessing
            with Pool(processes=num_workers) as pool:
                results = pool.map(_process_file_worker, args_list)
            
            # Merge results from all workers
            for local_stats in results:
                if local_stats is not None:
                    self._merge_stats(stats, local_stats, variables)
        
        # Compute standard deviation and build results DataFrame
        results = []
        for var in variables:
            count = stats[var]['count']
            mean = stats[var]['mean']
            M2 = stats[var]['M2']
            
            # Compute standard deviation (sample std: sqrt(M2 / (n-1)) for n>1)
            std = np.zeros_like(mean)
            valid_count = count > 1
            std[valid_count] = np.sqrt(M2[valid_count] / (count[valid_count] - 1))
            
            # Create record for each position
            for i in range(n_lat):
                for j in range(n_lon):
                    if count[i, j] > 0:
                        results.append({
                            'variable': var,
                            'latitude': lat_coords[i],
                            'longitude': lon_coords[j],
                            'std': std[i, j],
                            'mean': mean[i, j],
                            'count': int(count[i, j]),
                        })
        
        # Create DataFrame and save
        df = pd.DataFrame(results)
        
        # Print summary statistics
        print(f"\nSpatial standard deviation statistics:")
        for var in variables:
            var_data = df[df['variable'] == var]
            if len(var_data) > 0:
                print(f"  {var}: mean_std={var_data['std'].mean():.4f}, "
                      f"max_std={var_data['std'].max():.4f}, "
                      f"valid_positions={len(var_data)}")
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        print(f"Total records: {len(df)}")
        
        return df
    
    def _merge_stats(self, global_stats, local_stats, variables):
        """Merge local statistics into global statistics using parallel Welford's algorithm."""
        for var in variables:
            if local_stats is None:
                continue
            
            count_a = global_stats[var]['count']
            mean_a = global_stats[var]['mean']
            M2_a = global_stats[var]['M2']
            
            count_b = local_stats[var]['count']
            mean_b = local_stats[var]['mean']
            M2_b = local_stats[var]['M2']
            
            # Find positions where both have data
            valid_merge = (count_a > 0) & (count_b > 0)
            valid_only_b = (count_a == 0) & (count_b > 0)
            
            # For positions with data in both: merge using parallel algorithm
            if np.any(valid_merge):
                n_a = count_a[valid_merge].astype(np.float64)
                n_b = count_b[valid_merge].astype(np.float64)
                mu_a = mean_a[valid_merge]
                mu_b = mean_b[valid_merge]
                
                n_total = n_a + n_b
                delta = mu_b - mu_a
                
                # Update mean
                mean_a[valid_merge] = mu_a + delta * n_b / n_total
                
                # Update M2
                M2_a[valid_merge] = M2_a[valid_merge] + M2_b[valid_merge] + delta**2 * n_a * n_b / n_total
                
                # Update count
                count_a[valid_merge] = n_total.astype(np.int64)
            
            # For positions with data only in local: copy directly
            if np.any(valid_only_b):
                count_a[valid_only_b] = count_b[valid_only_b]
                mean_a[valid_only_b] = mean_b[valid_only_b]
                M2_a[valid_only_b] = M2_b[valid_only_b]


def _process_file_worker(args):
    """Worker function for parallel file processing (must be at module level for pickling)."""
    file_path, variables, n_lat, n_lon = args
    
    # Initialize local statistics
    local_stats = {}
    for var in variables:
        local_stats[var] = {
            'count': np.zeros((n_lat, n_lon), dtype=np.int64),
            'mean': np.zeros((n_lat, n_lon), dtype=np.float64),
            'M2': np.zeros((n_lat, n_lon), dtype=np.float64),
        }
    
    try:
        with xr.open_dataset(file_path) as ds:
            for var in variables:
                if var not in ds:
                    continue
                
                # Extract data and remove time dimension
                data = ds[var].values[0]  # shape: (lat, lon)
                
                # Create valid data mask (exclude NaN)
                valid_mask = ~np.isnan(data)
                
                if not np.any(valid_mask):
                    continue
                
                # Update statistics using Welford's algorithm
                count = local_stats[var]['count']
                mean = local_stats[var]['mean']
                M2 = local_stats[var]['M2']
                
                # Update only valid data positions
                valid_indices = np.where(valid_mask)
                for i, j in zip(valid_indices[0], valid_indices[1]):
                    x = data[i, j]
                    count[i, j] += 1
                    delta = x - mean[i, j]
                    mean[i, j] += delta / count[i, j]
                    delta2 = x - mean[i, j]
                    M2[i, j] += delta * delta2
                
    except Exception as e:
        print(f"Warning: Error processing {file_path}: {e}")
        return None
    
    return local_stats

    @staticmethod
    def _find_nc_file_by_time(folder_path: str, target_time: str) -> Optional[str]:
        """
        Helper method to find .nc file matching target timestamp.
        
        Args:
            folder_path: Directory containing .nc files
            target_time: Target timestamp (YYYYMMDD)
            
        Returns:
            Matching filename if found, None otherwise
        """

        if not os.path.exists(folder_path):
            return None
            
        for filename in os.listdir(folder_path):
            if filename.endswith('.nc'):
                parts = filename.split('_')
                for part in parts:
                    if len(part) == 8 and part.isdigit() and part == target_time:
                        return filename
        return None
