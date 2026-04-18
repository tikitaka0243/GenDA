import os
import glob
from collections import defaultdict
import re
import random
import csv
from datetime import datetime
from typing import Dict
import xarray as xr
from tqdm import tqdm
import requests
import pandas as pd
import numpy as np
import pickle
import hashlib

from modules.plot.plot import plot_argo_variables, plot_argo_nc_profiles, plot_argo_daily_profile_counts, plot_argo_spatial_distribution


class ArgoProcessorRaw():
    
    def __init__(self, data_dir='./data/argo'):
        self.data_dir = data_dir
        self.raw_data_dir = os.path.join(self.data_dir, 'raw')
        
        self.raw_files = self.get_argo_nc_files()
        print(f"Found {len(self.raw_files)} .nc files in Argo data directory.")
        self.print_file_stats()
    
    def get_argo_nc_files(self):
        """
        Get all .nc files from subdirectories starting with 'argo' in the data folder
        
        Returns:
            list: A list containing paths of all .nc files
        """
        
        # Find all subdirectories starting with 'argo'
        argo_dirs = glob.glob(os.path.join(self.raw_data_dir, 'argo*'))
        
        nc_files = []
        # Iterate through each argo directory to find .nc files
        for argo_dir in argo_dirs:
            if os.path.isdir(argo_dir):
                # Search for .nc files in the current argo directory and its subdirectories
                for root, dirs, files in os.walk(argo_dir):
                    for file in files:
                        if file.endswith('.nc'):
                            nc_files.append(os.path.join(root, file))
        
        return nc_files
    
    def print_file_stats(self):
        """
        Print statistics of files starting with different letters
        Extract 1-2 leading alphabetic characters before numbers in filename
        """
        # Count files by leading letters
        letter_count = defaultdict(int)
        # Store files grouped by leading letters
        self.files_by_prefix = defaultdict(list)
        
        for file_path in self.raw_files:
            # Get filename without path
            filename = os.path.basename(file_path)
            # Extract leading alphabetic characters before numbers
            match = re.match(r'^([a-zA-Z]{1,2})', filename)
            if match:
                prefix = match.group(1)
                letter_count[prefix] += 1
                self.files_by_prefix[prefix].append(file_path)
            else:
                letter_count['other'] += 1
                self.files_by_prefix['other'].append(file_path)
        
        # Print statistics
        print("Files count by leading letters:")
        for prefix in sorted(letter_count.keys()):
            print(f"  {prefix}: {letter_count[prefix]}")
        
        # Check file completeness
        self._check_file_completeness()
        
        return dict(letter_count)
    
    def _check_file_completeness(self):
        """
        Check if all SR files have corresponding R files and SD files have corresponding D files.
        Matching is based on the numeric part after the prefix.
        """
        # Initialize missing files attributes
        self.missing_r_files = []
        self.missing_d_files = []
        
        # Check if all SR files have corresponding R files
        # Extract numeric part after prefix for matching
        if 'SR' in self.files_by_prefix:
            sr_files = self.files_by_prefix['SR']
            r_files = self.files_by_prefix['R'] if 'R' in self.files_by_prefix else []
            
            # Extract identifier parts from filenames (e.g., "2903009_344" from "SR2903009_344.nc")
            sr_numbers = {}
            for file_path in sr_files:
                filename = os.path.basename(file_path)
                match = re.search(r'^SR([a-zA-Z0-9_]+)\.', filename)
                if match:
                    identifier = match.group(1)
                    sr_numbers[identifier] = file_path
            
            r_number_set = set()
            for file_path in r_files:
                filename = os.path.basename(file_path)
                match = re.search(r'^R([a-zA-Z0-9_]+)\.', filename)
                if match:
                    r_number_set.add(match.group(1))
            
            # Check if all SR numbers have corresponding R numbers
            missing_r_identifiers = set(sr_numbers.keys()) - r_number_set
            if missing_r_identifiers:
                print(f"\nWarning: Found {len(missing_r_identifiers)} SR files without corresponding R files:")
                for identifier in sorted(missing_r_identifiers):
                    print(f"  SR{identifier} ({sr_numbers[identifier]})")
                    # Store the full path of missing R files
                    self.missing_r_files.append(sr_numbers[identifier])
            else:
                print("\nAll SR files have corresponding R files.")
        
        if 'SD' in self.files_by_prefix:
            sd_files = self.files_by_prefix['SD']
            d_files = self.files_by_prefix['D'] if 'D' in self.files_by_prefix else []
            
            # Extract identifier parts from filenames (e.g., "2903009_344" from "SD2903009_344.nc")
            sd_numbers = {}
            for file_path in sd_files:
                filename = os.path.basename(file_path)
                match = re.search(r'^SD([a-zA-Z0-9_]+)\.', filename)
                if match:
                    identifier = match.group(1)
                    sd_numbers[identifier] = file_path
            
            d_number_set = set()
            for file_path in d_files:
                filename = os.path.basename(file_path)
                match = re.search(r'^D([a-zA-Z0-9_]+)\.', filename)
                if match:
                    d_number_set.add(match.group(1))
            
            # Check if all SD numbers have corresponding D numbers
            missing_d_identifiers = set(sd_numbers.keys()) - d_number_set
            if missing_d_identifiers:
                print(f"\nWarning: Found {len(missing_d_identifiers)} SD files without corresponding D files:")
                for identifier in sorted(missing_d_identifiers):
                    # print(f"  SD{identifier} ({sd_numbers[identifier]})")
                    # Store the full path of missing D files
                    self.missing_d_files.append(sd_numbers[identifier])
            else:
                print("\nAll SD files have corresponding D files.")
    
    def read_d_files(self, file_path):
        """
        Read PRES_ADJUSTED, TEMP_ADJUSTED, PSAL_ADJUSTED variables from a single D file
        
        Args:
            file_path (str): Path to the D file to read
            
        Returns:
            dict: A dictionary containing the three variables data
        """
        try:
            # Open NetCDF file using xarray
            ds = xr.open_dataset(file_path)
            
            # Dictionary to store results
            result = {}
            
            # Check and extract variables
            if 'PRES_ADJUSTED' in ds.variables:
                result['PRES_ADJUSTED'] = ds['PRES_ADJUSTED'].values
            if 'TEMP_ADJUSTED' in ds.variables:
                result['TEMP_ADJUSTED'] = ds['TEMP_ADJUSTED'].values
            if 'PSAL_ADJUSTED' in ds.variables:
                result['PSAL_ADJUSTED'] = ds['PSAL_ADJUSTED'].values
            
            ds.close()
            return result
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return {}
        
    def read_r_files(self, file_path):
        pass
    
    def read_sd_files(self, file_path):
        pass
    
    def read_sr_files(self, file_path):
        pass

    def sample_and_inspect_files(self, n_per_prefix=3):
        """
        Randomly sample files from each prefix group and display basic information
        
        Args:
            n_per_prefix (int): Number of files to sample from each prefix group
        """
        print(f"Randomly sampling {n_per_prefix} files from each prefix group for inspection:")
        print("=" * 60)
        
        for prefix in sorted(self.files_by_prefix.keys()):
            files = self.files_by_prefix[prefix]
            # Sample n_per_prefix files or all files if there are fewer
            sampled_files = random.sample(files, min(n_per_prefix, len(files)))
            
            print(f"\nPrefix '{prefix}' samples:")
            for i, file_path in enumerate(sampled_files, 1):
                file_size = os.path.getsize(file_path)
                print(f"  {i}. {os.path.basename(file_path)}")
                print(f"     Path: {file_path}")
                    
                if prefix == 'D':
                    d_vars = self.read_d_files(file_path)
                    pres = d_vars['PRES_ADJUSTED']
                    temp = d_vars['TEMP_ADJUSTED']
                    psal = d_vars['PSAL_ADJUSTED']
                    
                elif prefix == 'R':
                    r_vars = self.read_r_files(file_path)
                    
                elif prefix == 'SD':
                    sd_vars = self.read_sd_files(file_path)
                    
                elif prefix == 'SR':
                    sr_vars = self.read_sr_files(file_path)
                
                # Plot key ARGO variables if they exist
                plot_filename = os.path.join("./output/plot/data/argo/random_selected_samples", f"argo_sample_{prefix}_{i}.png")
                print(f"     Creating plot: {plot_filename}")
                plot_argo_variables(pres, temp, psal, output_path=plot_filename)

            exit(0)

class ArgoProcessorCSV():
    def __init__(self, data_dir='./data/argo'):
        self.data_dir = data_dir
        self.csv_data_dir = os.path.join(self.data_dir, 'csv')

    def count_profiles_per_year(self, base_path: str = None, year_folders: list = None, 
                                 verbose: bool = True) -> Dict[int, int]:
        """
        Count the number of Argo profiles per year from CSV files.
        Profiles are identified by unique timestamps within each CSV file.
        
        Args:
            base_path (str, optional): Base path to CSV folders. If None, uses self.csv_data_dir.
            year_folders (list, optional): List of specific year folder names to process (e.g., ['csv_2013_2014']).
                                          If None, processes all year folders.
            verbose (bool, optional): Whether to print detailed information. Defaults to True.
        
        Returns:
            Dict[int, int]: Dictionary with years as keys and profile counts as values.
                           Format: {2013: 1500, 2014: 1800, ...}
        
        Example:
            >>> processor = ArgoProcessorCSV()
            >>> # Count profiles for all years
            >>> yearly_counts = processor.count_profiles_per_year()
            >>> print(yearly_counts)
            >>> # Count profiles for specific year folders
            >>> yearly_counts = processor.count_profiles_per_year(year_folders=['csv_2013_2014'])
        """
        # Use default path if not specified
        if base_path is None:
            base_path = self.csv_data_dir
            
        if not os.path.exists(base_path):
            raise ValueError(f"Path does not exist: {base_path}")
        
        # Get all year folders
        all_year_folders = sorted([f for f in os.listdir(base_path) 
                                  if os.path.isdir(os.path.join(base_path, f)) and re.search(r'\d{4}_\d{4}', f)])
        
        # Filter year folders based on user input
        if year_folders is not None:
            invalid_folders = [f for f in year_folders if f not in all_year_folders]
            if invalid_folders:
                print(f"⚠️  Warning: Following year folders not found and will be skipped: {invalid_folders}")
            
            folders_to_process = [f for f in year_folders if f in all_year_folders]
            
            if not folders_to_process:
                raise ValueError(f"None of the specified year folders exist in {base_path}. Available folders: {all_year_folders}")
        else:
            folders_to_process = all_year_folders
        
        if verbose:
            print(f"Found {len(all_year_folders)} year folders in total")
            print(f"Processing {len(folders_to_process)} year folders: {folders_to_process}")
        
        # Dictionary to store profile counts per year
        yearly_counts = defaultdict(int)
        total_profiles = 0
        
        # Process each year folder
        for year_folder in tqdm(folders_to_process, desc="Processing year folders", disable=not verbose):
            year_folder_path = os.path.join(base_path, year_folder)
            
            # Get all CSV files in this folder
            csv_files = sorted([f for f in os.listdir(year_folder_path) if f.endswith('.csv')])
            
            if verbose:
                print(f"\nProcessing folder: {year_folder} (Total {len(csv_files)} CSV files)")
            
            # Process each CSV file
            for csv_file in tqdm(csv_files, desc=f"  {year_folder}", leave=False, disable=not verbose):
                csv_path = os.path.join(year_folder_path, csv_file)
                
                try:
                    # Read CSV file
                    df = pd.read_csv(csv_path)
                    
                    # Check if required column exists
                    date_col = 'DATE (YYYY-MM-DDTHH:MI:SSZ)'
                    if date_col not in df.columns:
                        if verbose:
                            print(f"    ⚠️  Skipping {csv_file}: Missing date column")
                        continue
                    
                    # Get unique timestamps (each unique timestamp = one profile)
                    unique_timestamps = df[date_col].unique()
                    
                    # Count profiles per year
                    for timestamp in unique_timestamps:
                        try:
                            profile_datetime = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ')
                            year = profile_datetime.year
                            yearly_counts[year] += 1
                            total_profiles += 1
                        except ValueError:
                            if verbose:
                                print(f"    ⚠️  Invalid timestamp format: {timestamp}, skipping...")
                            continue
                
                except Exception as e:
                    if verbose:
                        print(f"    ❌ Error processing file {csv_file}: {str(e)}")
                    continue
        
        # Convert to regular dict and sort by year
        yearly_counts = dict(sorted(yearly_counts.items()))
        
        # Print statistics
        if verbose:
            print(f"\n📊 Profile counts per year:")
            for year in sorted(yearly_counts.keys()):
                print(f"   {year}: {yearly_counts[year]} profiles")
            print(f"\n   Total profiles: {sum(yearly_counts.values())}")
        
        return yearly_counts
    
    def check_csv_folder_year_consistency(self, base_path: str = None, verbose: bool = True) -> Dict[str, Dict]:
        """
        Check if CSV files in folders contain data matching the year range in folder names.
        
        Args:
            base_path (str, optional): Base path to CSV folders. If None, uses self.csv_data_dir.
            verbose (bool, optional): Whether to print detailed information. Defaults to True.
        
        Returns:
            Dict[str, Dict]: Dictionary containing check results for each folder:
                {
                    'folder_name': {
                        'expected_years': (start_year, end_year),
                        'total_files': int,
                        'checked_files': int,
                        'valid_files': int,
                        'invalid_files': int,
                        'error_files': int,
                        'invalid_details': [(filename, date_range), ...],
                        'error_details': [(filename, error_message), ...]
                    },
                    ...
                }
        
        Example:
            >>> processor = ArgoProcessorCSV()
            >>> results = processor.check_csv_folder_year_consistency()
            >>> print(results['csv_2013_2014'])
        """
        # Use default csv_data_dir if base_path is not provided
        if base_path is None:
            base_path = self.csv_data_dir
            
        if not os.path.exists(base_path):
            raise ValueError(f"Path does not exist: {base_path}")
        
        results = {}
        
        # Get all folders to process
        all_folders = sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])
        
        # Iterate through all folders in base path
        for folder_name in tqdm(all_folders, desc="Checking folders", disable=not verbose):
            folder_path = os.path.join(base_path, folder_name)
            
            # Extract year range from folder name (e.g., csv_2013_2014)
            year_match = re.search(r'(\d{4})_(\d{4})', folder_name)
            if not year_match:
                if verbose:
                    print(f"⚠️  Skipping folder '{folder_name}': Cannot extract year range from folder name")
                continue
            
            start_year = int(year_match.group(1))
            end_year = int(year_match.group(2))
            
            if verbose:
                print(f"\n📁 Checking folder: {folder_name} (Expected years: {start_year}-{end_year})")
            
            # Initialize results
            folder_result = {
                'expected_years': (start_year, end_year),
                'total_files': 0,
                'checked_files': 0,
                'valid_files': 0,
                'invalid_files': 0,
                'error_files': 0,
                'invalid_details': [],
                'error_details': []
            }
            
            # Iterate through all CSV files in folder
            csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            folder_result['total_files'] = len(csv_files)
            
            for csv_file in tqdm(csv_files, desc=f"  {folder_name}", leave=False, disable=not verbose):
                csv_path = os.path.join(folder_path, csv_file)
                folder_result['checked_files'] += 1
                
                try:
                    # Read CSV file and check date column
                    with open(csv_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        
                        # Check if date column exists
                        if 'DATE (YYYY-MM-DDTHH:MI:SSZ)' not in reader.fieldnames:
                            folder_result['error_files'] += 1
                            folder_result['error_details'].append(
                                (csv_file, "Date column not found")
                            )
                            continue
                        
                        # Extract all dates and check years
                        dates = []
                        all_valid = True
                        
                        for row in reader:
                            date_str = row.get('DATE (YYYY-MM-DDTHH:MI:SSZ)', '')
                            if date_str:
                                try:
                                    # Parse date format YYYY-MM-DDTHH:MI:SSZ
                                    date_obj = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')
                                    dates.append(date_obj)
                                    
                                    # Check if year is within expected range
                                    if not (start_year <= date_obj.year <= end_year):
                                        all_valid = False
                                except ValueError:
                                    # Date format error
                                    all_valid = False
                        
                        if not dates:
                            folder_result['error_files'] += 1
                            folder_result['error_details'].append(
                                (csv_file, "No valid date data in file")
                            )
                            continue
                        
                        # Determine if file is valid
                        if all_valid:
                            folder_result['valid_files'] += 1
                        else:
                            folder_result['invalid_files'] += 1
                            min_date = min(dates)
                            max_date = max(dates)
                            folder_result['invalid_details'].append(
                                (csv_file, f"{min_date.year}-{max_date.year}")
                            )
                
                except Exception as e:
                    folder_result['error_files'] += 1
                    folder_result['error_details'].append(
                        (csv_file, str(e))
                    )
            
            results[folder_name] = folder_result
            
            # Output statistics
            if verbose:
                print(f"   Total files: {folder_result['total_files']}")
                print(f"   ✅ Valid files: {folder_result['valid_files']}")
                print(f"   ❌ Invalid files: {folder_result['invalid_files']}")
                print(f"   ⚠️  Error files: {folder_result['error_files']}")
                
                if folder_result['invalid_details']:
                    print(f"\n   Invalid file details (year mismatch):")
                    for filename, date_range in folder_result['invalid_details'][:10]:  # Show first 10 only
                        print(f"      - {filename}: Actual year range {date_range}")
                    if len(folder_result['invalid_details']) > 10:
                        print(f"      ... and {len(folder_result['invalid_details']) - 10} more files")
                
                if folder_result['error_details']:
                    print(f"\n   Error file details:")
                    for filename, error in folder_result['error_details'][:5]:  # Show first 5 only
                        print(f"      - {filename}: {error}")
                    if len(folder_result['error_details']) > 5:
                        print(f"      ... and {len(folder_result['error_details']) - 5} more files")
        
        # Output summary
        if verbose and results:
            print("\n" + "="*60)
            print("📊 Overall Statistics:")
            total_folders = len(results)
            total_files = sum(r['total_files'] for r in results.values())
            total_valid = sum(r['valid_files'] for r in results.values())
            total_invalid = sum(r['invalid_files'] for r in results.values())
            total_errors = sum(r['error_files'] for r in results.values())
            
            print(f"   Folders checked: {total_folders}")
            print(f"   Total files: {total_files}")
            print(f"   ✅ Valid files: {total_valid} ({total_valid/total_files*100:.1f}%)")
            print(f"   ❌ Invalid files: {total_invalid} ({total_invalid/total_files*100:.1f}%)")
            print(f"   ⚠️  Error files: {total_errors} ({total_errors/total_files*100:.1f}%)")
            print("="*60)
        
        return results
    
    def convert_csv_to_profile_nc(self, base_path: str = None, output_dir: str = None, 
                                   year_folders: list = None, verbose: bool = True):
        """
        Read all CSV files from year folders and group them by Argo float profile cycles.
        Save each profile as a separate nc file. Data within the same profile share the same timestamp.
            
        Args:
            base_path (str, optional): Base path to CSV folders. If None, uses self.csv_data_dir.
            output_dir (str, optional): Output directory for nc files. If None, creates 'nc_profiles' folder under base_path.
            year_folders (list, optional): List of specific year folder names to process (e.g., ['csv_2013_2014', 'csv_2015_2016']).
                                          If None, processes all year folders.
            verbose (bool, optional): Whether to print detailed information. Defaults to True.
            
        Example:
            >>> processor = ArgoProcessorCSV()
            >>> # Process all year folders
            >>> processor.convert_csv_to_profile_nc()
            >>> # Process specific year folders
            >>> processor.convert_csv_to_profile_nc(year_folders=['csv_2013_2014', 'csv_2015_2016'])
        """
        # Use default path if not specified
        if base_path is None:
            base_path = self.csv_data_dir
            
        if output_dir is None:
            output_dir = os.path.join(self.data_dir, 'nc_profiles')
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
            
        if not os.path.exists(base_path):
            raise ValueError(f"Path does not exist: {base_path}")
            
        # Get all year folders
        all_year_folders = sorted([f for f in os.listdir(base_path) 
                              if os.path.isdir(os.path.join(base_path, f)) and re.search(r'\d{4}_\d{4}', f)])
        
        # Filter year folders based on user input
        if year_folders is not None:
            # Validate specified year folders
            invalid_folders = [f for f in year_folders if f not in all_year_folders]
            if invalid_folders:
                print(f"⚠️  Warning: Following year folders not found and will be skipped: {invalid_folders}")
            
            # Use only valid specified folders
            folders_to_process = [f for f in year_folders if f in all_year_folders]
            
            if not folders_to_process:
                raise ValueError(f"None of the specified year folders exist in {base_path}. Available folders: {all_year_folders}")
        else:
            # Process all year folders
            folders_to_process = all_year_folders
            
        if verbose:
            print(f"Found {len(all_year_folders)} year folders in total")
            print(f"Processing {len(folders_to_process)} year folders: {folders_to_process}")
            print(f"Output directory: {output_dir}")
            
        total_profiles = 0
            
        # Process each year folder
        for year_folder in tqdm(folders_to_process, desc="Processing year folders", disable=not verbose):
            year_folder_path = os.path.join(base_path, year_folder)
                
            # Get all CSV files in this folder
            csv_files = sorted([f for f in os.listdir(year_folder_path) if f.endswith('.csv')])
                
            if verbose:
                print(f"\nProcessing folder: {year_folder} (Total {len(csv_files)} CSV files)")
                
            # Process each CSV file
            for csv_file in tqdm(csv_files, desc=f"  {year_folder}", leave=False, disable=not verbose):
                csv_path = os.path.join(year_folder_path, csv_file)
                    
                try:
                    # Read CSV file
                    df = pd.read_csv(csv_path)
                        
                    # Check if required column exists
                    date_col = 'DATE (YYYY-MM-DDTHH:MI:SSZ)'
                    if date_col not in df.columns:
                        if verbose:
                            print(f"    ⚠️  Skipping {csv_file}: Missing date column")
                        continue
                        
                    # Group by timestamp (same profile shares the same timestamp)
                    profiles = df.groupby(date_col)
                        
                    # Create nc file for each profile
                    for profile_time, profile_data in profiles:
                        # Extract year from profile timestamp
                        try:
                            profile_datetime = datetime.strptime(profile_time, '%Y-%m-%dT%H:%M:%SZ')
                            profile_year = str(profile_datetime.year)
                        except ValueError:
                            if verbose:
                                print(f"    ⚠️  Invalid timestamp format: {profile_time}, skipping...")
                            continue
                        
                        # Create output subdirectory for each year (one folder per year)
                        single_year_output_dir = os.path.join(output_dir, profile_year)
                        os.makedirs(single_year_output_dir, exist_ok=True)
                        
                        # Create safe filename (remove special characters)
                        safe_time = profile_time.replace(':', '-').replace('T', '_').replace('Z', '')
                        base_name = os.path.splitext(csv_file)[0]
                        nc_filename = f"profile_{safe_time}_{base_name}.nc"
                        nc_path = os.path.join(single_year_output_dir, nc_filename)
                            
                        # Convert DataFrame to xarray Dataset
                        ds = self._create_profile_dataset(profile_data, profile_time)
                            
                        # Save as nc file
                        ds.to_netcdf(nc_path)
                        ds.close()
                            
                        total_profiles += 1
                    
                except Exception as e:
                    if verbose:
                        print(f"    ❌ Error processing file {csv_file}: {str(e)}")
                    continue
            
        if verbose:
            print(f"\n✅ Completed! Generated {total_profiles} profile nc files")
            print(f"Output directory: {output_dir}")
            
        return total_profiles
    
    def _create_profile_dataset(self, profile_data: pd.DataFrame, profile_time: str) -> xr.Dataset:
        """
        Convert DataFrame data from a single profile to xarray Dataset.
        Profile-level metadata (coordinates that should be constant) are stored as scalars.
        Depth-varying measurements are stored as arrays.
        
        Args:
            profile_data (pd.DataFrame): Data from a single profile
            profile_time (str): Timestamp of the profile
        
        Returns:
            xr.Dataset: Converted xarray Dataset
        """
        # Create dimensions for depth-varying data
        n_obs = len(profile_data)
        
        # Create data variable dictionary
        data_vars = {}
        coords = {}
        
        # Add observation dimension (depth levels)
        coords['obs'] = np.arange(n_obs)
        
        # Define profile-level columns that should be constant across a profile
        profile_level_cols = [
            'PLATFORM_CODE',
            'DATE (YYYY-MM-DDTHH:MI:SSZ)',
            'DATE_QC',
            'LATITUDE (degree_north)',
            'LONGITUDE (degree_east)',
            'POSITION_QC'
        ]
        
        # Process profile-level metadata
        for col in profile_data.columns:
            if col in profile_level_cols:
                # Check if values are consistent within the profile
                unique_values = profile_data[col].unique()
                
                if len(unique_values) > 1:
                    # Warning: inconsistent values found
                    print(f"    ⚠️  Warning: Inconsistent values in {col}: {unique_values}")
                    # Use the most common value
                    value = profile_data[col].mode()[0] if not profile_data[col].mode().empty else unique_values[0]
                else:
                    value = unique_values[0]
                
                # Create safe variable name
                safe_var_name = col.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
                
                # Try to convert to numeric if possible
                try:
                    numeric_value = pd.to_numeric(value)
                    data_vars[safe_var_name] = numeric_value
                except (ValueError, TypeError):
                    # Keep as string/original type
                    data_vars[safe_var_name] = value
        
        # Process depth-varying measurements
        for col in profile_data.columns:
            if col not in profile_level_cols:
                # Try to convert to numeric type
                try:
                    values = pd.to_numeric(profile_data[col], errors='coerce').values
                except:
                    # Keep as string if cannot convert to numeric
                    values = profile_data[col].values
                
                # Create safe variable name (replace special characters)
                safe_var_name = col.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
                data_vars[safe_var_name] = (['obs'], values)
        
        # Create Dataset
        ds = xr.Dataset(data_vars, coords=coords)
        
        # Add global attributes
        ds.attrs['profile_time'] = profile_time
        ds.attrs['creation_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ds.attrs['description'] = 'Argo float profile data extracted from CSV'
        ds.attrs['n_levels'] = n_obs
        
        return ds
    
    def visualize_random_nc_profiles(self, nc_dir: str = None, n_samples: int = 5,
                                     output_dir: str = None, year_filter: str = None,
                                     verbose: bool = True) -> None:
        """
        Randomly sample some processed nc profile files and visualize them.
        
        Args:
            nc_dir (str, optional): Base directory of nc files. If None, use self.data_dir/nc_profiles.
            n_samples (int, optional): Number of files to randomly sample. Default is 5.
            output_dir (str, optional): Output directory for plots. If None, use './output/plot/argo_profiles'.
            year_filter (str, optional): Specify year subdirectory to sample from (e.g., '2013', '2014').
                                        If None, randomly sample from all years.
            verbose (bool, optional): Whether to print detailed information. Default is True.
        
        Returns:
            None: Generate plots and save to specified directory
        
        Example:
            >>> processor = ArgoProcessorCSV()
            >>> # Randomly sample 10 nc files and visualize
            >>> processor.visualize_random_nc_profiles(n_samples=10)
            >>> # Only sample from 2013 nc files
            >>> processor.visualize_random_nc_profiles(n_samples=5, year_filter='2013')
        """
        # Set default directories
        if nc_dir is None:
            nc_dir = os.path.join(self.data_dir, 'nc_profiles')
        
        if output_dir is None:
            output_dir = './output/plot/argo_profiles'
        
        # Check if nc directory exists
        if not os.path.exists(nc_dir):
            raise ValueError(f"NC directory does not exist: {nc_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        if verbose:
            print(f"Scanning nc files in: {nc_dir}")
        
        # Collect all nc files
        all_nc_files = []
        
        # Traverse year subdirectories
        year_folders = sorted([f for f in os.listdir(nc_dir) 
                              if os.path.isdir(os.path.join(nc_dir, f))])
        
        if verbose:
            print(f"Found {len(year_folders)} year folders: {year_folders}")
        
        # If year_filter is specified, only process that year
        if year_filter is not None:
            if year_filter not in year_folders:
                raise ValueError(f"Year folder '{year_filter}' not found. Available folders: {year_folders}")
            year_folders = [year_filter]
            if verbose:
                print(f"Filtering by year: {year_filter}")
        
        # Collect all nc files that meet the criteria
        for year_folder in year_folders:
            year_path = os.path.join(nc_dir, year_folder)
            nc_files = glob.glob(os.path.join(year_path, '*.nc'))
            all_nc_files.extend(nc_files)
        
        if len(all_nc_files) == 0:
            print(f"No nc files found in {nc_dir}")
            return
        
        if verbose:
            print(f"Total nc files found: {len(all_nc_files)}")
        
        # Randomly sample n_samples files
        n_samples = min(n_samples, len(all_nc_files))
        sampled_files = random.sample(all_nc_files, n_samples)
        
        if verbose:
            print(f"Randomly sampling {n_samples} nc files for visualization")
            print("=" * 80)
        
        # Visualize each sampled file
        for i, nc_file in enumerate(sampled_files, 1):
            if verbose:
                print(f"\n[{i}/{n_samples}] Processing: {os.path.basename(nc_file)}")
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(nc_file))[0]
            output_path = os.path.join(output_dir, f"{base_name}.png")
            
            try:
                # Load nc file as xarray Dataset
                ds = xr.open_dataset(nc_file)
                
                # Call visualization function from plot script
                plot_argo_nc_profiles(ds, output_path=output_path)
                
                # Close dataset
                ds.close()
            except Exception as e:
                if verbose:
                    print(f"  ❌ Error visualizing {nc_file}: {str(e)}")
                continue
        
        if verbose:
            print("\n" + "=" * 80)
            print(f"✅ Completed! Generated {n_samples} visualization plots")
            print(f"Output directory: {output_dir}")
    
    def analyze_daily_profile_counts(self, nc_dir: str = None, 
                                     output_path: str = None,
                                     year_filter: list = None,
                                     verbose: bool = True,
                                     use_cache: bool = True) -> dict:
        """
        Analyze and visualize the distribution of Argo profile counts per day.
        Scans all nc profile files, counts profiles for each day, and generates a time series plot.
        
        Args:
            nc_dir (str, optional): Base directory of nc files. If None, uses self.data_dir/nc_profiles.
            output_path (str, optional): Output path for the plot. If None, uses './output/plot/argo_daily_profile_counts.png'.
            year_filter (list, optional): List of year strings to filter (e.g., ['2013', '2014']). 
                                         If None, processes all years.
            verbose (bool, optional): Whether to print detailed information. Default is True.
            use_cache (bool, optional): Whether to use cached results. Default is True.
        
        Returns:
            dict: Dictionary with date strings as keys and profile counts as values.
                 Format: {'YYYY-MM-DD': count, ...}
        
        Example:
            >>> processor = ArgoProcessorCSV()
            >>> # Analyze all years
            >>> daily_counts = processor.analyze_daily_profile_counts()
            >>> # Analyze only 2013 and 2014
            >>> daily_counts = processor.analyze_daily_profile_counts(year_filter=['2013', '2014'])
        """
        # Set default directories
        if nc_dir is None:
            nc_dir = os.path.join(self.data_dir, 'nc_profiles')
        
        if output_path is None:
            output_path = './output/plot/data/argo/argo_daily_profile_counts.png'
        
        # Check if nc directory exists
        if not os.path.exists(nc_dir):
            raise ValueError(f"NC directory does not exist: {nc_dir}")
        
        # Generate cache file path based on nc_dir and year_filter
        cache_key = f"{nc_dir}_{sorted(year_filter) if year_filter else 'all'}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_dir = os.path.join(self.data_dir, '.cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'daily_counts_{cache_hash}.pkl')
        
        # Try to load from cache
        if use_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    daily_counts_dict = pickle.load(f)
                if verbose:
                    print(f"✅ Loaded cached results from: {cache_file}")
                    print(f"   Total unique days: {len(daily_counts_dict)}")
                    if daily_counts_dict:
                        print(f"   Date range: {min(daily_counts_dict.keys())} to {max(daily_counts_dict.keys())}")
                
                # Generate plot using cached data
                if daily_counts_dict:
                    if verbose:
                        print(f"\nGenerating plot...")
                    plot_argo_daily_profile_counts(daily_counts_dict, output_path=output_path)
                
                return daily_counts_dict


            except Exception as e:
                if verbose:
                    print(f"⚠️  Failed to load cache: {str(e)}. Will recompute.")
        
        if verbose:
            print(f"Scanning nc files in: {nc_dir}")
        
        # Get all year folders (excluding 2025)
        year_folders = sorted([f for f in os.listdir(nc_dir) 
                              if os.path.isdir(os.path.join(nc_dir, f)) and f != '2025'])
        
        if verbose:
            print(f"Found {len(year_folders)} year folders (excluding 2025): {year_folders}")
        
        # Apply year filter if specified
        if year_filter is not None:
            invalid_years = [y for y in year_filter if y not in year_folders]
            if invalid_years:
                print(f"⚠️  Warning: Following years not found and will be skipped: {invalid_years}")
            year_folders = [y for y in year_filter if y in year_folders]
            if not year_folders:
                raise ValueError(f"None of the specified years exist in {nc_dir}. Available years: {year_folders}")
            if verbose:
                print(f"Filtering by years: {year_folders}")
        
        # Dictionary to store daily profile counts
        daily_counts = defaultdict(int)
        
        # Process each year folder
        total_profiles = 0
        for year_folder in tqdm(year_folders, desc="Processing year folders", disable=not verbose):
            year_path = os.path.join(nc_dir, year_folder)
            nc_files = glob.glob(os.path.join(year_path, '*.nc'))
            
            if verbose:
                print(f"\nProcessing year {year_folder}: {len(nc_files)} nc files")
            
            # Process each nc file
            for nc_file in tqdm(nc_files, desc=f"  Year {year_folder}", leave=False, disable=not verbose):
                try:
                    # Load dataset to extract profile time
                    ds = xr.open_dataset(nc_file)
                    
                    # Extract profile time from attributes
                    profile_time = ds.attrs.get('profile_time', None)
                    
                    if profile_time:
                        # Parse datetime and extract date (YYYY-MM-DD)
                        try:
                            profile_datetime = datetime.strptime(profile_time, '%Y-%m-%dT%H:%M:%SZ')
                            # Skip 2025 data
                            if profile_datetime.year == 2025:
                                continue
                            date_str = profile_datetime.strftime('%Y-%m-%d')
                            daily_counts[date_str] += 1
                            total_profiles += 1
                        except ValueError as ve:
                            if verbose:
                                print(f"    ⚠️  Invalid time format in {os.path.basename(nc_file)}: {profile_time}")
                    else:
                        if verbose:
                            print(f"    ⚠️  No profile_time attribute in {os.path.basename(nc_file)}")
                    
                    ds.close()
                    
                except Exception as e:
                    if verbose:
                        print(f"    ❌ Error reading {os.path.basename(nc_file)}: {str(e)}")
                    continue
        
        if verbose:
            print(f"\n📊 Statistics:")
            print(f"   Total profiles processed: {total_profiles}")
            print(f"   Total unique days: {len(daily_counts)}")
            if daily_counts:
                print(f"   Date range: {min(daily_counts.keys())} to {max(daily_counts.keys())}")
                print(f"   Average profiles per day: {total_profiles/len(daily_counts):.2f}")
        
        # Convert daily_counts to regular dict for return
        daily_counts_dict = dict(daily_counts)
        
        # Save to cache
        if use_cache:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(daily_counts_dict, f)
                if verbose:
                    print(f"\n💾 Cached results saved to: {cache_file}")
            except Exception as e:
                if verbose:
                    print(f"⚠️  Failed to save cache: {str(e)}")
        
        # Generate plot using the plot module function
        if daily_counts_dict:
            if verbose:
                print(f"\nGenerating plot...")
            plot_argo_daily_profile_counts(daily_counts_dict, output_path=output_path)
        else:
            print("Warning: No profile data to plot")
        
        return daily_counts_dict

    def plot_profile_length_distribution(self, base_path: str = None, year_folders: list = None, 
                                         save_path: str = None, verbose: bool = True, 
                                         use_cache: bool = True):
        """
        Calculate and plot the distribution of Argo profile lengths (number of sampling points).
        
        Args:
            base_path (str, optional): Base path to CSV folders. If None, uses self.csv_data_dir.
            year_folders (list, optional): List of specific year folder names to process.
            save_path (str, optional): Path to save the plot. If None, saves to default location.
            verbose (bool, optional): Whether to print progress.
            use_cache (bool, optional): Whether to use cached results. Default is True.
        """
        import matplotlib.pyplot as plt
        
        # Use default path if not specified
        if base_path is None:
            base_path = self.csv_data_dir
            
        if not os.path.exists(base_path):
            raise ValueError(f"Path does not exist: {base_path}")
        
        # Get all year folders
        all_year_folders = sorted([f for f in os.listdir(base_path) 
                                  if os.path.isdir(os.path.join(base_path, f)) and re.search(r'\d{4}_\d{4}', f)])
        
        # Filter year folders based on user input
        if year_folders is not None:
            folders_to_process = [f for f in year_folders if f in all_year_folders]
            if not folders_to_process:
                raise ValueError(f"None of the specified year folders exist in {base_path}.")
        else:
            # Default: limit to 2012-2023
            folders_to_process = [f for f in all_year_folders 
                                  if int(re.search(r'(\d{4})_\d{4}', f).group(1)) >= 2012 
                                  and int(re.search(r'\d{4}_(\d{4})', f).group(1)) <= 2023]
        
        # Generate cache file path based on base_path and year_folders
        cache_key = f"{base_path}_{sorted(folders_to_process)}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_dir = os.path.join(self.data_dir, '.cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'profile_lengths_{cache_hash}.pkl')
        
        # Try to load from cache
        profile_lengths = []
        if use_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    profile_lengths = pickle.load(f)
                if verbose:
                    print(f"✅ Loaded cached profile lengths from: {cache_file}")
                    print(f"   Total profiles: {len(profile_lengths)}")
            except Exception as e:
                if verbose:
                    print(f"⚠️  Failed to load cache: {e}")
                profile_lengths = []
        
        # Process data if not loaded from cache
        if not profile_lengths:
            if verbose:
                print(f"Processing {len(folders_to_process)} year folders to calculate profile lengths...")
            
            # Process each year folder
            for year_folder in tqdm(folders_to_process, desc="Collecting profile lengths", disable=not verbose):
                year_folder_path = os.path.join(base_path, year_folder)
                csv_files = sorted([f for f in os.listdir(year_folder_path) if f.endswith('.csv')])
                
                for csv_file in tqdm(csv_files, desc=f"    Processing CSV files", leave=False, disable=not verbose):
                    csv_path = os.path.join(year_folder_path, csv_file)
                    try:
                        # Read only the date column to save memory
                        df = pd.read_csv(csv_path, usecols=['DATE (YYYY-MM-DDTHH:MI:SSZ)'])
                        
                        # Count occurrences of each unique timestamp (profile length)
                        counts = df['DATE (YYYY-MM-DDTHH:MI:SSZ)'].value_counts()
                        profile_lengths.extend(counts.values)
                        
                        # Print profiles with length > 5000
                        for timestamp, length in counts.items():
                            if length > 5000:
                                print(f"⚠️  Long profile detected: Float {csv_file}, Float time {timestamp}, Length {length}")
                    except Exception as e:
                        if verbose:
                            print(f"Error processing {csv_file}: {e}")
                        continue
            
            # Save to cache
            if use_cache and profile_lengths:
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(profile_lengths, f)
                    if verbose:
                        print(f"\n💾 Cached profile lengths saved to: {cache_file}")
                except Exception as e:
                    if verbose:
                        print(f"⚠️  Failed to save cache: {e}")
        
        if not profile_lengths:
            print("No profiles found.")
            return

        # Plotting - JASA style
        import matplotlib
        # Set Times New Roman font globally
        matplotlib.rcParams['font.family'] = 'serif'
        matplotlib.rcParams['font.serif'] = ['Times New Roman']
        matplotlib.rcParams['axes.labelsize'] = 12
        matplotlib.rcParams['axes.titlesize'] = 13
        matplotlib.rcParams['xtick.labelsize'] = 10
        matplotlib.rcParams['ytick.labelsize'] = 10
        matplotlib.rcParams['legend.fontsize'] = 10
        
        fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=900)
        
        # JASA style histogram with steelblue color
        n, bins, patches = ax.hist(profile_lengths, bins=50, color='#4682B4', 
                                   edgecolor='black', linewidth=0.5, alpha=0.85)
        ax.set_yscale('log')
        
        # Labels with Times New Roman
        ax.set_title('Distribution of Argo Profile Lengths', fontweight='normal', pad=10)
        ax.set_xlabel('Number of Sampling Points', labelpad=8)
        ax.set_ylabel('Frequency (Log Scale)', labelpad=8)
        
        # Clean grid - only horizontal lines, lighter
        ax.grid(True, axis='y', linestyle='-', linewidth=0.5, alpha=0.4, color='gray')
        ax.set_axisbelow(True)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        
        # Add stats to plot with refined positioning
        mean_len = np.mean(profile_lengths)
        median_len = np.median(profile_lengths)
        min_len = np.min(profile_lengths)
        max_len = np.max(profile_lengths)
        
        # Use more subtle colors for reference lines
        ax.axvline(mean_len, color='#8B0000', linestyle='--', linewidth=1.2, 
                   label=f'Mean: {mean_len:.1f}', alpha=0.9)
        ax.axvline(median_len, color='#006400', linestyle='--', linewidth=1.2, 
                   label=f'Median: {median_len:.1f}', alpha=0.9)
        ax.axvline(min_len, color='#4169E1', linestyle=':', linewidth=1.0, 
                   label=f'Min: {min_len}', alpha=0.8)
        ax.axvline(max_len, color='#8B008B', linestyle=':', linewidth=1.0, 
                   label=f'Max: {max_len}', alpha=0.8)
        
        # Legend with better positioning
        ax.legend(loc='upper right', frameon=True, fancybox=False, 
                  edgecolor='gray', framealpha=0.95)
        
        plt.tight_layout()
        
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path)
            if verbose:
                print(f"Plot saved to {save_path}")
        else:
            default_save_path = os.path.join("./output/plot/data/argo", "profile_length_distribution.png")
            os.makedirs(os.path.dirname(default_save_path), exist_ok=True)
            plt.savefig(default_save_path)
            if verbose:
                print(f"Plot saved to {default_save_path}")
        
        plt.close()

    def process_argo_to_netcdf(self, output_file='argo_profiles.nc', year_folders=None, verbose=True):
        """
        Process Argo data from all CSV files and generate a NetCDF file.
        
        Requirements:
        1. Extract Argo profiles (one profile per time point)
        2. Interpolate to 0-2000m (1024 points)
        3. Handle missing values and depths
        4. Synthesize into (n, 1024, 2) matrix
        5. Generate NetCDF with metadata
        
        Args:
            output_file (str): Path to the output NetCDF file.
            year_folders (list, optional): List of specific year folder names to process.
            verbose (bool): Whether to print detailed information.
            
        Returns:
            str: Path to the generated NetCDF file.
        """
        if verbose:
            print(f"Starting Argo CSV processing to NetCDF...")
            print(f"Target output: {output_file}")

        # Target depth levels
        target_depths = np.linspace(0, 2000, 1024)
        
        # Containers for data
        all_temps = []
        all_psals = []
        all_times = []
        all_lats = []
        all_lons = []
        seen_times = set()
        total_profiles_duplicates_removed = 0
        
        # Get all year folders
        if not os.path.exists(self.csv_data_dir):
            raise FileNotFoundError(f"CSV data directory not found: {self.csv_data_dir}")
            
        all_available_folders = sorted([f for f in os.listdir(self.csv_data_dir) 
                             if os.path.isdir(os.path.join(self.csv_data_dir, f))])
        
        if year_folders is None:
            folders_to_process = all_available_folders
        else:
            folders_to_process = [f for f in year_folders if f in all_available_folders]
            if not folders_to_process:
                 raise ValueError(f"None of the specified year folders exist in {self.csv_data_dir}")
        
        total_profiles_processed = 0
        total_profiles_skipped = 0
        total_profiles_seen = 0
        error_count = 0
        
        for year_folder in tqdm(folders_to_process, desc="Processing folders", disable=not verbose):
            year_folder_path = os.path.join(self.csv_data_dir, year_folder)
            csv_files = sorted([f for f in os.listdir(year_folder_path) if f.endswith('.csv')])
            
            for csv_file in tqdm(csv_files, desc=f"  {year_folder}", leave=False, disable=not verbose):
                csv_path = os.path.join(year_folder_path, csv_file)
                
                try:
                    # Read CSV
                    df = pd.read_csv(csv_path)
                    
                    # Ensure required columns exist
                    req_cols = ['DATE (YYYY-MM-DDTHH:MI:SSZ)', 'PRES_ADJUSTED (decibar)', 
                                'TEMP_ADJUSTED (degree_Celsius)', 'PSAL_ADJUSTED (psu)',
                                'LATITUDE (degree_north)', 'LONGITUDE (degree_east)',
                                'PRES_ADJUSTED_QC', 'TEMP_ADJUSTED_QC', 'PSAL_ADJUSTED_QC',
                                'DATE_QC', 'POSITION_QC']
                    
                    # Check columns existence
                    if not all(col in df.columns for col in req_cols):
                        continue

                    raw_profile_times = pd.to_datetime(
                        df['DATE (YYYY-MM-DDTHH:MI:SSZ)'],
                        format='%Y-%m-%dT%H:%M:%SZ',
                        errors='coerce',
                    )
                    raw_profile_times = pd.Index(raw_profile_times.dropna().unique())
                    total_profiles_seen += len(raw_profile_times)

                    # Filter based on QC flags
                    # 1. If DATE_QC, POSITION_QC, or PRES_ADJUSTED_QC is not 1, discard the row
                    # Use pd.to_numeric to handle '1', 1, 1.0 correctly
                    good_date = pd.to_numeric(df['DATE_QC'], errors='coerce') == 1
                    good_pos = pd.to_numeric(df['POSITION_QC'], errors='coerce') == 1
                    good_pres = pd.to_numeric(df['PRES_ADJUSTED_QC'], errors='coerce') == 1
                    
                    df = df[good_date & good_pos & good_pres]

                    # 2. If TEMP_ADJUSTED_QC is not 1, discard temperature data
                    mask_temp_bad = pd.to_numeric(df['TEMP_ADJUSTED_QC'], errors='coerce') != 1
                    df.loc[mask_temp_bad, 'TEMP_ADJUSTED (degree_Celsius)'] = np.nan

                    # 3. If PSAL_ADJUSTED_QC is not 1, discard salinity data
                    mask_psal_bad = pd.to_numeric(df['PSAL_ADJUSTED_QC'], errors='coerce') != 1
                    df.loc[mask_psal_bad, 'PSAL_ADJUSTED (psu)'] = np.nan

                    # 4. If both temperature and salinity data are invalid (NaN), discard the row
                    df = df.dropna(subset=['TEMP_ADJUSTED (degree_Celsius)', 'PSAL_ADJUSTED (psu)'], how='all')
                        
                    # Group by Date (Profile)
                    df['date_obj'] = pd.to_datetime(
                        df['DATE (YYYY-MM-DDTHH:MI:SSZ)'],
                        format='%Y-%m-%dT%H:%M:%SZ',
                        errors='coerce',
                    )

                    filtered_profile_times = pd.Index(df['date_obj'].dropna().unique())
                    total_profiles_skipped += len(raw_profile_times.difference(filtered_profile_times))
                    
                    for timestamp, group in df.groupby('date_obj'):
                        # 2. Extract Data
                        group_clean = group.dropna(subset=['PRES_ADJUSTED (decibar)']).sort_values('PRES_ADJUSTED (decibar)')
                        
                        if group_clean.empty:
                            total_profiles_skipped += 1
                            continue
                            
                        pres = group_clean['PRES_ADJUSTED (decibar)'].values
                        
                        # 3. Interpolate
                        # Temperature
                        if not group_clean['TEMP_ADJUSTED (degree_Celsius)'].isna().all():
                            temp = group_clean['TEMP_ADJUSTED (degree_Celsius)'].values
                            # Interpolate (left=NaN, right=NaN)
                            temp_interp = np.interp(target_depths, pres, temp, left=np.nan, right=np.nan)
                        else:
                            temp_interp = np.full(1024, np.nan)
                            
                        # Salinity
                        if not group_clean['PSAL_ADJUSTED (psu)'].isna().all():
                            psal = group_clean['PSAL_ADJUSTED (psu)'].values
                            psal_interp = np.interp(target_depths, pres, psal, left=np.nan, right=np.nan)
                        else:
                            psal_interp = np.full(1024, np.nan)
                            
                        # Check if all data is NaN
                        if np.isnan(temp_interp).all() and np.isnan(psal_interp).all():
                            total_profiles_skipped += 1
                            continue
                        
                        if timestamp in seen_times:
                            total_profiles_duplicates_removed += 1
                            continue
                        seen_times.add(timestamp)

                        # 1. Extract Metadata (Only if valid data exists)
                        all_times.append(timestamp)
                        
                        # Lat/Lon (Mode)
                        lat_mode = group['LATITUDE (degree_north)'].mode().iloc[0]
                        lon_mode = group['LONGITUDE (degree_east)'].mode().iloc[0]
                        all_lats.append(lat_mode)
                        all_lons.append(lon_mode)

                        all_temps.append(temp_interp)
                        all_psals.append(psal_interp)
                        total_profiles_processed += 1
                        
                except Exception as e:
                    error_count += 1
                    continue

        if verbose:
            print(f"Processed {total_profiles_processed} profiles.")
            print(f"Skipped {total_profiles_skipped} profiles.")
            print(f"Removed {total_profiles_duplicates_removed} duplicate profiles.")
            print(f"Encountered {error_count} errors.")
            print(f"Constructing NetCDF file...")

        # Convert to numpy arrays
        temps_array = np.array(all_temps, dtype=np.float32)
        psals_array = np.array(all_psals, dtype=np.float32)
        # Convert times to datetime64[ns]
        times_array = pd.to_datetime(all_times).values
        lats_array = np.array(all_lats, dtype=np.float32)
        lons_array = np.array(all_lons, dtype=np.float32)
        
        # Create Xarray Dataset
        ds = xr.Dataset(
            data_vars={
                'temperature': (['profile', 'depth'], temps_array, {
                    'units': 'degree_Celsius',
                    'long_name': 'Sea Water Temperature',
                    'standard_name': 'sea_water_temperature'
                }),
                'salinity': (['profile', 'depth'], psals_array, {
                    'units': 'psu',
                    'long_name': 'Sea Water Salinity',
                    'standard_name': 'sea_water_salinity'
                })
            },
            coords={
                'time': (['profile'], times_array, {
                    'long_name': 'Time',
                    'standard_name': 'time'
                }),
                'latitude': (['profile'], lats_array, {
                    'units': 'degree_north',
                    'long_name': 'Latitude',
                    'standard_name': 'latitude'
                }),
                'longitude': (['profile'], lons_array, {
                    'units': 'degree_east',
                    'long_name': 'Longitude',
                    'standard_name': 'longitude'
                }),
                'depth': (['depth'], target_depths, {
                    'units': 'decibar',
                    'long_name': 'Pressure',
                    'standard_name': 'sea_water_pressure',
                    'positive': 'down'
                })
            },
            attrs={
                'title': 'Argo Float Profiles',
                'description': 'Interpolated Argo profiles from CSV data',
                'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'source': 'Argo CSV Data',
                'total_profiles_seen': int(total_profiles_seen),
                'total_profiles_processed': int(total_profiles_processed),
                'total_profiles_skipped': int(total_profiles_skipped),
                'total_profiles_duplicates_removed': int(total_profiles_duplicates_removed),
                'Conventions': 'CF-1.8'
            }
        )
        
        # Save to NetCDF
        try:
            out_dir = os.path.dirname(output_file)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir)
                
            ds.to_netcdf(output_file)
            if verbose:
                print(f"✅ Successfully created NetCDF file: {output_file}")
                print(f"   Dimensions: {ds.dims}")
                print(f"   Variables: {list(ds.data_vars)}")
        except Exception as e:
            print(f"❌ Error saving NetCDF file: {e}")
            raise

        return output_file

    def split_netcdf_by_date(self, nc_file: str, output_dir: str, filename_prefix: str = 'argo_profiles', verbose: bool = True):
        if not os.path.exists(nc_file):
            raise FileNotFoundError(f"NetCDF file not found: {nc_file}")

        os.makedirs(output_dir, exist_ok=True)

        ds = xr.open_dataset(nc_file)
        if 'time' not in ds.coords:
            ds.close()
            raise ValueError("NetCDF file does not contain 'time' coordinate")

        times = pd.to_datetime(ds['time'].values)
        valid_mask = ~pd.isna(times)
        if not valid_mask.any():
            ds.close()
            raise ValueError("NetCDF file contains no valid time values")

        date_strings = pd.Series(times[valid_mask]).dt.strftime('%Y-%m-%d').values
        indices = np.where(valid_mask)[0]

        date_to_indices = defaultdict(list)
        for idx, date_str in zip(indices, date_strings):
            date_to_indices[date_str].append(idx)

        output_files = []
        date_keys = sorted(date_to_indices.keys())
        date_iter = tqdm(date_keys, desc="Writing daily NetCDF files", disable=not verbose)
        for date_str in date_iter:
            sel_indices = date_to_indices[date_str]
            daily_ds = ds.isel(profile=sel_indices)
            output_file = os.path.join(output_dir, f"{filename_prefix}_{date_str}.nc")
            daily_ds.to_netcdf(output_file)
            output_files.append(output_file)
            if verbose:
                date_iter.set_postfix(profiles=len(sel_indices))

        ds.close()
        if verbose:
            print(f"Total daily files created: {len(output_files)}")

        return output_files

    def analyze_netcdf_daily_profile_counts(self, nc_file: str = './data/argo/argo_profiles.nc', 
                                            output_path: str = None,
                                            figsize: tuple = (12, 5),
                                            dpi: int = 600,
                                            verbose: bool = True) -> dict:
        """
        Analyze and visualize the distribution of Argo profile counts per day from a consolidated NetCDF file.
        
        Args:
            nc_file (str): Path to the NetCDF file containing Argo profiles.
            output_path (str, optional): Output path for the plot. If None, uses default path.
            figsize (tuple): Figure size (width, height) in inches. Default: (12, 5)
            dpi (int): Resolution of the output image in dots per inch. Default: 600
            verbose (bool): Whether to print detailed information.
            
        Returns:
            dict: Dictionary with date strings as keys and profile counts as values.
        """
        # Set default output path
        if output_path is None:
            output_path = './output/plot/data/argo/argo_daily_profile_counts_from_nc.png'
            
        if not os.path.exists(nc_file):
            raise FileNotFoundError(f"NetCDF file not found: {nc_file}")
            
        if verbose:
            print(f"Reading NetCDF file: {nc_file}")
            
        try:
            ds = xr.open_dataset(nc_file)
            
            # Check if 'time' coordinate exists
            if 'time' not in ds.coords:
                raise ValueError("NetCDF file does not contain 'time' coordinate")
                
            times = ds['time'].values
            ds.close()
            
            # Count profiles per day
            daily_counts = defaultdict(int)
            total_profiles = 0
            
            for t in times:
                # Convert numpy.datetime64 to datetime
                ts = pd.Timestamp(t)
                if pd.isna(ts):
                    continue
                    
                # Only keep data from 2012-2023
                if ts.year < 2012 or ts.year > 2023:
                    continue
                    
                date_str = ts.strftime('%Y-%m-%d')
                daily_counts[date_str] += 1
                total_profiles += 1
                
            if verbose:
                print(f"\\n📊 Statistics:")
                print(f"   Total profiles processed: {total_profiles}")
                print(f"   Total unique days: {len(daily_counts)}")
                if daily_counts:
                    print(f"   Date range: {min(daily_counts.keys())} to {max(daily_counts.keys())}")
                    
            # Convert to regular dict
            daily_counts_dict = dict(daily_counts)

            if daily_counts_dict and verbose:
                low_count_days = sorted(
                    ((date_str, count) for date_str, count in daily_counts_dict.items() if count < 250),
                    key=lambda x: x[0],
                )
                if low_count_days:
                    print(f"\nDays with counts < 250: {len(low_count_days)}")
                    for date_str, count in low_count_days:
                        print(f"   {date_str}: {count}")
            
            # Generate plot
            if daily_counts_dict:
                if verbose:
                    print(f"\nGenerating plot...")
                plot_argo_daily_profile_counts(daily_counts_dict, output_path=output_path, figsize=figsize, dpi=dpi)
            else:
                print("Warning: No profile data to plot")
                
            return daily_counts_dict
            
        except Exception as e:
            if verbose:
                print(f"❌ Error processing NetCDF file: {str(e)}")
            raise

    def analyze_netcdf_spatial_distribution(self, nc_file: str = './data/argo/argo_profiles.nc',
                                            output_path: str = None,
                                            year_range: tuple = (2012, 2023),
                                            figsize: tuple = (10, 6),
                                            dpi: int = 300,
                                            verbose: bool = True) -> dict:
        """
        Analyze and visualize the spatial distribution of Argo profiles from a consolidated NetCDF file.
        Generates a single plot for the entire period.
        
        Args:
            nc_file (str): Path to the NetCDF file containing Argo profiles.
            output_path (str, optional): Output path for the plot. If None, uses default path.
            year_range (tuple): Range of years to analyze (start_year, end_year). Default: (2012, 2023)
            figsize (tuple): Figure size (width, height) in inches. Default: (10, 6)
            dpi (int): Resolution of the output image in dots per inch. Default: 300
            verbose (bool): Whether to print detailed information.
            
        Returns:
            dict: Dictionary containing spatial data with keys:
                  'latitudes': array, 'longitudes': array, 'count': int
        """
        # Set default output path
        if output_path is None:
            output_path = './output/plot/data/argo/argo_spatial_distribution.png'
        
        if not os.path.exists(nc_file):
            raise FileNotFoundError(f"NetCDF file not found: {nc_file}")
        
        if verbose:
            print(f"Reading NetCDF file: {nc_file}")
        
        try:
            ds = xr.open_dataset(nc_file)
            
            # Check required coordinates (case-insensitive)
            required_coords = ['latitude', 'longitude', 'time']
            available_vars = list(ds.coords) + list(ds.variables)
            var_lower_map = {v.lower(): v for v in available_vars}
            
            missing_coords = [c for c in required_coords if c not in var_lower_map]
            if missing_coords:
                raise ValueError(f"NetCDF file missing required variables: {missing_coords}")
            
            # Extract data using actual variable names
            lat_var = var_lower_map['latitude']
            lon_var = var_lower_map['longitude']
            time_var = var_lower_map['time']
            
            lats = ds[lat_var].values
            lons = ds[lon_var].values
            times = ds[time_var].values
            ds.close()
            
            # Filter by year range
            start_year, end_year = year_range
            valid_indices = []
            
            for i, t in enumerate(times):
                ts = pd.Timestamp(t)
                if pd.isna(ts):
                    continue
                year = ts.year
                if start_year <= year <= end_year:
                    valid_indices.append(i)
            
            if verbose:
                print(f"\n📊 Statistics:")
                print(f"   Total profiles in file: {len(times)}")
                print(f"   Profiles in year range {start_year}-{end_year}: {len(valid_indices)}")
            
            if not valid_indices:
                print("Warning: No profiles found in the specified year range")
                return {'latitudes': np.array([]), 'longitudes': np.array([]), 'count': 0}
            
            # Prepare data
            result = {
                'latitudes': lats[valid_indices],
                'longitudes': lons[valid_indices],
                'count': len(valid_indices)
            }
            
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Generate single spatial distribution plot
            if verbose:
                print(f"\nGenerating spatial distribution plot...")
            plot_argo_spatial_distribution(
                latitudes=result['latitudes'],
                longitudes=result['longitudes'],
                output_path=output_path,
                title=f'Argo Profiles Spatial Distribution',
                year=None,
                figsize=figsize,
                dpi=dpi
            )
            
            if verbose:
                print(f"\n✅ Spatial distribution plot saved to: {output_path}")
            
            return result
            
        except Exception as e:
            if verbose:
                print(f"❌ Error processing NetCDF file: {str(e)}")
            raise

    
