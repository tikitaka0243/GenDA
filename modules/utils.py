import os
import random
import csv
from datetime import datetime
from typing import List, Optional, Dict, Tuple


def get_random_files(directory_path: str, n: int = 1, recursive: bool = False) -> List[str]:
    """
    Randomly select and return n file paths from the specified directory.
    
    Args:
        directory_path (str): Path to the target directory
        n (int, optional): Number of random files to return. Defaults to 1.
        recursive (bool, optional): Whether to search subdirectories recursively. Defaults to False.
    
    Returns:
        List[str]: List of randomly selected file paths (absolute paths)
    
    Raises:
        ValueError: If the directory doesn't exist or n is invalid
        FileNotFoundError: If no files are found in the directory
    """
    # Check if directory exists
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory does not exist: {directory_path}")
    
    if not os.path.isdir(directory_path):
        raise ValueError(f"Path is not a directory: {directory_path}")
    
    # Validate n parameter
    if n < 1:
        raise ValueError("n must be a positive integer")
    
    # Collect all file paths in the directory
    all_files = []
    
    if recursive:
        # Walk through all subdirectories if recursive is True
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    all_files.append(os.path.abspath(file_path))
    else:
        # Only files in the immediate directory
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path):
                all_files.append(os.path.abspath(item_path))
    
    # Check if any files were found
    if not all_files:
        raise FileNotFoundError(f"No files found in directory: {directory_path}")
    
    # Check if n is larger than available files
    if n > len(all_files):
        print(f"Warning: Requested {n} files, but only {len(all_files)} available. Returning all files.")
        n = len(all_files)
    
    # Randomly select n files without replacement
    selected_files = random.sample(all_files, n)
    
    return selected_files