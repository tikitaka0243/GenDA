import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ndtr
from scipy.ndimage import gaussian_filter1d


def plot_pixel_distributions(ensemble, gt, variables, selected_pixels, calibration_models, save_path,
                              lon_coords=None, lat_coords=None):
    """
    Plot pixel distributions with JASA/ACS academic style.
    Single legend for the entire figure.

    Args:
        ensemble: (N, C, H, W) numpy array
        gt: (C, H, W) numpy array
        variables: list of variable names
        selected_pixels: list/array of (h_idx, w_idx)
        calibration_models: dict of calibration models {var: model}
        save_path: path to save the figure
        lon_coords: (W,) array of longitudes, optional. If provided, used for title instead of pixel indices.
        lat_coords: (H,) array of latitudes, optional. If provided, used for title instead of pixel indices.
    """
    # JASA/ACS style settings
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    num_pixels = len(selected_pixels)
    N = ensemble.shape[0]

    # Create figure with extra space for legend
    fig, axes = plt.subplots(len(variables), num_pixels,
                             figsize=(4.0 * num_pixels * 0.7, 3.2 * len(variables) * 0.7),
                             squeeze=False)

    # Color palette for academic style (colorful but professional)
    colors = {
        'raw_pdf': '#1f77b4',      # Blue
        'raw_cdf': '#1f77b4',      # Blue
        'cal_pdf': '#d62728',      # Red
        'cal_cdf': '#d62728',      # Red
        'gt': '#2ca02c',           # Green
        'rug': '#555555'           # Gray
    }

    # Store legend handles and labels
    legend_handles = []
    legend_labels = []

    for v_idx, var in enumerate(variables):
        for p_idx in range(num_pixels):
            h_idx, w_idx = selected_pixels[p_idx]
            ax = axes[v_idx, p_idx]

            # Data
            ens_vals = ensemble[:, v_idx, h_idx, w_idx]  # (N,)
            gt_val = gt[v_idx, h_idx, w_idx]

            # KDE parameters
            sigma = np.std(ens_vals, ddof=1)
            if sigma < 1e-6:
                sigma = 1e-6
            bw = 1.06 * sigma * (N ** (-0.2))

            # Grid
            x_min = min(ens_vals.min(), gt_val) - 3 * bw
            x_max = max(ens_vals.max(), gt_val) + 3 * bw
            x_grid = np.linspace(x_min, x_max, 200)

            # Calculate Z scores for all grid points against all ensemble members
            z = (x_grid[:, None] - ens_vals[None, :]) / bw

            # Raw CDF
            cdf_raw = np.mean(ndtr(z), axis=1)

            # Raw PDF
            pdf_term = np.exp(-0.5 * z**2) / (np.sqrt(2 * np.pi))
            pdf_raw = np.mean(pdf_term / bw, axis=1)

            # Plot Ensemble Members (Rug plot)
            ln0 = ax.plot(ens_vals, np.zeros_like(ens_vals), '|',
                          color=colors['rug'], markersize=8,
                          markeredgewidth=1.0, alpha=0.6)

            # Plot PDF (Left Axis)
            ln1 = ax.plot(x_grid, pdf_raw, color=colors['raw_pdf'],
                          linestyle='--', linewidth=1.5)
            # Only show Density label on leftmost column
            if p_idx == 0:
                ax.set_ylabel("Density", fontsize=12)
            ax.tick_params(axis='y', labelsize=11)

            # Create Twin Axis for CDF
            ax2 = ax.twinx()
            ln2 = ax2.plot(x_grid, cdf_raw, color=colors['raw_cdf'],
                           linestyle='-', linewidth=1.5)
            # Only show CDF label on rightmost column
            if p_idx == num_pixels - 1:
                ax2.set_ylabel("CDF", fontsize=12)
            else:
                ax2.set_yticklabels([])
            ax2.tick_params(axis='y', labelsize=11)
            ax2.set_ylim(0, 1.05)

            # Store handles for legend (only from first subplot)
            if v_idx == 0 and p_idx == 0:
                from matplotlib.lines import Line2D
                legend_handles.append(Line2D([], [], color=colors['rug'], marker='|',
                                              linestyle='None', markersize=8,
                                              markeredgewidth=1.5, label='Ensemble'))
                legend_handles.append(Line2D([], [], color=colors['raw_pdf'],
                                              linestyle='--', linewidth=1.5,
                                              label='Raw PDF'))
                legend_handles.append(Line2D([], [], color=colors['raw_cdf'],
                                              linestyle='-', linewidth=1.5,
                                              label='Raw CDF'))

            # Calibrated
            if calibration_models and var in calibration_models:
                model_data = calibration_models[var]
                # Handle both new format (dict with 'type' and 'model') and old format (direct model)
                if isinstance(model_data, dict) and 'type' in model_data:
                    ir = model_data['model']
                else:
                    ir = model_data
                cdf_cal = ir.predict(cdf_raw)

                # Numerical PDF for Calibrated Distribution
                dx = x_grid[1] - x_grid[0]
                if dx > 0:
                    pdf_cal = np.gradient(cdf_cal, dx)
                    pdf_cal = gaussian_filter1d(pdf_cal, sigma=1.0)
                else:
                    pdf_cal = np.zeros_like(cdf_cal)

                ln3 = ax.plot(x_grid, pdf_cal, color=colors['cal_pdf'],
                              linestyle='-.', linewidth=1.5)
                ln4 = ax2.plot(x_grid, cdf_cal, color=colors['cal_cdf'],
                               linestyle='-', linewidth=1.5)

                # Store handles for legend (only from first subplot)
                if v_idx == 0 and p_idx == 0:
                    legend_handles.append(Line2D([], [], color=colors['cal_pdf'],
                                                  linestyle='-.', linewidth=1.5,
                                                  label='Cal. PDF'))
                    legend_handles.append(Line2D([], [], color=colors['cal_cdf'],
                                                  linestyle='-', linewidth=1.5,
                                                  label='Cal. CDF'))

            # GT
            ln_gt = ax.axvline(gt_val, color=colors['gt'],
                               linestyle=':', linewidth=1.5)

            # Store handle for legend (only from first subplot)
            if v_idx == 0 and p_idx == 0:
                legend_handles.append(Line2D([], [], color=colors['gt'],
                                              linestyle=':', linewidth=1.5,
                                              label='Observation'))

            # Title and labels - use full variable names
            var_name_map = {
                'thetao': 'Temperature',
                'so': 'Salinity',
                'uo': 'Eastward Velocity',
                'vo': 'Northward Velocity'
            }
            var_display = var_name_map.get(var, var)
            
            # Format title with lat/lon if coordinates provided, otherwise use pixel indices
            if lon_coords is not None and lat_coords is not None:
                lon_val = lon_coords[w_idx]
                lat_val = lat_coords[h_idx]
                # Format coordinates with appropriate precision
                lon_str = f"{lon_val:.1f}°E" if lon_val >= 0 else f"{-lon_val:.1f}°W"
                lat_str = f"{lat_val:.1f}°N" if lat_val >= 0 else f"{-lat_val:.1f}°S"
                title_coord = f"({lat_str}, {lon_str})"
            else:
                title_coord = f"({h_idx},{w_idx})"
            
            ax.set_title(f"{var_display} {title_coord}", fontsize=11, fontweight='normal')
            if v_idx == len(variables) - 1:
                ax.set_xlabel("Value", fontsize=12)

            # Clean spines - JASA/ACS style
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)

            # Light grid only on y-axis for PDF
            ax.yaxis.grid(True, linestyle='-', linewidth=0.5,
                          alpha=0.3, color='gray')
            ax.set_axisbelow(True)

    # Add single legend at the bottom of the figure
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=len(legend_handles), fontsize=11,
               frameon=True, fancybox=False,
               edgecolor='gray', framealpha=0.9,
               bbox_to_anchor=(0.5, 0.05))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Pixel distributions saved to {save_path}")

def plot_histograms(rank_counts, n_bins, variables, save_path, all_ranks=None, sample_ratio=0.1):
    """
    Plot rank histograms using KDE on raw PIT values if available,
    otherwise fall back to smoothed binned histogram.

    Args:
        rank_counts: dict {var: counts array}, used as fallback if all_ranks not provided
        n_bins: int, number of bins for evaluation grid
        variables: list of variable names
        save_path: path to save the figure
        all_ranks: dict {var: list of arrays} or dict {var: array} or None. 
                   If provided, uses KDE on raw PIT values for more accurate PDF estimation.
        sample_ratio: float, ratio of data points to sample for KDE (default 0.1 = 10%).
                      Used to reduce memory usage for large datasets.
    """
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    # Evaluation grid for KDE
    x_grid = np.linspace(0, 1, n_bins)

    for i, var in enumerate(variables):
        if i >= len(axes): break
        ax = axes[i]

        if all_ranks is not None and var in all_ranks and len(all_ranks[var]) > 0:
            # Use KDE on raw PIT values
            ranks_data = all_ranks[var]
            
            # Handle both list of arrays and single array
            if isinstance(ranks_data, list):
                # List of arrays - concatenate them
                ranks_concat = np.concatenate(ranks_data)
            else:
                # Single array (e.g., calibrated ranks)
                ranks_concat = ranks_data

            if len(ranks_concat) > 0:
                # Sample data if too large (to reduce memory usage)
                if sample_ratio < 1.0 and len(ranks_concat) > 100000:
                    n_samples = max(int(len(ranks_concat) * sample_ratio), 10000)
                    indices = np.random.choice(len(ranks_concat), size=n_samples, replace=False)
                    ranks_sample = ranks_concat[indices]
                else:
                    ranks_sample = ranks_concat

                # Use gaussian_kde for KDE estimation
                kde = gaussian_kde(ranks_sample)
                density = kde(x_grid)
                ax.plot(x_grid, density, label=f'{var}', linewidth=2)
            else:
                # Fallback to binned data if no raw ranks
                counts = rank_counts[var]
                total_counts = np.sum(counts)
                if total_counts > 0:
                    frequencies = (counts / total_counts) * n_bins
                else:
                    frequencies = counts
                frequencies_smooth = gaussian_filter1d(frequencies, sigma=5.0)
                ax.plot(x_grid, frequencies_smooth, label=f'{var}', linewidth=2)
        else:
            # Fallback to smoothed binned histogram
            counts = rank_counts[var]
            total_counts = np.sum(counts)
            if total_counts > 0:
                frequencies = (counts / total_counts) * n_bins
            else:
                frequencies = counts
            frequencies_smooth = gaussian_filter1d(frequencies, sigma=5.0)
            ax.plot(x_grid, frequencies_smooth, label=f'{var}', linewidth=2)

        ax.plot([0, 1], [1, 1], 'k--', alpha=0.5, label='Ideal')

        ax.set_title(f'{var} PIT Distribution (KDE)')
        ax.set_xlabel('Probability Integral Transform (PIT)')
        ax.set_ylabel('Frequency Density')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 2.0)  # Limit y-axis to avoid extreme spikes
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Rank histogram saved to {save_path}")


def plot_combined_rank_histograms(cal_output_dir, test_output_dir, 
                                   n_bins=200, save_path=None):
    """
    Plot combined rank histograms for calibration and test sets.
    
    Creates a 2x2 grid where each subplot corresponds to one variable.
    Each subplot shows 4 lines:
    - Calibration set raw PIT (dashed)
    - Test set raw PIT (dashed)  
    - Calibration set calibrated PIT (solid)
    - Test set calibrated PIT (solid)
    
    Data is loaded from cache files in conformal/.cache to avoid recomputation.
    Uses binned histogram counts (not raw ranks) for plotting.
    
    Args:
        cal_output_dir: calibration set output directory (contains conformal/.cache)
        test_output_dir: test set output directory (contains conformal/.cache)
        n_bins: number of bins for histogram (default 200)
        save_path: path to save the figure (default: parent_dir/combined_rank_histograms.png)
    """
    import os
    import glob
    import joblib
    from scipy.ndimage import gaussian_filter1d
    
    def _find_cache_files(cache_dir, calibrated=False):
        """Find cache files in the given directory."""
        prefix = 'calibrated_rank_histogram_cache_' if calibrated else 'rank_histogram_cache_'
        pattern = os.path.join(cache_dir, f'{prefix}*.pkl')
        files = glob.glob(pattern)
        return files[0] if files else None
    
    def _get_cache_path(output_dir, calibrated=False):
        """Get cache path from output directory."""
        cache_dir = os.path.join(output_dir, 'conformal', '.cache')
        cache_path = _find_cache_files(cache_dir, calibrated)
        return cache_path
    
    # Default save path: save to the parent directory of cal_output_dir and test_output_dir
    # (the common parent directory containing both calibration and test output folders)
    if save_path is None:
        # Get the parent directory of test_output_dir (e.g., geoFuse_argo5)
        parent_dir = os.path.dirname(os.path.normpath(test_output_dir))
        save_path = os.path.join(parent_dir, 'combined_rank_histograms.pdf')
    
    # Variables and their display names
    variables = ['thetao', 'so', 'uo', 'vo']
    var_labels = {
        'thetao': 'Temperature',
        'so': 'Salinity', 
        'uo': 'Eastward Velocity',
        'vo': 'Northward Velocity'
    }
    
    # Line styles and labels
    line_configs = [
        {'key': 'cal_raw', 'label': 'Calibration Raw', 'linestyle': '--', 'color_idx': 0},
        {'key': 'test_raw', 'label': 'Test Raw', 'linestyle': '--', 'color_idx': 1},
        {'key': 'cal_calibrated', 'label': 'Calibration Calibrated', 'linestyle': '-', 'color_idx': 0},
        {'key': 'test_calibrated', 'label': 'Test Calibrated', 'linestyle': '-', 'color_idx': 1},
    ]
    
    # Color palette (2 colors for cal/test, different line styles for raw/calibrated)
    colors = ['#003049', '#F77F00']  # Dark blue for cal, orange for test
    
    # Load all cache data (rank_counts, not all_ranks)
    cache_data = {}
    
    # Load calibration set raw cache
    cal_raw_path = _get_cache_path(cal_output_dir, calibrated=False)
    if cal_raw_path and os.path.exists(cal_raw_path):
        try:
            cache = joblib.load(cal_raw_path)
            cache_data['cal_raw'] = cache.get('rank_counts')
            print(f"Loaded calibration raw cache: {cal_raw_path}")
        except Exception as e:
            print(f"Error loading calibration raw cache: {e}")
    else:
        print(f"Calibration raw cache not found in {cal_output_dir}/conformal/.cache")
    
    # Load calibration set calibrated cache
    cal_cal_path = _get_cache_path(cal_output_dir, calibrated=True)
    if cal_cal_path and os.path.exists(cal_cal_path):
        try:
            cache = joblib.load(cal_cal_path)
            cache_data['cal_calibrated'] = cache.get('calibrated_rank_counts')
            print(f"Loaded calibration calibrated cache: {cal_cal_path}")
        except Exception as e:
            print(f"Error loading calibration calibrated cache: {e}")
    else:
        print(f"Calibration calibrated cache not found in {cal_output_dir}/conformal/.cache")
    
    # Load test set raw cache
    test_raw_path = _get_cache_path(test_output_dir, calibrated=False)
    if test_raw_path and os.path.exists(test_raw_path):
        try:
            cache = joblib.load(test_raw_path)
            cache_data['test_raw'] = cache.get('rank_counts')
            print(f"Loaded test raw cache: {test_raw_path}")
        except Exception as e:
            print(f"Error loading test raw cache: {e}")
    else:
        print(f"Test raw cache not found in {test_output_dir}/conformal/.cache")
    
    # Load test set calibrated cache
    test_cal_path = _get_cache_path(test_output_dir, calibrated=True)
    if test_cal_path and os.path.exists(test_cal_path):
        try:
            cache = joblib.load(test_cal_path)
            cache_data['test_calibrated'] = cache.get('calibrated_rank_counts')
            print(f"Loaded test calibrated cache: {test_cal_path}")
        except Exception as e:
            print(f"Error loading test calibrated cache: {e}")
    else:
        print(f"Test calibrated cache not found in {test_output_dir}/conformal/.cache")
    
    # Set Times New Roman font for all text elements
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create figure with 1x4 subplots (one per variable, horizontal layout)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes = axes.flatten()
    
    # Evaluation grid for histogram
    x_grid = np.linspace(0, 1, n_bins)
    
    # Store legend handles and labels from the first subplot
    legend_handles = []
    legend_labels = []
    
    for var_idx, (ax, var) in enumerate(zip(axes, variables)):
        var_display = var_labels.get(var, var)
        ax.set_title(f'{var_display}', fontsize=12, fontname='Times New Roman')
        ax.set_xlabel('Probability Integral Transform (PIT)', fontsize=10, fontname='Times New Roman')
        if var_idx == 0:
            ax.set_ylabel('Frequency Density', fontsize=10, fontname='Times New Roman')
        else:
            ax.set_yticklabels([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 2.0)
        ax.grid(True, alpha=0.3)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Plot ideal uniform line
        line_ideal = ax.plot([0, 1], [1, 1], 'k--', alpha=0.5, linewidth=1, label='Ideal')
        
        # Plot each line type using binned counts
        for config in line_configs:
            data_key = config['key']
            if data_key not in cache_data or cache_data[data_key] is None:
                continue
            
            rank_counts_dict = cache_data[data_key]
            if var not in rank_counts_dict or rank_counts_dict[var] is None:
                continue
            
            counts = rank_counts_dict[var]
            
            if len(counts) == 0:
                continue
            
            # Convert counts to frequency density (same as plot_histograms)
            total_counts = np.sum(counts)
            if total_counts > 0:
                frequencies = (counts / total_counts) * n_bins
            else:
                frequencies = counts
            
            # Smooth the frequencies using gaussian filter
            frequencies_smooth = gaussian_filter1d(frequencies, sigma=5.0)
            
            # Plot the smoothed histogram
            color = colors[config['color_idx']]
            line = ax.plot(x_grid, frequencies_smooth, color=color, linewidth=1.5, 
                   linestyle=config['linestyle'], label=config['label'])
            
            # Store legend info from first subplot
            if var_idx == 0:
                legend_handles.append(line[0])
                legend_labels.append(config['label'])
        
        # Store ideal line handle from first subplot
        if var_idx == 0:
            legend_handles.insert(0, line_ideal[0])
            legend_labels.insert(0, 'Ideal')
    
    # Add single legend below all subplots
    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=5, 
               fontsize=10, frameon=True, fancybox=False, edgecolor='gray', 
               framealpha=0.95, bbox_to_anchor=(0.5, 0))
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Combined rank histograms saved to {save_path}")


def plot_coverage_results_comparison(files_by_group, save_path,
                                      group_display_names=None,
                                      colors=None):
    """
    Plot coverage results comparison for calibration and test sets.

    Creates a 1x4 subplot grid showing Empirical Coverage vs Target Coverage
    for Temperature, Salinity, Eastward Velocity, and Northward Velocity.
    Each subplot shows 5 lines:
    - Ideal (y=x) dashed line
    - Calibration Uncalibrated (dashed with markers)
    - Calibration Calibrated (solid with markers)
    - Test Uncalibrated (dashed with markers)
    - Test Calibrated (solid with markers)

    Args:
        files_by_group: dict with keys 'cal' and 'test', each containing list of file paths
                       for different confidence levels [0.8, 0.9, 0.95, 0.99]
        save_path: path to save the figure
        group_display_names: dict mapping group keys to display names,
                            e.g., {'test': 'Test', 'cal': 'Calibration'}
        colors: list of two colors for [Calibration, Test] groups
    """
    import re
    
    # Default display names
    if group_display_names is None:
        group_display_names = {'test': 'Test', 'cal': 'Calibration'}
    
    # Default colors - blue-green color scheme
    if colors is None:
        colors = ['#1A5F7A', '#57C5B6']  # Deep teal and light turquoise
    
    # Variables and their display names
    variables = ['thetao', 'so', 'uo', 'vo']
    var_labels = {
        'thetao': 'Temperature',
        'so': 'Salinity', 
        'uo': 'Eastward Velocity',
        'vo': 'Northward Velocity'
    }
    
    # Parse coverage results from files
    # Structure: data[group][calibrated][variable] = list of (target_coverage, empirical_coverage)
    data = {
        'cal': {'uncalibrated': {}, 'calibrated': {}},
        'test': {'uncalibrated': {}, 'calibrated': {}}
    }

    for group in ['cal', 'test']:
        if group not in files_by_group:
            continue
        for file_path in files_by_group[group]:
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue

            # Extract confidence level from filename (e.g., coverage_results_0.8.txt -> 0.8)
            match = re.search(r'coverage_results_(\d+\.?\d*)\.txt', os.path.basename(file_path))
            if match:
                confidence_level = float(match.group(1))
            else:
                continue

            # Parse the file - table format with columns: Variable | Uncalibrated | Calibrated
            with open(file_path, 'r') as f:
                content = f.read()

            # Extract target coverage from header line: "Coverage Results (Target: 0.8000, Interval: [...])"
            target_match = re.search(r'Target:\s*(\d+\.?\d*)', content)
            if target_match:
                target_coverage = float(target_match.group(1))
            else:
                target_coverage = confidence_level

            # Parse table rows - look for variable names followed by coverage values
            # Format: "thetao     | 0.5058          | 0.7520"
            for var in variables:
                # Match variable name at start of line, followed by two decimal numbers
                pattern = rf'^{var}\s*\|\s*(\d+\.?\d*)\s*\|\s*(\d+\.?\d*)'
                var_match = re.search(pattern, content, re.MULTILINE)
                if var_match:
                    uncalibrated = float(var_match.group(1))
                    calibrated = float(var_match.group(2))

                    # Store uncalibrated data
                    if var not in data[group]['uncalibrated']:
                        data[group]['uncalibrated'][var] = []
                    data[group]['uncalibrated'][var].append((target_coverage, uncalibrated))

                    # Store calibrated data
                    if var not in data[group]['calibrated']:
                        data[group]['calibrated'][var] = []
                    data[group]['calibrated'][var].append((target_coverage, calibrated))
    
    # Set Times New Roman font
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create figure with 1x4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes = axes.flatten()
    
    # Store legend handles and labels
    legend_handles = []
    legend_labels = []
    
    # Line styles and markers
    marker_size = 6
    line_width = 1.5
    
    for var_idx, (ax, var) in enumerate(zip(axes, variables)):
        var_display = var_labels.get(var, var)
        ax.set_title(f'{var_display}', fontsize=12, fontname='Times New Roman')
        ax.set_xlabel('Target Coverage', fontsize=10, fontname='Times New Roman')
        if var_idx == 0:
            ax.set_ylabel('Empirical Coverage', fontsize=10, fontname='Times New Roman')
        else:
            ax.set_yticklabels([])
        
        ax.set_xlim(0.78, 1.02)
        ax.set_ylim(0.45, 1.05)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Plot ideal line (y=x) - light gray color and extended to bottom
        ideal_x = np.array([0.45, 1.0])
        line_ideal = ax.plot(ideal_x, ideal_x, '--', color='#A0A0A0', linewidth=1.2, label='Ideal (y=x)', zorder=0)
        
        # Plot data for each group and calibration status
        # Order: Calibration Uncalibrated, Calibration Calibrated, Test Uncalibrated, Test Calibrated
        plot_configs = [
            ('cal', 'uncalibrated', f"{group_display_names.get('cal', 'Calibration')} Uncalibrated", '--', 'o'),
            ('cal', 'calibrated', f"{group_display_names.get('cal', 'Calibration')} Calibrated", '-', 'o'),
            ('test', 'uncalibrated', f"{group_display_names.get('test', 'Test')} Uncalibrated", '--', 'o'),
            ('test', 'calibrated', f"{group_display_names.get('test', 'Test')} Calibrated", '-', 'o'),
        ]
        
        for config_idx, (group, cal_status, label, linestyle, marker) in enumerate(plot_configs):
            if var not in data[group].get(cal_status, {}):
                continue
            
            points = data[group][cal_status][var]
            if len(points) == 0:
                continue
            
            # Sort by target coverage
            points = sorted(points, key=lambda x: x[0])
            targets = [p[0] for p in points]
            empiricals = [p[1] for p in points]
            
            # Choose color: index 0 for cal group, index 1 for test group
            # Both calibrated and uncalibrated within same group use the same color
            color_idx = 0 if group == 'cal' else 1
            plot_color = colors[color_idx]
            
            line = ax.plot(targets, empiricals, color=plot_color, linestyle=linestyle,
                          marker=marker, markersize=marker_size, linewidth=line_width,
                          markerfacecolor=plot_color, markeredgecolor='white',
                          markeredgewidth=0.5, label=label)
            
            # Store legend info from first subplot
            if var_idx == 0:
                legend_handles.append(line[0])
                legend_labels.append(label)
        
        # Store ideal line handle from first subplot
        if var_idx == 0:
            legend_handles.insert(0, line_ideal[0])
            legend_labels.insert(0, 'Ideal (y=x)')
    
    # Add single legend below all subplots
    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=5,
               fontsize=10, frameon=True, fancybox=False, edgecolor='gray',
               framealpha=0.95, bbox_to_anchor=(0.5, 0.06))

    plt.tight_layout(rect=[0, 0.14, 1, 1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Coverage results comparison saved to {save_path}")


def plot_calibration_fit(var, U_sorted, V, model, method, output_dir, predict_fn=None):
    """
    Visualize the isotonic regression fitting results and data.
    Creates a 3-panel plot: raw data scatter, calibration curve, and residuals.
    
    Args:
        var: variable name (str)
        U_sorted: sorted PIT values (input to calibration), shape (M,)
        V: target uniform distribution values, shape (M,)
        model: fitted calibration model (IsotonicRegression or MLP model)
        method: 'sklearn' or 'mlp'
        output_dir: directory to save the figure (will create 'calibration_fit' subfolder)
        predict_fn: optional prediction function for MLP (if None, uses model.predict)
    """
    # Create output directory with subfolder for calibration fit visualizations
    output_dir = os.path.join(output_dir, 'calibration_fit')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Scatter plot of raw data (U vs V)
    ax1 = axes[0]
    # Sample points for visualization if too many
    if len(U_sorted) > 10000:
        indices = np.random.choice(len(U_sorted), 10000, replace=False)
        indices = np.sort(indices)
        U_plot = U_sorted[indices]
        V_plot = V[indices]
    else:
        U_plot = U_sorted
        V_plot = V
    ax1.scatter(U_plot, V_plot, alpha=0.3, s=1, color='blue', label='Data points')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration (y=x)')
    ax1.set_xlabel('PIT Values (U)', fontsize=11)
    ax1.set_ylabel('Target Uniform (V)', fontsize=11)
    ax1.set_title(f'{var}: Raw Data (U vs V)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Calibration curve (fitted function)
    ax2 = axes[1]
    U_test = np.linspace(0, 1, 1000)
    
    if predict_fn is not None:
        # Use custom prediction function (for MLP)
        V_pred = predict_fn(model, U_test)
    else:
        # Use model's predict method (for sklearn)
        V_pred = model.predict(U_test)
    
    ax2.plot(U_test, V_pred, 'r-', linewidth=2, label='Calibration function f(U)')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration (y=x)')
    ax2.set_xlabel('Input PIT (U)', fontsize=11)
    ax2.set_ylabel('Calibrated Output f(U)', fontsize=11)
    ax2.set_title(f'{var}: Calibration Curve ({method})', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Residuals (V - f(U))
    ax3 = axes[2]
    if predict_fn is not None:
        V_fitted = predict_fn(model, U_sorted)
    else:
        V_fitted = model.predict(U_sorted)
    residuals = V - V_fitted
    
    # Sample for visualization
    if len(residuals) > 10000:
        indices = np.random.choice(len(residuals), 10000, replace=False)
        indices = np.sort(indices)
        U_resid = U_sorted[indices]
        resid_plot = residuals[indices]
    else:
        U_resid = U_sorted
        resid_plot = residuals
    
    ax3.scatter(U_resid, resid_plot, alpha=0.3, s=1, color='green')
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax3.set_xlabel('PIT Values (U)', fontsize=11)
    ax3.set_ylabel('Residuals (V - f(U))', fontsize=11)
    ax3.set_title(f'{var}: Residuals (mean={np.mean(residuals):.4f})', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)
    
    # Add statistics as text
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    stats_text = f'MSE: {mse:.6f}\nRMSE: {rmse:.6f}\nSamples: {len(U_sorted)}'
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    # Save figure
    save_path = os.path.join(output_dir, f'calibration_fit_{var}_{method}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Calibration fit visualization saved to {save_path}")
