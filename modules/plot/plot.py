import os

from torch.nn.modules import padding
from modules.utils import get_random_files
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import torch
import numpy as np
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import math


def plot_ocean_variables(ds: xr.Dataset, 
                        output_path: str = "ocean_variables_plot.png",
                        figsize: tuple = (16, 12),
                        dpi: int = 300,
                        vmin_vmax: dict = None,
                        title: str = None) -> None:
    """
    Plot four ocean variables (thetao, so, uo, vo) from an xarray Dataset in a 2x2 subplot layout.
    Colorbars for velocity variables (uo, vo) are symmetric around zero.
    
    Args:
        ds (xarray.Dataset): Dataset containing ocean variables with dimensions (time, latitude, longitude)
        output_path (str): Path where the plot image will be saved
        figsize (tuple): Figure size (width, height) in inches
        dpi (int): Resolution of the output image
        vmin_vmax (dict, optional): Dictionary containing vmin/vmax for each variable. 
                                    Format: {'var_name': (vmin, vmax)}
        title (str, optional): Custom title for the figure.
    
    Returns:
        None: Saves the plot to the specified output path
    """
    # Extract the first time step for plotting (assuming single time or using first time)
    ds_plot = ds.isel(time=0)
    
    # Create figure and subplots with PlateCarree projection
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True,
                             subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.ravel()  # Flatten the 2x2 array for easy indexing
    
    # Define variable names and their descriptive titles
    variables = ['thetao', 'so', 'uo', 'vo']
    titles = [
        'Sea Water Potential Temperature (°C)',
        'Sea Water Salinity (psu)',
        'Eastward Sea Water Velocity (m/s)',
        'Northward Sea Water Velocity (m/s)'
    ]
    
    # Define colormaps for each variable
    cmaps = ['plasma', 'viridis', 'RdBu_r', 'RdGy_r']
    
    # Identify which variables are velocity fields (need symmetric colorbar)
    velocity_vars = ['uo', 'vo']
    
    # Plot each variable
    for i, (var, title_, cmap) in enumerate(zip(variables, titles, cmaps)):
        # Select data for current variable
        data = ds_plot[var]
        
        # Determine colorbar limits
        if vmin_vmax and var in vmin_vmax:
            vmin, vmax = vmin_vmax[var]
        else:
            if var in velocity_vars:
                # For velocity variables: make colorbar symmetric around zero
                # Handle NaNs in data
                data_min = np.nanmin(data.values) if not np.all(np.isnan(data.values)) else 0
                data_max = np.nanmax(data.values) if not np.all(np.isnan(data.values)) else 0
                vmax = max(abs(data_min), abs(data_max))
                vmin = -vmax
                # Ensure we don't have invalid range when all values are zero
                if vmax == 0:
                    vmin, vmax = -1, 1
            else:
                # For non-velocity variables: use data range
                vmin = np.nanmin(data.values) if not np.all(np.isnan(data.values)) else 0
                vmax = np.nanmax(data.values) if not np.all(np.isnan(data.values)) else 1
                # Handle case where min == max
                if vmin == vmax:
                    vmin, vmax = vmin - 0.5, vmax + 0.5
        
        # Create the plot with specified color limits
        im = axes[i].pcolormesh(data.longitude, data.latitude, data.values, 
                               cmap=cmap, shading='auto', vmin=vmin, vmax=vmax,
                               transform=ccrs.PlateCarree())
        
        # Add coastlines and land
        axes[i].add_feature(cfeature.COASTLINE, linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[i], shrink=0.8, extend='both', pad=0.03)
        
        # Extract units for colorbar label (more robust method)
        if '(' in title_ and ')' in title_:
            # Extract content between parentheses for units
            units = title_.split('(')[-1].split(')')[0]
        else:
            # Fallback: use variable name if no parentheses found
            units = var
        cbar.set_label(units, rotation=90)
        
        # Set title and labels
        axes[i].set_title(title_, fontsize=12, fontweight='bold')
        
        # Gridlines with labels
        gl = axes[i].gridlines(draw_labels=True, alpha=0.3, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}
        
        # Set reasonable axis limits (global extent)
        axes[i].set_extent([-180, 180, -80, 90], crs=ccrs.PlateCarree())
        
        # Add annotation for symmetric colorbar if applicable
        if var in velocity_vars:
            axes[i].text(0.02, 0.98, 'Symmetric colorbar', 
                        transform=axes[i].transAxes, fontsize=8,
                        verticalalignment='top', bbox=dict(boxstyle='round', 
                                                          facecolor='white', alpha=0.8))
    
    # Add super title with time information
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    else:
        time_str = str(ds_plot.time.values).split('T')[0]  # Extract date part
        fig.suptitle(f'Ocean Variables - {time_str}', fontsize=16, fontweight='bold')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', 
                exist_ok=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()  # Close the figure to free memory
    
    print(f"Plot saved successfully to: {output_path}")


def plot_diff_variables(ds: xr.Dataset, 
                       output_path: str = "ocean_variables_diff_plot.png",
                       figsize: tuple = (20, 12),
                       dpi: int = 300,
                       title: str = None) -> None:
    """
    Plot differences of four ocean variables (thetao, so, uo, vo) from an xarray Dataset.
    Uses symmetric colorbars centered at zero for all variables.
    
    Args:
        ds (xarray.Dataset): Dataset containing difference variables
        output_path (str): Path where the plot image will be saved
        figsize (tuple): Figure size (width, height) in inches
        dpi (int): Resolution of the output image
        title (str, optional): Custom title for the figure.
    
    Returns:
        None: Saves the plot to the specified output path
    """
    # Extract the first time step for plotting
    ds_plot = ds.isel(time=0)
    
    # Create figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axes = axes.ravel()
    
    # Define variable names and their descriptive titles
    variables = ['thetao', 'so', 'uo', 'vo']
    titles = [
        'Sea Water Potential Temperature Diff (°C)',
        'Sea Water Salinity Diff (psu)',
        'Eastward Sea Water Velocity Diff (m/s)',
        'Northward Sea Water Velocity Diff (m/s)'
    ]
    
    # Use diverging colormap for all difference plots
    cmap = 'RdBu_r'
    
    # Plot each variable
    for i, (var, title_) in enumerate(zip(variables, titles)):
        # Select data for current variable
        data = ds_plot[var]
        
        # Determine symmetric colorbar limits
        vmax = max(abs(data.min()), abs(data.max()))
        vmin = -vmax
        
        # Ensure we don't have invalid range when all values are zero
        if vmax == 0:
            vmin, vmax = -1, 1
        
        # Create the plot
        im = axes[i].pcolormesh(data.longitude, data.latitude, data.values, 
                               cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[i], shrink=0.8)
        
        # Extract units for colorbar label
        if '(' in title_ and ')' in title_:
            units = title_.split('(')[-1].split(')')[0]
        else:
            units = var
        cbar.set_label(units, rotation=90)
        
        # Set title and labels
        axes[i].set_title(title_, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Longitude (°)')
        axes[i].set_ylabel('Latitude (°)')
        
        # Set reasonable axis limits
        axes[i].set_xlim(-180, 180)
        axes[i].set_ylim(-80, 90)
        
        # Add grid
        axes[i].grid(alpha=0.3, linestyle='--')
        
        # Add annotation
        axes[i].text(0.02, 0.98, 'Difference (Sample - GT)', 
                    transform=axes[i].transAxes, fontsize=8,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                                                      facecolor='white', alpha=0.8))
    
    # Add super title with time information
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    else:
        time_str = str(ds_plot.time.values).split('T')[0]
        fig.suptitle(f'Ocean Variables Difference - {time_str}', fontsize=16, fontweight='bold')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', 
                exist_ok=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Diff plot saved successfully to: {output_path}")


def plot_argo_variables(pres=None, temp=None, psal=None,
                       output_path: str = "argo_variables_plot.png",
                       figsize: tuple = (10, 5),
                       dpi: int = 300) -> None:
    """
    Plot ARGO float temperature and salinity profiles against pressure (depth).
    
    Args:
        pres (array-like, optional): Pressure data (dbar) - used as vertical axis
        temp (array-like, optional): Temperature data (°C)
        psal (array-like, optional): Salinity data (psu)
        output_path (str): Path where the plot image will be saved
        figsize (tuple): Figure size (width, height) in inches
        dpi (int): Resolution of the output image
    
    Returns:
        None: Saves the plot to the specified output path
    """
    # Squeeze input variables to remove singleton dimensions
    if pres is not None:
        pres = pres.squeeze()
    if temp is not None:
        temp = temp.squeeze()
    if psal is not None:
        psal = psal.squeeze()
    
    # Create figure and subplots (now only 2 subplots)
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    
    # Convert pressure to array if needed
    pres_array = xr.DataArray(pres) if not isinstance(pres, xr.DataArray) else pres
    
    # Define variable data, names and their descriptive titles
    variables_data = [temp, psal]
    variables = ['TEMP', 'PSAL']
    titles = [
        'Temperature Profile (°C)',
        'Salinity Profile (psu)'
    ]
    
    # Define units for each variable
    units = ['°C', 'psu']
    
    # Plot each variable
    for i, (data, var, title, unit) in enumerate(zip(variables_data, variables, titles, units)):
        # Check if variable data is provided
        if data is None:
            axes[i].text(0.5, 0.5, f'{var}\nNot Provided', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(title)
            continue
            
        # Convert to array if needed
        data_array = xr.DataArray(data) if not isinstance(data, xr.DataArray) else data
        
        # Convert to array if needed
        data_array = xr.DataArray(data) if not isinstance(data, xr.DataArray) else data
        
        # Plot values against pressure (y-axis)
        axes[i].plot(data_array.values, pres_array.values, 
                    color='black', linewidth=1.5)
        
        # Set title and labels
        axes[i].set_title(title, fontsize=12, fontweight='bold')
        axes[i].set_xlabel(unit)
        axes[i].set_ylabel('Pressure (dbar)')
        
        # Invert y-axis so depth increases downward
        axes[i].invert_yaxis()
        
        # Add grid
        axes[i].grid(alpha=0.3, linestyle='--')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', 
                exist_ok=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()  # Close the figure to free memory
    
    print(f"ARGO variables plot saved successfully to: {output_path}")


def plot_variable_distribution(data_dict: dict,
                              output_path: str = "variable_distribution.png",
                              figsize: tuple = (16, 10),
                              dpi: int = 300,
                              bins: int = 50,
                              log_scale: bool = True) -> None:
    """
    Plot histograms showing the distribution of values for each variable.
    
    Args:
        data_dict (dict): Dictionary where keys are variable names and values are numpy arrays or tensors
                         Format: {'thetao': array, 'so': array, 'uo': array, 'vo': array}
        output_path (str): Path where the plot image will be saved
        figsize (tuple): Figure size (width, height) in inches
        dpi (int): Resolution of the output image
        bins (int): Number of bins for histograms
        log_scale (bool): Whether to use log scale for y-axis. Default: True
    
    Returns:
        None: Saves the plot to the specified output path
    """
    import numpy as np
    import torch
    
    num_vars = len(data_dict)
    if num_vars == 0:
        print("Warning: No variables to plot")
        return
    
    # Calculate grid layout
    ncols = 2
    nrows = (num_vars + 1) // 2
    
    # Create figure and subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    if num_vars == 1:
        axes = np.array([axes])
    axes = axes.ravel() if num_vars > 1 else axes
    
    # Define nice titles and colors for common ocean variables
    var_info = {
        'thetao': {'title': 'Sea Water Potential Temperature', 'unit': '°C', 'color': '#e74c3c'},
        'so': {'title': 'Sea Water Salinity', 'unit': 'psu', 'color': '#3498db'},
        'uo': {'title': 'Eastward Sea Water Velocity', 'unit': 'm/s', 'color': '#2ecc71'},
        'vo': {'title': 'Northward Sea Water Velocity', 'unit': 'm/s', 'color': '#f39c12'}
    }
    
    # Plot histogram for each variable
    for i, (var_name, data) in enumerate(data_dict.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Convert tensor to numpy if needed
        if torch.is_tensor(data):
            data = data.cpu().numpy()
        
        # Flatten the data and remove NaN values
        data_flat = data.flatten()
        data_valid = data_flat[~np.isnan(data_flat)]
        
        if len(data_valid) == 0:
            ax.text(0.5, 0.5, f'{var_name}\nNo valid data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(var_info.get(var_name, {}).get('title', var_name))
            continue
        
        # Get variable info
        info = var_info.get(var_name, {'title': var_name, 'unit': '', 'color': '#95a5a6'})
        
        # Plot histogram
        n, bins_edges, patches = ax.hist(data_valid, bins=bins, 
                                         color=info['color'], alpha=0.7, 
                                         edgecolor='black', linewidth=0.5)
        
        # Add statistical lines
        mean_val = np.mean(data_valid)
        median_val = np.median(data_valid)
        std_val = np.std(data_valid)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')
        
        # Add shaded region for one standard deviation
        ax.axvspan(mean_val - std_val, mean_val + std_val, alpha=0.2, color='gray', 
                  label=f'±1 Std: {std_val:.4f}')
        
        # Set title and labels
        title = info['title']
        unit = info['unit']
        ax.set_title(f"{title}\n({var_name})", fontsize=12, fontweight='bold')
        ax.set_xlabel(f"Value ({unit})" if unit else "Value", fontsize=10)
        ax.set_ylabel('Frequency (log scale)' if log_scale else 'Frequency', fontsize=10)
        
        # Set y-axis to log scale if requested
        if log_scale:
            ax.set_yscale('log')
        
        # Add grid
        ax.grid(alpha=0.3, linestyle='--', axis='y')
        
        # Add legend
        ax.legend(loc='upper right', fontsize=8)
        
        # Add statistics text box
        stats_text = f"Total: {len(data_valid):,}\nMin: {np.min(data_valid):.4f}\nMax: {np.max(data_valid):.4f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    # Add super title
    fig.suptitle('Variable Value Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', 
                exist_ok=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()  # Close the figure to free memory
    
    print(f"Variable distribution plot saved successfully to: {output_path}")


def plot_rmse_boxplots(rmse_data_by_model: dict,
                       variables: list,
                       output_path: str = "rmse_boxplot_comparison.png",
                       figsize: tuple = (16, 12),
                       dpi: int = 300,
                       title: str = None,
                       srmse_data_by_model: dict = None,
                       show_srmse_summary: bool = True) -> None:
    if not rmse_data_by_model:
        print("Warning: No RMSE data to plot")
        return

    model_names = list(rmse_data_by_model.keys())
    n_vars = len(variables)
    has_srmse = bool(srmse_data_by_model) and show_srmse_summary
    
    # --- 布局逻辑修改开始 ---
    ncols = 2
    # 计算变量所需的行数 (向上取整)
    n_var_rows = math.ceil(n_vars / ncols)
    # 总行数 = 变量行数 + (如果有sRMSE则加1行)
    nrows = n_var_rows + (1 if has_srmse else 0)

    # 创建 Figure，使用 constrained_layout 自动调整间距
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    # 定义网格布局
    gs = fig.add_gridspec(nrows, ncols)
    # --- 布局逻辑修改结束 ---

    var_info = {
        'thetao': {'title': 'Sea Water Potential Temperature', 'unit': '°C'},
        'so': {'title': 'Sea Water Salinity', 'unit': 'psu'},
        'uo': {'title': 'Eastward Sea Water Velocity', 'unit': 'm/s'},
        'vo': {'title': 'Northward Sea Water Velocity', 'unit': 'm/s'}
    }

    # 1. 绘制普通变量 Boxplot
    for i, var in enumerate(variables):
        # 计算当前变量所在的行和列
        row = i // ncols
        col = i % ncols
        
        # 在指定网格位置创建子图
        ax = fig.add_subplot(gs[row, col])
        
        data_list = [rmse_data_by_model.get(model, {}).get(var, []) for model in model_names]
        has_data = any(len(values) > 0 for values in data_list)

        if not has_data:
            ax.text(0.5, 0.5, f'{var}\nNo valid data',
                    ha='center', va='center', transform=ax.transAxes)
            info = var_info.get(var, {'title': var, 'unit': ''})
            ax.set_title(f"{info['title']}\n({var})", fontsize=12, fontweight='bold')
            ax.set_ylabel('RMSE', fontsize=10)
            ax.set_xticks([])
        else:
            ax.boxplot(data_list, labels=model_names, showfliers=True)
            info = var_info.get(var, {'title': var, 'unit': ''})
            ax.set_title(f"{info['title']}\n({var})", fontsize=12, fontweight='bold')
            ax.set_ylabel(f"RMSE ({info['unit']})" if info['unit'] else "RMSE", fontsize=10)
            ax.tick_params(axis='x', rotation=30)
            ax.grid(alpha=0.3, linestyle='--', axis='y')

    # 2. 绘制 sRMSE Boxplot (如果存在)
    if has_srmse:
        # 在最后一行，横跨所有列 (0到最后)
        # gs[n_var_rows, :] 表示占据第 n_var_rows 行的所有列
        ax = fig.add_subplot(gs[n_var_rows, :])
        
        srmse_list = []
        srmse_labels = []
        for model in model_names:
            for var in variables:
                srmse_list.append(srmse_data_by_model.get(model, {}).get(var, []))
                # 标签可能比较长，横跨两列后有更多空间显示
                srmse_labels.append(f"{model}\n{var}")
        
        has_data = any(len(values) > 0 for values in srmse_list)
        if not has_data:
            ax.text(0.5, 0.5, 'sRMSE\nNo valid data',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Standardized RMSE', fontsize=12, fontweight='bold')
            ax.set_ylabel('sRMSE', fontsize=10)
            ax.set_xticks([])
        else:
            # 增加宽度参数 width，使箱体看起来更协调（可选）
            ax.boxplot(srmse_list, labels=srmse_labels, showfliers=True)
            ax.set_title('Standardized RMSE', fontsize=12, fontweight='bold')
            ax.set_ylabel('sRMSE', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(alpha=0.3, linestyle='--', axis='y')

    # 设置总标题
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    else:
        fig.suptitle('RMSE Boxplot Comparison', fontsize=16, fontweight='bold')

    # 保存图片
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.',
                exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"RMSE boxplot saved successfully to: {output_path}")


def plot_tensor_variables(tensor_data,
                          output_path: str = "tensor_variables_plot.png",
                          title_prefix: str = "Tensor",
                          figsize: tuple = (16, 12),
                          dpi: int = 300) -> None:
    """
    Plot 4-channel tensor data (thetao, so, uo, vo) in a 2x2 subplot layout.
    
    Args:
        tensor_data: Tensor of shape (C, H, W) where C=4 for four ocean variables
        output_path (str): Path where the plot image will be saved
        title_prefix (str): Prefix for the plot title
        figsize (tuple): Figure size (width, height) in inches
        dpi (int): Resolution of the output image
    
    Returns:
        None: Saves the plot to the specified output path
    """
    # Convert tensor to numpy if needed
    if torch.is_tensor(tensor_data):
        # Convert BFloat16 to Float32 first, then to numpy
        data = tensor_data.detach().float().cpu().numpy()
    else:
        data = np.array(tensor_data)
    
    # Ensure data has correct shape (C, H, W)
    if data.ndim != 3 or data.shape[0] != 4:
        raise ValueError(f"Expected tensor shape (4, H, W), got {data.shape}")
    
    # Create figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axes = axes.ravel()
    
    # Define variable names and their descriptive titles
    variables = ['thetao', 'so', 'uo', 'vo']
    titles = [
        'Sea Water Potential Temperature (°C)',
        'Sea Water Salinity (psu)',
        'Eastward Sea Water Velocity (m/s)',
        'Northward Sea Water Velocity (m/s)'
    ]
    
    # Define colormaps for each variable
    cmaps = ['plasma', 'viridis', 'RdBu_r', 'RdGy_r']
    
    # Identify which variables are velocity fields (need symmetric colorbar)
    velocity_indices = [2, 3]  # uo and vo
    
    # Plot each channel
    for i in range(4):
        channel_data = data[i]
        
        # Determine colorbar limits
        if i in velocity_indices:
            # For velocity variables: make colorbar symmetric around zero
            vmax = max(abs(channel_data.min()), abs(channel_data.max()))
            vmin = -vmax
            if vmax == 0:
                vmin, vmax = -1, 1
        else:
            # For non-velocity variables: use data range
            vmin, vmax = channel_data.min(), channel_data.max()
            if vmin == vmax:
                vmin, vmax = vmin - 0.5, vmax + 0.5
        
        # Create the plot
        im = axes[i].imshow(channel_data, cmap=cmaps[i], 
                           vmin=vmin, vmax=vmax, aspect='auto', origin='lower')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[i], shrink=0.8)
        
        # Extract units for colorbar label
        if '(' in titles[i] and ')' in titles[i]:
            units = titles[i].split('(')[-1].split(')')[0]
        else:
            units = variables[i]
        cbar.set_label(units, rotation=90)
        
        # Set title
        axes[i].set_title(f"{titles[i]}\n{variables[i]}", fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Width')
        axes[i].set_ylabel('Height')
        
        # Add statistics text box
        stats_text = f"Min: {channel_data.min():.4f}\nMax: {channel_data.max():.4f}\nMean: {channel_data.mean():.4f}\nStd: {channel_data.std():.4f}"
        axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add grid
        axes[i].grid(alpha=0.2, linestyle='--', color='white')
    
    # Add super title
    fig.suptitle(f'{title_prefix} - Ocean Variables (Shape: {data.shape})', 
                fontsize=16, fontweight='bold')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', 
                exist_ok=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Tensor plot saved successfully to: {output_path}")


def plot_argo_3d(argo_data: dict, 
                 output_path: str = "argo_3d_plot.png",
                 figsize: tuple = (10, 5),
                 dpi: int = 300,
                 vmin_vmax: dict = None) -> None:
    """
    Visualize Argo profiles in 3D (Longitude, Latitude, Depth) colored by Temperature and Salinity.
    
    Args:
        argo_data (dict): Dictionary containing Argo data with keys:
                          'temperature', 'salinity', 'latitude', 'longitude', 'depth'
        output_path (str): Path where the plot image will be saved
        figsize (tuple): Figure size (width, height) in inches
        dpi (int): Resolution of the output image
        vmin_vmax (dict, optional): Dictionary containing vmin/vmax for each variable.
                                    Format: {'thetao': (vmin, vmax), 'so': (vmin, vmax)}
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Check if data is empty
    if not argo_data or argo_data['temperature'].shape[0] == 0:
        print("Warning: No Argo data to plot")
        return

    # Extract data
    temps = argo_data['temperature']  # (N_profiles, N_depths)
    salins = argo_data['salinity']    # (N_profiles, N_depths)
    lats = argo_data['latitude']      # (N_profiles)
    lons = argo_data['longitude']     # (N_profiles)
    depths = argo_data['depth']       # (N_depths)
    
    # Convert to numpy if tensors
    if torch.is_tensor(temps): temps = temps.cpu().numpy()
    if torch.is_tensor(salins): salins = salins.cpu().numpy()
    if torch.is_tensor(lats): lats = lats.cpu().numpy()
    if torch.is_tensor(lons): lons = lons.cpu().numpy()
    if torch.is_tensor(depths): depths = depths.cpu().numpy()
    
    # Prepare coordinates for scatter plot
    # We need to repeat lat/lon for each depth level
    n_profiles, n_depths = temps.shape
    
    # Meshgrid-like expansion
    # depths_mesh: (N_profiles, N_depths)
    depths_mesh = np.tile(depths, (n_profiles, 1))
    
    # lats_mesh: (N_profiles, N_depths)
    lats_mesh = np.tile(lats[:, np.newaxis], (1, n_depths))
    
    # lons_mesh: (N_profiles, N_depths)
    lons_mesh = np.tile(lons[:, np.newaxis], (1, n_depths))
    
    # Flatten arrays for scatter plot
    flat_lons = lons_mesh.flatten()
    flat_lats = lats_mesh.flatten()
    flat_depths = depths_mesh.flatten()
    flat_temps = temps.flatten()
    flat_salins = salins.flatten()
    
    # Filter out NaN values
    valid_mask = ~np.isnan(flat_temps) & ~np.isnan(flat_salins)
    
    if not np.any(valid_mask):
        print("Warning: No valid Argo data to plot (all NaNs)")
        return
        
    flat_lons = flat_lons[valid_mask]
    flat_lats = flat_lats[valid_mask]
    flat_depths = flat_depths[valid_mask]
    flat_temps = flat_temps[valid_mask]
    flat_salins = flat_salins[valid_mask]
    
    # Create figure with 2 subplots (Temperature and Salinity)
    fig = plt.figure(figsize=figsize)
    
    # Determine colorbar limits
    # Temperature: use vmin_vmax['thetao'] if provided, otherwise use data range
    if vmin_vmax and 'thetao' in vmin_vmax:
        temp_vmin, temp_vmax = vmin_vmax['thetao']
    else:
        temp_vmin, temp_vmax = np.nanmin(flat_temps), np.nanmax(flat_temps)
    
    # Salinity: use vmin_vmax['so'] if provided, otherwise use data range
    if vmin_vmax and 'so' in vmin_vmax:
        sal_vmin, sal_vmax = vmin_vmax['so']
    else:
        sal_vmin, sal_vmax = np.nanmin(flat_salins), np.nanmax(flat_salins)
    
    # Calculate axis limits for consistent scaling
    lon_range = flat_lons.max() - flat_lons.min() if len(flat_lons) > 0 else 1
    lat_range = flat_lats.max() - flat_lats.min() if len(flat_lats) > 0 else 1
    depth_range = flat_depths.max() - flat_depths.min() if len(flat_depths) > 0 else 1
    
    # Plot 1: Temperature
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(flat_lons, flat_lats, -flat_depths, c=flat_temps, cmap='plasma', 
                           s=1, alpha=1, vmin=temp_vmin, vmax=temp_vmax, linewidths=0)
    ax1.set_title('Argo Temperature (°C)', fontweight='bold')
    ax1.set_xlabel('Longitude', labelpad=10)
    ax1.set_ylabel('Latitude')
    ax1.set_zlabel('Depth (m)')
    # Set coordinate limits to global Earth range
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-90, 90)
    ax1.set_zlim(-2000, 0)
    # Set custom tick labels with N/S/E/W format
    ax1.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax1.set_xticklabels(['180°W', '120°W', '60°W', '0°', '60°E', '120°E', '180°E'])
    ax1.set_yticks([-45, 0, 45])
    ax1.set_yticklabels(['45°S', '0°', '45°N'])
    # Set aspect ratio to make horizontal dimensions much larger than vertical
    ax1.set_box_aspect([12, 8, 2])
    # Set z-axis ticks with larger intervals
    ax1.set_zticks([-2000, -1000, 0])
    cbar1 = fig.colorbar(scatter1, ax=ax1, shrink=0.6, label='Temperature (°C)', 
                         orientation='vertical', pad=0.05, extend='both')
    cbar1.ax.set_ylabel('Temperature (°C)', fontsize=10)
    
    # Plot 2: Salinity
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(flat_lons, flat_lats, -flat_depths, c=flat_salins, cmap='viridis', 
                           s=1, alpha=1, vmin=sal_vmin, vmax=sal_vmax, linewidths=0)
    ax2.set_title('Argo Salinity (psu)', fontweight='bold')
    ax2.set_xlabel('Longitude', labelpad=10)
    ax2.set_ylabel('Latitude')
    ax2.set_zlabel('Depth (m)')
    # Set coordinate limits to global Earth range
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(-90, 90)
    ax2.set_zlim(-2000, 0)
    # Set custom tick labels with N/S/E/W format
    ax2.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax2.set_xticklabels(['180°W', '120°W', '60°W', '0°', '60°E', '120°E', '180°E'])
    ax2.set_yticks([-45, 0, 45])
    ax2.set_yticklabels(['45°S', '0°', '45°N'])
    # Set aspect ratio to make horizontal dimensions much larger than vertical
    ax2.set_box_aspect([12, 8, 2])
    # Set z-axis ticks with larger intervals
    ax2.set_zticks([-2000, -1000, 0])
    cbar2 = fig.colorbar(scatter2, ax=ax2, shrink=0.6, label='Salinity (psu)', 
                         orientation='vertical', pad=0.05, extend='both')
    cbar2.ax.set_ylabel('Salinity (psu)', fontsize=10)
    
    plt.suptitle(f'Argo Profiles 3D Visualization ({n_profiles} profiles)', fontsize=16, y=0.83)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', 
                exist_ok=True)
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Argo 3D plot saved successfully to: {output_path}")


def plot_argo_nc_profiles(ds: xr.Dataset,
                          output_path: str,
                          figsize: tuple = (2.5, 6),
                          dpi: int = 900) -> None:
    """
    Plot temperature and salinity profiles from Argo profile Dataset.
    JASA style: clean, professional, minimal design. Both profiles in one figure.
    
    Args:
        ds (xr.Dataset): Argo profile xarray Dataset
        output_path (str): Output image save path
        figsize (tuple): Figure size (width, height) in inches
        dpi (int): Output image resolution
    
    Returns:
        None: Saves the plot to the specified path
    """
    try:
        # JASA style rcParams
        matplotlib.rcParams['font.family'] = 'serif'
        matplotlib.rcParams['font.serif'] = ['Times New Roman']
        matplotlib.rcParams['axes.labelsize'] = 11
        matplotlib.rcParams['axes.titlesize'] = 12
        matplotlib.rcParams['xtick.labelsize'] = 10
        matplotlib.rcParams['ytick.labelsize'] = 10
        
        # Extract pressure, temperature, salinity data
        pres = ds['PRES_ADJUSTED_decibar'].values if 'PRES_ADJUSTED_decibar' in ds else None
        temp = ds['TEMP_ADJUSTED_degree_Celsius'].values if 'TEMP_ADJUSTED_degree_Celsius' in ds else None
        psal = ds['PSAL_ADJUSTED_psu'].values if 'PSAL_ADJUSTED_psu' in ds else None
        
        # Check if any variable is None and print dataset for debugging
        if pres is None or temp is None or psal is None:
            print("Warning: Missing required variables in dataset")
            print("pres is None:", pres is None)
            print("temp is None:", temp is None)
            print("psal is None:", psal is None)
            print("Dataset contents:")
            print(ds)
        
        # Sort data by pressure (ascending order) for proper line plotting
        if pres is not None:
            valid_idx = ~np.isnan(pres)
            pres_valid = pres[valid_idx]
            sort_idx = np.argsort(pres_valid)
            pres = pres_valid[sort_idx]
            
            if temp is not None:
                temp_valid = temp[valid_idx]
                temp = temp_valid[sort_idx]
            if psal is not None:
                psal_valid = psal[valid_idx]
                psal = psal_valid[sort_idx]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create single figure with twin x-axes
        fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor('white')
        
        # Get profile metadata
        profile_time = ds.attrs.get('profile_time', 'Unknown')
        latitude = ds['LATITUDE_degree_north'].values if 'LATITUDE_degree_north' in ds else None
        longitude = ds['LONGITUDE_degree_east'].values if 'LONGITUDE_degree_east' in ds else None
        platform = ds['PLATFORM_CODE'].values if 'PLATFORM_CODE' in ds else 'Unknown'
        
        # JASA style colors
        temp_color = '#1f77b4'  # Steel blue
        sal_color = '#8B0000'   # Dark red
        
        # Plot temperature on left axis
        if temp is not None and pres is not None and len(pres) > 0:
            ax1.plot(temp, pres, '-', linewidth=1.2, color=temp_color,
                    marker='o', markersize=1.5, markerfacecolor=temp_color,
                    label='Temperature')
            ax1.set_xlabel('Temperature (°C)', color=temp_color, labelpad=8)
            ax1.tick_params(axis='x', labelcolor=temp_color)
        
        ax1.set_ylabel('Pressure (dbar)', labelpad=8)
        ax1.set_ylim(2000, 0)  # Fixed y-axis range: 0-2000 dbar (inverted)
        
        # Create second x-axis for salinity
        ax2 = ax1.twiny()
        
        # Plot salinity on right axis (top x-axis)
        if psal is not None and pres is not None and len(pres) > 0:
            ax2.plot(psal, pres, '-', linewidth=1.2, color=sal_color,
                    marker='s', markersize=1.5, markerfacecolor=sal_color,
                    label='Salinity')
            ax2.set_xlabel('Salinity (psu)', color=sal_color, labelpad=8)
            ax2.tick_params(axis='x', labelcolor=sal_color)
        
        # Clean grid - only horizontal, subtle
        ax1.grid(True, alpha=0.4, linestyle='-', linewidth=0.5,
                color='gray', axis='y')
        ax1.set_axisbelow(True)
        
        # Remove top and right spines for ax1
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_linewidth(0.8)
        ax1.spines['bottom'].set_linewidth(0.8)
        ax1.spines['bottom'].set_color(temp_color)
        
        # Style the top spine (salinity axis)
        ax2.spines['top'].set_visible(True)
        ax2.spines['top'].set_linewidth(0.8)
        ax2.spines['top'].set_color(sal_color)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right',
                  frameon=True, fancybox=False, edgecolor='gray', framealpha=0.95)
        
        # Get profile length (number of sampling points)
        profile_length = len(pres) if pres is not None else 0
        
        # Add title with multi-line format
        title_text = f'Argo Profile - Platform: {platform}\n'
        if latitude is not None and longitude is not None:
            title_text += f'Location: ({latitude:.2f}°N, {longitude:.2f}°E)\nTime: {profile_time}\nProfile Length: {profile_length}'
        else:
            title_text += f'Time: {profile_time}\nProfile Length: {profile_length}'
        fig.suptitle(title_text, fontsize=10, fontweight='normal')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white',
                   edgecolor='none', pad_inches=0.15)
        plt.close()
        
        print(f"Argo profile plot saved successfully to: {output_path}")
        
    except Exception as e:
        print(f"Error plotting Argo profile: {str(e)}")


def plot_argo_daily_profile_counts(daily_counts: dict,
                                   output_path: str = "argo_daily_profile_counts.png",
                                   figsize: tuple = (12, 5),
                                   dpi: int = 600) -> None:
    """
    Plot the distribution of Argo profile counts over time (daily).
    
    Args:
        daily_counts (dict): Dictionary with date strings as keys and profile counts as values.
                            Format: {'YYYY-MM-DD': count, ...}
        output_path (str): Path where the plot image will be saved
        figsize (tuple): Figure size (width, height) in inches
        dpi (int): Resolution of the output image
    
    Returns:
        None: Saves the plot to the specified output path
    """
    if not daily_counts:
        print("Warning: No daily count data to plot")
        return
    
    # Sort dates and extract counts
    sorted_dates = sorted(daily_counts.keys())
    dates = [datetime.strptime(date_str, '%Y-%m-%d') for date_str in sorted_dates]
    counts = [daily_counts[date_str] for date_str in sorted_dates]
    
    # Convert to numpy array for calculations
    counts_array = np.array(counts)
    
    # Calculate rolling mean (30-day window) for trend line
    window = 30
    rolling_mean = np.convolve(counts_array, np.ones(window)/window, mode='valid')
    rolling_dates = dates[window-1:]
    
    # Add statistical information
    mean_count = np.mean(counts)
    max_count = max(counts)
    min_count = min(counts)
    total_count = sum(counts)
    std_count = np.std(counts)
    
    # JASA style plotting
    import matplotlib
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Times New Roman']
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['axes.labelsize'] = 12
    matplotlib.rcParams['axes.titlesize'] = 13
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['ytick.labelsize'] = 10
    matplotlib.rcParams['legend.fontsize'] = 10
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot daily data as scatter - JASA style colors
    ax.scatter(dates, counts, s=6, color='#4682B4', alpha=0.35, 
               label='Daily Profile Count', zorder=2, edgecolors='none')
    
    # Plot rolling mean trend line - darker red
    ax.plot(rolling_dates, rolling_mean, linewidth=1.8, color='#8B0000', 
            label=f'{window}-Day Moving Average', zorder=3)
    
    # Fill area under the rolling mean curve - subtle
    ax.fill_between(rolling_dates, rolling_mean, alpha=0.12, color='#8B0000', zorder=1)
    
    # Add horizontal line for overall mean - dark green
    ax.axhline(y=mean_count, color='#006400', linestyle='--', linewidth=1.2, 
               alpha=0.8, label=f'Overall Mean: {mean_count:.0f}', zorder=2)
    
    # Add shaded region for mean ± std - very subtle
    ax.axhspan(mean_count - std_count, mean_count + std_count, 
               alpha=0.08, color='#006400', zorder=1)
    
    # Set title and labels - JASA style (normal weight, clean)
    ax.set_title('Argo Profile Counts Distribution Over Time', 
                fontweight='normal', pad=10)
    ax.set_xlabel('Date', labelpad=8)
    ax.set_ylabel('Number of Profiles', labelpad=8)
    
    # Clean grid - only horizontal, lighter
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='gray', axis='y')
    ax.set_axisbelow(True)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Set x-axis limits to 2012-2023
    from datetime import datetime as dt_module
    ax.set_xlim(dt_module(2012, 1, 1), dt_module(2023, 12, 31))
    
    # Format x-axis dates
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
    plt.setp(ax.xaxis.get_minorticklabels(), rotation=0, ha='center', fontsize=8)
    
    # Customize tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='x', which='minor', length=3)
    
    # Add legend - JASA style (upper right, simple frame)
    ax.legend(loc='upper left', frameon=True, fancybox=False, 
              edgecolor='gray', framealpha=0.95)
    
    # Add statistics text box - JASA style (clean, no fancy box)
    stats_text = f"Total Days: {len(dates):,}\n"
    stats_text += f"Total Profiles: {total_count:,}\n"
    stats_text += f"Mean: {mean_count:.1f} ± {std_count:.1f}\n"
    stats_text += f"Max: {max_count:,}\n"
    stats_text += f"Min: {min_count:,}"
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray',
                    alpha=0.95, linewidth=0.8))
    
    # Set y-axis limits with some padding
    y_min = max(0, min_count - 50)
    y_max = max_count + 30
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', 
                exist_ok=True)
    
    # Save the figure with high quality
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Argo daily profile counts plot saved successfully to: {output_path}")


def plot_ensemble_summary(ds_gt: xr.Dataset,
                          ds_sample: xr.Dataset,
                          ds_mean: xr.Dataset,
                          ds_std: xr.Dataset,
                          output_path: str = "ensemble_summary.png",
                          figsize: tuple = (12, 5.58),
                          dpi: int = 900,
                          vmin_vmax: dict = None,
                          std_vmin_vmax: dict = None,
                          title: str = None,
                          extent: tuple = None,
                          cbar_pad: float = 0.03) -> None:
    """
    Plot a comprehensive 4x4 grid showing:
    - Rows: 4 variables (thetao, so, uo, vo)
    - Columns: Ground Truth, Sample, Ensemble Mean, Ensemble Std
    
    Args:
        ds_gt (xr.Dataset): Ground truth dataset
        ds_sample (xr.Dataset): Single sample member dataset
        ds_mean (xr.Dataset): Ensemble mean dataset
        ds_std (xr.Dataset): Ensemble standard deviation dataset
        output_path (str): Path where the plot image will be saved
        figsize (tuple): Figure size (width, height) in inches
        dpi (int): Resolution of the output image
        vmin_vmax (dict, optional): Dictionary containing vmin/vmax for each variable
        std_vmin_vmax (dict, optional): Dictionary containing vmin/vmax for std plot for each variable.
                                       If provided, will be used for std column colorbar limits.
        title (str, optional): Custom title for the figure
        extent (tuple, optional): Map extent as (lon_min, lon_max, lat_min, lat_max).
                                 If None, uses global extent (-180, 180, -80, 90).
        cbar_pad (float): Padding between colorbar and plot, default 0.03.
    
    Returns:
        None: Saves the plot to the specified output path
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    # Set global font to Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    # Set thinner borders and ticks
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    
    # Extract the first time step for plotting
    ds_gt_plot = ds_gt.isel(time=0)
    ds_sample_plot = ds_sample.isel(time=0)
    ds_mean_plot = ds_mean.isel(time=0)
    ds_std_plot = ds_std.isel(time=0)
    
    # Create figure with 4x4 subplots
    fig, axes = plt.subplots(4, 4, figsize=figsize, constrained_layout=True,
                             subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Define variable names and their descriptive titles
    variables = ['thetao', 'so', 'uo', 'vo']
    titles = [
        'Sea Water Potential Temperature (°C)',
        'Sea Water Salinity (psu)',
        'Eastward Sea Water Velocity (m/s)',
        'Northward Sea Water Velocity (m/s)'
    ]
    
    # Define colormaps for each variable
    cmaps = ['plasma', 'viridis', 'RdBu_r', 'RdGy_r']
    
    # Define colorbar labels for each variable (variable name with units)
    cbar_labels = [
        'Temperature (°C)',
        'Salinity (psu)',
        'Eastward Vel. (m/s)',
        'Northward Vel. (m/s)'
    ]
    
    # Column labels
    column_labels = ['Ground Truth', 'Sample', 'Ensemble Mean', 'Ensemble Std']
    
    # Identify which variables are velocity fields
    velocity_vars = ['uo', 'vo']
    
    # Plot each row (variable)
    for row, (var, var_title, cmap) in enumerate(zip(variables, titles, cmaps)):
        # Get data for each column
        gt_data = ds_gt_plot[var]
        sample_data = ds_sample_plot[var]
        mean_data = ds_mean_plot[var]
        std_data = ds_std_plot[var]
        
        # Determine colorbar limits based on GT for consistency across first 3 columns
        if vmin_vmax and var in vmin_vmax:
            vmin, vmax = vmin_vmax[var]
        else:
            if var in velocity_vars:
                vmax = max(abs(gt_data.min()), abs(gt_data.max()))
                vmin = -vmax
                if vmax == 0:
                    vmin, vmax = -1, 1
            else:
                vmin = np.nanmin(gt_data.values) if not np.all(np.isnan(gt_data.values)) else 0
                vmax = np.nanmax(gt_data.values) if not np.all(np.isnan(gt_data.values)) else 1
                if vmin == vmax:
                    vmin, vmax = vmin - 0.5, vmax + 0.5
        
        # Get colorbar label for this variable
        cbar_label = cbar_labels[row]
        
        # Column 0: Ground Truth (no colorbar)
        im0 = axes[row, 0].pcolormesh(gt_data.longitude, gt_data.latitude, gt_data.values,
                                      cmap=cmap, shading='auto', vmin=vmin, vmax=vmax,
                                      transform=ccrs.PlateCarree())
        axes[row, 0].add_feature(cfeature.COASTLINE, linewidth=0.3)
        
        # Column 1: Sample (no colorbar)
        im1 = axes[row, 1].pcolormesh(sample_data.longitude, sample_data.latitude, sample_data.values,
                                      cmap=cmap, shading='auto', vmin=vmin, vmax=vmax,
                                      transform=ccrs.PlateCarree())
        axes[row, 1].add_feature(cfeature.COASTLINE, linewidth=0.3)
        
        # Column 2: Ensemble Mean (with colorbar)
        im2 = axes[row, 2].pcolormesh(mean_data.longitude, mean_data.latitude, mean_data.values,
                                      cmap=cmap, shading='auto', vmin=vmin, vmax=vmax,
                                      transform=ccrs.PlateCarree())
        axes[row, 2].add_feature(cfeature.COASTLINE, linewidth=0.3)
        cbar2 = plt.colorbar(im2, ax=axes[row, 2], shrink=1.0, extend='both', pad=cbar_pad)
        cbar2.set_label(cbar_label, rotation=90, fontsize=9, labelpad=1.5)
        cbar2.outline.set_linewidth(0.5)
        cbar2.ax.tick_params(width=0.5)
        
        # Column 3: Ensemble Std (with colorbar)
        if std_vmin_vmax and var in std_vmin_vmax:
            # Use provided std vmin/vmax (for regional plots)
            std_vmin, std_vmax = std_vmin_vmax[var]
        elif var in velocity_vars:
            # For velocity variables: make colorbar symmetric around zero
            std_vmax = np.nanpercentile(np.abs(std_data.values), 99.5) if not np.all(np.isnan(std_data.values)) else 1
            if std_vmax == 0:
                std_vmax = 1
            std_vmin = -std_vmax
        else:
            # For other variables: use 0 as minimum
            std_vmin = 0
            std_vmax = np.nanmax(std_data.values) if not np.all(np.isnan(std_data.values)) else 1
            if std_vmax == 0:
                std_vmax = 1
        
        im3 = axes[row, 3].pcolormesh(std_data.longitude, std_data.latitude, std_data.values,
                                      cmap=cmap, shading='auto', vmin=std_vmin, vmax=std_vmax,
                                      transform=ccrs.PlateCarree())
        axes[row, 3].add_feature(cfeature.COASTLINE, linewidth=0.3)
        cbar3 = plt.colorbar(im3, ax=axes[row, 3], shrink=1.0, extend='both', pad=cbar_pad)
        cbar3.set_label(cbar_label, rotation=90, fontsize=9, labelpad=1.5)
        cbar3.outline.set_linewidth(0.5)
        cbar3.ax.tick_params(width=0.5)
        
        # Set row title (variable name) on the leftmost subplot
        axes[row, 0].set_ylabel(var_title.split('(')[0].strip(), fontsize=11, fontweight='bold')
        
        # Set extent for all subplots in this row
        if extent is not None:
            lon_min, lon_max, lat_min, lat_max = extent
        else:
            lon_min, lon_max, lat_min, lat_max = -180, 180, -80, 90
        
        for col in range(4):
            axes[row, col].set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            # Add gridlines
            # Use finer longitude intervals for regional plots
            if extent is not None:
                # For regional plots, use 25-degree intervals
                xlocs = np.arange(-180, 181, 25)
                # Filter to only include lines within the extent (with some margin)
                xlocs = xlocs[(xlocs >= lon_min - 25) & (xlocs <= lon_max + 25)]
            else:
                xlocs = [-150, -100, -50, 0, 50, 100, 150]
            gl = axes[row, col].gridlines(draw_labels=True, alpha=0.3, linestyle='--',
                                          xlocs=xlocs)
            gl.top_labels = False
            gl.right_labels = False
            # Only show left labels on the first column
            if col > 0:
                gl.left_labels = False
            # Only show bottom labels on the last row
            if row < 3:
                gl.bottom_labels = False
            gl.xlabel_style = {'size': 8}
            gl.ylabel_style = {'size': 8}
    
    # Add column labels at the top
    for col, label in enumerate(column_labels):
        axes[0, col].set_title(label, fontsize=13, fontweight='bold', pad=10)
    
    # Add super title
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.',
                exist_ok=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Ensemble summary plot saved successfully to: {output_path}")


def plot_spectral_power(gt_data: np.ndarray,
                        sample_data: np.ndarray,
                        ensemble_mean_data: np.ndarray,
                        variables: list,
                        output_path: str = "spectral_power.png",
                        figsize: tuple = (16, 4),
                        dpi: int = 300,
                        title: str = None,
                        extent: tuple = None,
                        lat_coords: np.ndarray = None,
                        lon_coords: np.ndarray = None) -> None:
    """
    Plot spectral power analysis for 4 variables in a 2x2 grid.
    Each subplot shows spectral power vs wavelength for GT, sample, and ensemble mean.
    
    Args:
        gt_data (np.ndarray): Ground truth data with shape (C, H, W)
        sample_data (np.ndarray): Single sample data with shape (C, H, W)
        ensemble_mean_data (np.ndarray): Ensemble mean data with shape (C, H, W)
        variables (list): List of variable names
        output_path (str): Path where the plot image will be saved
        figsize (tuple): Figure size (width, height) in inches
        dpi (int): Resolution of the output image
        title (str, optional): Custom title for the figure
        extent (tuple, optional): Region extent as (lon_min, lon_max, lat_min, lat_max).
                                 If provided, will crop data to this region before computing spectrum.
        lat_coords (np.ndarray, optional): Latitude coordinates for the data
        lon_coords (np.ndarray, optional): Longitude coordinates for the data
    
    Returns:
        None: Saves the plot to the specified output path
    """
    # Set font before importing pyplot
    import matplotlib
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Times New Roman']
    matplotlib.rcParams['mathtext.fontset'] = 'stix'  # For math expressions like 10^n
    
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import fftpack
    
    # JASA style configuration
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['xtick.major.width'] = 0.6
    plt.rcParams['ytick.major.width'] = 0.6
    plt.rcParams['xtick.minor.width'] = 0.4
    plt.rcParams['ytick.minor.width'] = 0.4
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.edgecolor'] = 'gray'
    
    # Crop data to region if extent is provided
    if extent is not None and lat_coords is not None and lon_coords is not None:
        lon_min, lon_max, lat_min, lat_max = extent
        
        # Find indices for the region
        lat_mask = (lat_coords >= lat_min) & (lat_coords <= lat_max)
        lon_mask = (lon_coords >= lon_min) & (lon_coords <= lon_max)
        
        # Crop data
        gt_data = gt_data[:, :, lon_mask][:, lat_mask, :]
        sample_data = sample_data[:, :, lon_mask][:, lat_mask, :]
        ensemble_mean_data = ensemble_mean_data[:, :, lon_mask][:, lat_mask, :]
    
    # Create figure with 1x4 subplots (horizontal layout)
    fig, axes = plt.subplots(1, 4, figsize=figsize, constrained_layout=True)
    axes = axes.ravel()
    
    # Define variable titles (concise version)
    var_titles = {
        'thetao': 'Temperature',
        'so': 'Salinity',
        'uo': 'Eastward Velocity',
        'vo': 'Northward Velocity'
    }
    
    # Colors for lines (academic color palette)
    colors = {
        'gt': '#d93026',      # Red for GT (ground truth)
        'mean': '#4a7dba',    # Deep blue for ensemble mean
        'sample': '#8fc7de'   # Light blue for sample
    }
    
    # Line styles for distinction
    line_styles = {
        'gt': '-',      # Solid line
        'sample': '--',  # Dashed line
        'mean': '-.'     # Dash-dot line
    }
    
    # Define line labels
    line_labels = {
        'gt': 'Ground Truth',
        'sample': 'Sample',
        'mean': 'Ensemble Mean'
    }
    
    def compute_spectral_power(data_2d):
        """Compute 1D spectral power spectrum from 2D data."""
        # Remove NaN values and fill with mean
        valid_mask = ~np.isnan(data_2d)
        if np.sum(valid_mask) == 0:
            return None, None
        
        data_filled = data_2d.copy()
        data_filled[~valid_mask] = np.nanmean(data_2d[valid_mask])
        
        # Compute 2D FFT
        fft_2d = fftpack.fft2(data_filled)
        fft_shifted = fftpack.fftshift(fft_2d)
        
        # Compute power spectrum
        power = np.abs(fft_shifted) ** 2
        
        # Get dimensions
        H, W = power.shape
        
        # Compute radial wavenumbers
        kx = fftpack.fftfreq(W, d=1.0)
        ky = fftpack.fftfreq(H, d=1.0)
        kx = fftpack.fftshift(kx)
        ky = fftpack.fftshift(ky)
        
        # Create 2D wavenumber grid
        KX, KY = np.meshgrid(kx, ky)
        k_radial = np.sqrt(KX**2 + KY**2)
        
        # Bin the power spectrum by radial wavenumber
        k_bins = np.linspace(0, np.max(k_radial), min(H, W) // 2)
        power_1d = []
        k_centers = []
        
        for i in range(len(k_bins) - 1):
            mask = (k_radial >= k_bins[i]) & (k_radial < k_bins[i + 1])
            if np.sum(mask) > 0:
                power_1d.append(np.mean(power[mask]))
                k_centers.append((k_bins[i] + k_bins[i + 1]) / 2)
        
        power_1d = np.array(power_1d)
        k_centers = np.array(k_centers)
        
        # Filter out zero wavenumber and invalid values
        valid = (k_centers > 0) & (power_1d > 0) & np.isfinite(power_1d)
        k_centers = k_centers[valid]
        power_1d = power_1d[valid]
        
        if len(k_centers) == 0:
            return None, None
        
        # Convert wavenumber to wavelength (in km)
        # Grid resolution is 0.5 degrees per grid cell
        # At equator: 1 degree longitude ≈ 111.32 km, so 0.5° ≈ 55.66 km
        grid_spacing_km = 55.0  # km per 0.5 degree grid cell
        # Wavelength = 1/k × grid_spacing
        # k from fftfreq is in cycles per grid point, so we multiply by grid_spacing to get km
        wavelength_km = 1.0 / k_centers * grid_spacing_km
        
        return wavelength_km, power_1d
    
    # Plot each variable
    for i, var in enumerate(variables[:4]):
        ax = axes[i]
        
        # Compute spectral power for each dataset
        wavelength_gt, power_gt = compute_spectral_power(gt_data[i])
        wavelength_sample, power_sample = compute_spectral_power(sample_data[i])
        wavelength_mean, power_mean = compute_spectral_power(ensemble_mean_data[i])
        
        # Plot lines (JASA style: monochrome with different line styles)
        if wavelength_gt is not None:
            ax.loglog(wavelength_gt, power_gt, color=colors['gt'], 
                     linestyle=line_styles['gt'], linewidth=1.0, 
                     label=line_labels['gt'], alpha=1.0)
        if wavelength_sample is not None:
            ax.loglog(wavelength_sample, power_sample, color=colors['sample'], 
                     linestyle=line_styles['sample'], linewidth=1.0, 
                     label=line_labels['sample'], alpha=1.0)
        if wavelength_mean is not None:
            ax.loglog(wavelength_mean, power_mean, color=colors['mean'], 
                     linestyle=line_styles['mean'], linewidth=1.0, 
                     label=line_labels['mean'], alpha=1.0)
        
        # Set title (JASA style: normal weight, concise)
        var_title = var_titles.get(var, var)
        ax.set_title(f"{var_title}", fontsize=11, fontweight='normal', pad=8)
        
        # Set labels (JASA style with Times New Roman)
        ax.set_xlabel('Wavelength (km)', fontsize=10, labelpad=5, fontname='Times New Roman')
        if i == 0:  # Only leftmost subplot has ylabel
            ax.set_ylabel('Spectral power', fontsize=10, labelpad=5, fontname='Times New Roman')
        
        # Configure axes (JASA style: subtle grid, clean spines)
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='gray', which='major')
        ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.3, color='gray', which='minor')
        ax.set_axisbelow(True)
        
        # Clean spines (JASA style: remove top and right)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        
        # Invert x-axis to match reference (large wavelength on left)
        ax.invert_xaxis()
        
        # Set tick formatter to show as 10^n (only for valid positive values)
        def format_log_tick(x, pos):
            if x > 0:
                exp = int(np.log10(x))
                return f'$10^{{{exp}}}$'
            return ''
        
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_log_tick))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_log_tick))
        
        # Set axis labels
        ax.set_xlabel('Wavelength (km)', fontsize=10)
        if i == 0:
            ax.set_ylabel('Spectral Power', fontsize=10)
    
    # Add shared legend below all subplots
    # Get handles and labels from the first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, 
               frameon=True, fancybox=False, edgecolor='gray', 
               framealpha=0.95, bbox_to_anchor=(0.5, -0.11))
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.21)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.',
                exist_ok=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Spectral power plot saved successfully to: {output_path}")


def plot_argo_spatial_distribution(latitudes: np.ndarray,
                                   longitudes: np.ndarray,
                                   output_path: str = "argo_spatial_distribution.png",
                                   figsize: tuple = (10, 6),
                                   dpi: int = 300,
                                   title: str = None,
                                   year: int = None,
                                   grid_size: tuple = (180, 360)) -> None:
    """
    Plot the spatial distribution of Argo profiles on a world map with density heatmap.
    
    Args:
        latitudes (np.ndarray): Array of latitude values
        longitudes (np.ndarray): Array of longitude values
        output_path (str): Path where the plot image will be saved
        figsize (tuple): Figure size (width, height) in inches
        dpi (int): Resolution of the output image
        title (str, optional): Custom title for the figure
        year (int, optional): Year of the data (for title)
        grid_size (tuple): Grid size for density calculation (lat_bins, lon_bins)
    
    Returns:
        None: Saves the plot to the specified output path
    """
    if len(latitudes) == 0 or len(longitudes) == 0:
        print("Warning: No spatial data to plot")
        return
    
    # Convert to numpy if tensors
    if torch.is_tensor(latitudes):
        latitudes = latitudes.cpu().numpy()
    if torch.is_tensor(longitudes):
        longitudes = longitudes.cpu().numpy()
    
    # Flatten arrays
    lats = np.array(latitudes).flatten()
    lons = np.array(longitudes).flatten()
    
    # Remove NaN values
    valid_mask = ~(np.isnan(lats) | np.isnan(lons))
    lats = lats[valid_mask]
    lons = lons[valid_mask]
    
    if len(lats) == 0:
        print("Warning: No valid spatial data after removing NaNs")
        return
    
    # JASA style plotting - 与 visualize_argo_spatial_distribution 保持一致
    # Set Times New Roman font for all text elements
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Times New Roman']
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['axes.titlesize'] = 11
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['ytick.labelsize'] = 10
    
    # Create figure with cartopy projection (PlateCarree for rectangular map)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=0))
    
    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none', alpha=0.5, zorder=2)
    # ax.add_feature(cfeature.OCEAN, facecolor='#F0F8FF', edgecolor='none', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#333333', zorder=3)
    
    # Create 2D histogram for density
    lon_bins = np.linspace(-180, 180, grid_size[1] + 1)
    lat_bins = np.linspace(-90, 90, grid_size[0] + 1)
    
    # Calculate 2D histogram (density)
    H, xedges, yedges = np.histogram2d(lons, lats, bins=[lon_bins, lat_bins])
    H = H.T  # Transpose to match imshow orientation
    
    # Create meshgrid for pcolormesh
    lon_centers = (xedges[:-1] + xedges[1:]) / 2
    lat_centers = (yedges[:-1] + yedges[1:]) / 2
    lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)
    
    # Plot density heatmap with actual counts
    im = ax.pcolormesh(lon_mesh, lat_mesh, H, 
                       transform=ccrs.PlateCarree(),
                       cmap='YlOrRd', 
                       shading='auto',
                       norm=matplotlib.colors.LogNorm(vmin=1, vmax=H.max()),
                       zorder=1)
    
    # Add colorbar with scientific notation
    cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02, format=matplotlib.ticker.LogFormatterSciNotation())
    cbar.set_label('Profile Count', rotation=90, labelpad=5, fontsize=10)
    
    # Add gridlines - JASA style (lighter)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.3, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    
    # Calculate statistics
    total_profiles = len(lats)
    lat_range = (np.min(lats), np.max(lats))
    lon_range = (np.min(lons), np.max(lons))
    
    # Set title - JASA style (normal weight, no year range)
    if title:
        ax.set_title(title, fontsize=11, fontweight='normal', pad=10)
    else:
        ax.set_title('Argo Profiles Spatial Distribution', 
                    fontsize=11, fontweight='normal', pad=10)
    
    # Add statistics text box - JASA style (clean, no fancy box)
    stats_text = f"Total Profiles: {total_profiles:,}\n"
    stats_text += f"Latitude Range: {lat_range[0]:.2f}° to {lat_range[1]:.2f}°\n"
    stats_text += f"Longitude Range: {lon_range[0]:.2f}° to {lon_range[1]:.2f}°"
    
    ax.text(0.98, 0.02, stats_text, 
           transform=ax.transAxes, 
           fontsize=9, 
           verticalalignment='bottom',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', 
                    edgecolor='gray', alpha=0.95, linewidth=0.8),
           zorder=4)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', 
                exist_ok=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white',
                edgecolor='none', pad_inches=0.15)
    plt.close()
    
    print(f"✅ Argo spatial distribution plot saved successfully to: {output_path}")


def plot_polar_ensemble_summary(ds_gt: xr.Dataset,
                                ds_sample: xr.Dataset,
                                ds_mean: xr.Dataset,
                                ds_std: xr.Dataset,
                                output_path: str = "polar_ensemble_summary.png",
                                figsize: tuple = (12, 5.58),
                                dpi: int = 900,
                                vmin_vmax: dict = None,
                                std_vmin_vmax: dict = None,
                                title: str = None,
                                extent: tuple = None,
                                cbar_pad: float = 0.03,
                                pole: str = 'north') -> None:
    """
    Plot a comprehensive 4x4 grid for polar regions using polar stereographic projection.
    
    Args:
        ds_gt (xr.Dataset): Ground truth dataset
        ds_sample (xr.Dataset): Single sample member dataset
        ds_mean (xr.Dataset): Ensemble mean dataset
        ds_std (xr.Dataset): Ensemble standard deviation dataset
        output_path (str): Path where the plot image will be saved
        figsize (tuple): Figure size (width, height) in inches
        dpi (int): Resolution of the output image
        vmin_vmax (dict, optional): Dictionary containing vmin/vmax for each variable
        std_vmin_vmax (dict, optional): Dictionary containing vmin/vmax for std plot for each variable.
        title (str, optional): Custom title for the figure
        extent (tuple, optional): Map extent as (lon_min, lon_max, lat_min, lat_max).
                                 If None, uses default polar extent.
        cbar_pad (float): Padding between colorbar and plot, default 0.03.
        pole (str): 'north' for North Polar Stereo, 'south' for South Polar Stereo.
    
    Returns:
        None: Saves the plot to the specified output path
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    # Set global font to Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    # Set thinner borders and ticks
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    
    # Extract the first time step for plotting
    ds_gt_plot = ds_gt.isel(time=0)
    ds_sample_plot = ds_sample.isel(time=0)
    ds_mean_plot = ds_mean.isel(time=0)
    ds_std_plot = ds_std.isel(time=0)
    
    # Select projection based on pole
    if pole == 'north':
        projection = ccrs.NorthPolarStereo()
        central_latitude = 90
        default_lat_min = 60
    else:  # south
        projection = ccrs.SouthPolarStereo()
        central_latitude = -90
        default_lat_min = -90
    
    # Create figure with 4x4 subplots
    fig, axes = plt.subplots(4, 4, figsize=figsize, constrained_layout=True,
                             subplot_kw={'projection': projection})
    
    # Define variable names and their descriptive titles
    variables = ['thetao', 'so', 'uo', 'vo']
    titles = [
        'Sea Water Potential Temperature (°C)',
        'Sea Water Salinity (psu)',
        'Eastward Sea Water Velocity (m/s)',
        'Northward Sea Water Velocity (m/s)'
    ]
    
    # Define colormaps for each variable
    cmaps = ['plasma', 'viridis', 'RdBu_r', 'RdGy_r']
    
    # Define colorbar labels for each variable (variable name with units)
    cbar_labels = [
        'Temperature (°C)',
        'Salinity (psu)',
        'Eastward Vel. (m/s)',
        'Northward Vel. (m/s)'
    ]
    
    # Column labels
    column_labels = ['Ground Truth', 'Sample', 'Ensemble Mean', 'Ensemble Std']
    
    # Identify which variables are velocity fields
    velocity_vars = ['uo', 'vo']
    
    # Determine extent
    if extent is not None:
        lon_min, lon_max, lat_min, lat_max = extent
    else:
        lon_min, lon_max = -180, 180
        if pole == 'north':
            lat_min, lat_max = 60, 90
        else:
            lat_min, lat_max = -90, -60
    
    # Plot each row (variable)
    for row, (var, var_title, cmap) in enumerate(zip(variables, titles, cmaps)):
        # Get data for each column
        gt_data = ds_gt_plot[var]
        sample_data = ds_sample_plot[var]
        mean_data = ds_mean_plot[var]
        std_data = ds_std_plot[var]
        
        # Determine colorbar limits based on GT for consistency across first 3 columns
        if vmin_vmax and var in vmin_vmax:
            vmin, vmax = vmin_vmax[var]
        else:
            if var in velocity_vars:
                vmax = max(abs(gt_data.min()), abs(gt_data.max()))
                vmin = -vmax
                if vmax == 0:
                    vmin, vmax = -1, 1
            else:
                vmin = np.nanmin(gt_data.values) if not np.all(np.isnan(gt_data.values)) else 0
                vmax = np.nanmax(gt_data.values) if not np.all(np.isnan(gt_data.values)) else 1
                if vmin == vmax:
                    vmin, vmax = vmin - 0.5, vmax + 0.5
        
        # Get colorbar label for this variable
        cbar_label = cbar_labels[row]
        
        # Column 0: Ground Truth (no colorbar)
        im0 = axes[row, 0].pcolormesh(gt_data.longitude, gt_data.latitude, gt_data.values,
                                      cmap=cmap, shading='auto', vmin=vmin, vmax=vmax,
                                      transform=ccrs.PlateCarree())
        axes[row, 0].add_feature(cfeature.COASTLINE, linewidth=0.3)
        axes[row, 0].add_feature(cfeature.LAND, facecolor='white', alpha=1.0)
        
        # Column 1: Sample (no colorbar)
        im1 = axes[row, 1].pcolormesh(sample_data.longitude, sample_data.latitude, sample_data.values,
                                      cmap=cmap, shading='auto', vmin=vmin, vmax=vmax,
                                      transform=ccrs.PlateCarree())
        axes[row, 1].add_feature(cfeature.COASTLINE, linewidth=0.3)
        axes[row, 1].add_feature(cfeature.LAND, facecolor='white', alpha=1.0)
        
        # Column 2: Ensemble Mean (with colorbar)
        im2 = axes[row, 2].pcolormesh(mean_data.longitude, mean_data.latitude, mean_data.values,
                                      cmap=cmap, shading='auto', vmin=vmin, vmax=vmax,
                                      transform=ccrs.PlateCarree())
        axes[row, 2].add_feature(cfeature.COASTLINE, linewidth=0.3)
        axes[row, 2].add_feature(cfeature.LAND, facecolor='white', alpha=1.0)
        cbar2 = plt.colorbar(im2, ax=axes[row, 2], shrink=1.0, extend='both', pad=cbar_pad)
        cbar2.set_label(cbar_label, rotation=90, fontsize=9, labelpad=1.5)
        cbar2.outline.set_linewidth(0.5)
        cbar2.ax.tick_params(width=0.5)
        
        # Column 3: Ensemble Std (with colorbar)
        if std_vmin_vmax and var in std_vmin_vmax:
            std_vmin, std_vmax = std_vmin_vmax[var]
        elif var in velocity_vars:
            std_vmax = np.nanpercentile(np.abs(std_data.values), 99.5) if not np.all(np.isnan(std_data.values)) else 1
            if std_vmax == 0:
                std_vmax = 1
            std_vmin = -std_vmax
        else:
            std_vmin = 0
            std_vmax = np.nanmax(std_data.values) if not np.all(np.isnan(std_data.values)) else 1
            if std_vmax == 0:
                std_vmax = 1
        
        im3 = axes[row, 3].pcolormesh(std_data.longitude, std_data.latitude, std_data.values,
                                      cmap=cmap, shading='auto', vmin=std_vmin, vmax=std_vmax,
                                      transform=ccrs.PlateCarree())
        axes[row, 3].add_feature(cfeature.COASTLINE, linewidth=0.3)
        axes[row, 3].add_feature(cfeature.LAND, facecolor='white', alpha=1.0)
        cbar3 = plt.colorbar(im3, ax=axes[row, 3], shrink=1.0, extend='both', pad=cbar_pad)
        cbar3.set_label(cbar_label, rotation=90, fontsize=9, labelpad=1.5)
        cbar3.outline.set_linewidth(0.5)
        cbar3.ax.tick_params(width=0.5)
        
        # Set row title (variable name) on the leftmost subplot
        axes[row, 0].set_ylabel(var_title.split('(')[0].strip(), fontsize=11, fontweight='bold')
        
        # Set extent for all subplots in this row
        for col in range(4):
            axes[row, col].set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            # Add gridlines
            gl = axes[row, col].gridlines(draw_labels=True, alpha=0.3, linestyle='--',
                                          xlocs=np.arange(-180, 181, 30))
            gl.top_labels = False
            gl.right_labels = False
            # Only show left labels on the first column
            if col > 0:
                gl.left_labels = False
            # Only show bottom labels on the last row
            if row < 3:
                gl.bottom_labels = False
            gl.xlabel_style = {'size': 8}
            gl.ylabel_style = {'size': 8}
    
    # Add column labels at the top
    for col, label in enumerate(column_labels):
        axes[0, col].set_title(label, fontsize=13, fontweight='bold', pad=10)
    
    # Add super title
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.',
                exist_ok=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Polar ensemble summary plot saved successfully to: {output_path}")


def plot_vertical_profiles_3d(
    gt_3d: dict,
    ensemble_mean_3d: dict,
    ensemble_std_3d: dict,
    sample_3d_list: list,
    error_3d: dict,
    lat_coords: np.ndarray,
    lon_coords: np.ndarray,
    depth_coords: np.ndarray,
    variables: list,
    output_path: str,
    date_str: str,
    lon_step: float = 20.0,
    figsize: tuple = (10, 8),
    dpi: int = 300
):
    """
    Plot vertical cross-sections (north-south profiles) in 3D for all variables.
    
    Creates individual 3D plots for each variable and plot type combination.
    Each plot displays multiple longitude slices in 3D space and is saved separately.
    
    Args:
        gt_3d (dict): Ground truth data {var: (depth, lat, lon) array}
        ensemble_mean_3d (dict): Ensemble mean data {var: (depth, lat, lon) array}
        ensemble_std_3d (dict): Ensemble std data {var: (depth, lat, lon) array}
        sample_3d_list (list): List of sample data dicts [{var: (depth, lat, lon)}, ...]
        error_3d (dict): Error data (mean - gt) {var: (depth, lat, lon) array}
        lat_coords (np.ndarray): Latitude coordinates (lat,)
        lon_coords (np.ndarray): Longitude coordinates (lon,)
        depth_coords (np.ndarray): Depth coordinates (depth,) - should be negative values
        variables (list): List of variable names to plot
        output_path (str): Base path where the plot images will be saved
        date_str (str): Date string for the title (format: YYYYMMDD)
        lon_step (float): Longitude interval for cross-sections, default 20 degrees
        figsize (tuple): Figure size (width, height) in inches for individual plots
        dpi (int): Resolution of the output image
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Define plot types and their properties
    plot_types = ['Ground Truth', 'Sample 1', 'Sample 2', 'Ensemble Mean', 'Ensemble Std', 'Error']
    n_depths = len(depth_coords)
    
    # Define longitude slices
    lon_slices = np.arange(-180, 181, lon_step)
    
    # Create output directory based on output_path
    output_dir = os.path.dirname(output_path)
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    viz_dir = os.path.join(output_dir, f'{base_name}_individual')
    os.makedirs(viz_dir, exist_ok=True)
    
    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    saved_files = []
    
    for var_idx, var in enumerate(variables):
        # Calculate color limits based on GT
        gt_var = gt_3d[var]
        if np.all(np.isnan(gt_var)):
            vmin, vmax = 0, 1
        else:
            if var == 'so':
                vmin = np.nanpercentile(gt_var, 0.5)
                vmax = np.nanpercentile(gt_var, 99.5)
            elif var in ['uo', 'vo']:
                vmax = np.nanpercentile(np.abs(gt_var), 99.5)
                vmin = -vmax
                if vmax == 0:
                    vmin, vmax = -1, 1
            else:
                vmin = np.nanmin(gt_var)
                vmax = np.nanmax(gt_var)
                if vmin == vmax:
                    vmin, vmax = vmin - 0.5, vmax + 0.5
        
        # Error color limits (symmetric)
        err_var = error_3d[var]
        if np.all(np.isnan(err_var)):
            err_vmax = 1
        else:
            err_vmax = np.nanpercentile(np.abs(err_var), 99.5)
            if err_vmax == 0:
                err_vmax = 1
        err_vmin = -err_vmax
        
        # Std color limits (0 to max)
        std_var = ensemble_std_3d[var]
        if np.all(np.isnan(std_var)):
            std_vmax = 1
        else:
            std_vmax = np.nanpercentile(std_var, 99.5)
            if std_vmax == 0:
                std_vmax = 1
        
        for plot_type_idx, plot_type in enumerate(plot_types):
            # Create individual figure for each var + plot_type combination
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            # Select data and color settings based on plot type
            if plot_type == 'Ground Truth':
                data = gt_3d[var]
                plot_vmin, plot_vmax = vmin, vmax
                cmap = 'RdYlBu_r' if var == 'thetao' else 'viridis' if var == 'so' else 'RdBu_r'
            elif plot_type == 'Sample 1':
                data = sample_3d_list[0][var] if len(sample_3d_list) > 0 else gt_3d[var]
                plot_vmin, plot_vmax = vmin, vmax
                cmap = 'RdYlBu_r' if var == 'thetao' else 'viridis' if var == 'so' else 'RdBu_r'
            elif plot_type == 'Sample 2':
                data = sample_3d_list[1][var] if len(sample_3d_list) > 1 else gt_3d[var]
                plot_vmin, plot_vmax = vmin, vmax
                cmap = 'RdYlBu_r' if var == 'thetao' else 'viridis' if var == 'so' else 'RdBu_r'
            elif plot_type == 'Ensemble Mean':
                data = ensemble_mean_3d[var]
                plot_vmin, plot_vmax = vmin, vmax
                cmap = 'RdYlBu_r' if var == 'thetao' else 'viridis' if var == 'so' else 'RdBu_r'
            elif plot_type == 'Ensemble Std':
                data = ensemble_std_3d[var]
                plot_vmin, plot_vmax = 0, std_vmax
                cmap = 'YlOrRd'
            else:  # Error
                data = error_3d[var]
                plot_vmin, plot_vmax = err_vmin, err_vmax
                cmap = 'RdBu_r'
            
            # Plot vertical cross-sections at each longitude
            for lon_val in lon_slices:
                # Find closest longitude index
                lon_idx = np.argmin(np.abs(lon_coords - lon_val))
                
                # Extract cross-section (depth, lat)
                cross_section = data[:, :, lon_idx]
                n_lat = cross_section.shape[1]
                
                # Create meshgrid for this slice with matching dimensions
                # cross_section shape: (n_depths, n_lat)
                lat_mesh, depth_mesh = np.meshgrid(lat_coords[:n_lat], depth_coords)
                
                # Flatten for scatter plot
                cross_flat = cross_section.flatten()
                valid_mask = ~np.isnan(cross_flat)
                
                if np.any(valid_mask):
                    x_flat = np.full(np.sum(valid_mask), lon_val)
                    y_flat = lat_mesh.flatten()[valid_mask]
                    z_flat = depth_mesh.flatten()[valid_mask]
                    c_flat = cross_flat[valid_mask]
                    
                    ax.scatter(x_flat, y_flat, z_flat, c=c_flat, 
                              cmap=cmap, vmin=plot_vmin, vmax=plot_vmax,
                              s=2, alpha=0.8, linewidths=0)
            
            # Set labels and limits
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_zlabel('Depth Level')
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_zlim(depth_coords.min(), depth_coords.max())
            
            # Set title
            ax.set_title(f'{var} - {plot_type}\nDate: {formatted_date}, Lon step: {lon_step}°', 
                        fontsize=12, fontweight='bold')
            
            # Add colorbar
            mappable = plt.cm.ScalarMappable(cmap=cmap)
            mappable.set_clim(plot_vmin, plot_vmax)
            cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, pad=0.1)
            cbar.set_label(f'{var}', fontsize=10)
            
            # Save individual figure
            plot_type_clean = plot_type.replace(' ', '_').lower()
            individual_path = os.path.join(viz_dir, f'{var}_{plot_type_clean}_{date_str}.png')
            plt.tight_layout()
            plt.savefig(individual_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            plt.close()
            
            saved_files.append(individual_path)
            print(f"Saved: {individual_path}")
    
    print(f"\nAll {len(saved_files)} individual plots saved to: {viz_dir}")
