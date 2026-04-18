import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from modules.postprocess import SamplePostProcessor
import concurrent.futures
import joblib
from sklearn.isotonic import IsotonicRegression
from scipy.special import ndtr

# Optional PyTorch import for Monotonic MLP
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class MonotonicMLP(nn.Module):
    """
    Monotonic Multi-Layer Perceptron for isotonic regression.
    Uses non-negative weights to guarantee monotonicity.
    """
    def __init__(self, hidden_units=64):
        super(MonotonicMLP, self).__init__()
        # Define layers
        self.layer1 = nn.Linear(1, hidden_units)
        self.layer2 = nn.Linear(hidden_units, hidden_units)
        self.layer3 = nn.Linear(hidden_units, 1)
        
        # Initialize weights to positive values to help maintain monotonicity
        nn.init.uniform_(self.layer1.weight, 0.0, 0.1)
        nn.init.uniform_(self.layer2.weight, 0.0, 0.1)
        nn.init.uniform_(self.layer3.weight, 0.0, 0.1)

    def forward(self, x):
        # To ensure monotonicity, weights must be non-negative during forward pass
        # The clamp operation is the key to enforcing monotonicity
        w1 = torch.clamp(self.layer1.weight, min=0)
        b1 = self.layer1.bias
        x = torch.relu(torch.nn.functional.linear(x, w1, b1))
        
        w2 = torch.clamp(self.layer2.weight, min=0)
        b2 = self.layer2.bias
        x = torch.relu(torch.nn.functional.linear(x, w2, b2))
        
        w3 = torch.clamp(self.layer3.weight, min=0)
        b3 = self.layer3.bias
        # Output layer typically uses sigmoid to constrain output to [0, 1]
        x = torch.sigmoid(torch.nn.functional.linear(x, w3, b3))
        return x


def train_monotonic_nn(U_train, V_train, hidden_units=2048, epochs=1000, lr=0.001,
                       sample_weight=None, verbose=False, device=None, batch_size=None,
                       use_tqdm=True):
    """
    Train a Monotonic MLP for isotonic regression.

    Args:
        U_train: PIT values (input), shape (N,) or (N, 1)
        V_train: Target quantile values (target), shape (N,) or (N, 1)
        hidden_units: Number of hidden units in the MLP
        epochs: Number of training epochs
        lr: Learning rate
        sample_weight: Optional sample weights for weighted training
        verbose: Whether to print training progress
        device: Device to use for training ('cuda', 'cuda:0', 'cpu', etc.)
                If None, automatically selects CUDA if available, otherwise CPU
        batch_size: Batch size for training. If None, use full batch (original behavior)
        use_tqdm: Whether to use tqdm progress bar for batch iteration

    Returns:
        Trained MonotonicMLP model (detached from computation graph, on CPU)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for Monotonic MLP. "
                         "Please install it with: pip install torch")

    # Determine device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    if verbose:
        print(f"[MonotonicMLP] Training on device: {device}")
        print(f"[MonotonicMLP] Model config: hidden_units={hidden_units}, epochs={epochs}, lr={lr}")

    model = MonotonicMLP(hidden_units=hidden_units).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[MonotonicMLP] Total parameters: {total_params:,}")

    # Keep data in CPU memory, move to GPU batch by batch
    U_tensor = torch.FloatTensor(U_train).reshape(-1, 1)  # Stay on CPU
    V_tensor = torch.FloatTensor(V_train).reshape(-1, 1)  # Stay on CPU

    n_samples = len(U_tensor)

    # Handle sample weights (stay on CPU)
    if sample_weight is not None:
        weight_tensor = torch.FloatTensor(sample_weight).reshape(-1, 1)
        # Normalize weights to sum to 1
        weight_tensor = weight_tensor / weight_tensor.sum()
        criterion = nn.MSELoss(reduction='none')
        if verbose:
            print(f"[MonotonicMLP] Using sample weights (min={weight_tensor.min():.6f}, max={weight_tensor.max():.6f}, sum={weight_tensor.sum():.4f})")
    else:
        weight_tensor = None
        criterion = nn.MSELoss(reduction='mean')
        if verbose:
            print(f"[MonotonicMLP] No sample weights provided, using uniform weights")

    # Determine batch size
    if batch_size is None:
        batch_size = n_samples  # Full batch
        if verbose:
            print(f"[MonotonicMLP] Using full batch training (n_samples={n_samples:,})")
    else:
        batch_size = min(batch_size, n_samples)
        if verbose:
            print(f"[MonotonicMLP] Using mini-batch training (batch_size={batch_size:,}, n_samples={n_samples:,}, n_batches={(n_samples + batch_size - 1) // batch_size})")

    n_batches = (n_samples + batch_size - 1) // batch_size

    if verbose:
        print(f"[MonotonicMLP] Starting training for {epochs} epochs...")

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        # Shuffle all indices for each epoch
        if batch_size < n_samples:
            shuffled_indices = torch.randperm(n_samples)
        else:
            shuffled_indices = torch.arange(n_samples)

        # Mini-batch training with optional tqdm progress bar
        batch_iterator = range(n_batches)
        if use_tqdm and n_batches > 1:
            batch_iterator = tqdm(batch_iterator, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for i in batch_iterator:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)

            # Get shuffled indices for this batch
            indices = shuffled_indices[start_idx:end_idx]

            # Move batch data from CPU to GPU
            U_batch = U_tensor[indices].to(device)
            V_batch = V_tensor[indices].to(device)

            optimizer.zero_grad()
            outputs = model(U_batch)
            loss = criterion(outputs, V_batch)

            # Apply sample weights if provided
            if weight_tensor is not None:
                weight_batch = weight_tensor[indices].to(device)
                loss = (loss * weight_batch).mean()
                del weight_batch

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Explicitly release GPU memory (PyTorch manages memory automatically,
            # but explicit cleanup is safer for long training loops)
            del U_batch, V_batch, outputs, loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        avg_loss = epoch_loss / n_batches

        if verbose:
            print(f"[MonotonicMLP]   Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

    # Set to eval mode, move to CPU and return
    model.eval()
    model = model.cpu()

    if verbose:
        print(f"[MonotonicMLP] Training completed. Final loss: {avg_loss:.6f}")
        if device.type == 'cuda':
            print(f"[MonotonicMLP] Model moved back to CPU")

    return model


def predict_monotonic_nn(model, U, batch_size=None, device=None, verbose=False):
    """
    Predict using a trained Monotonic MLP with batch processing.
    
    Args:
        model: Trained MonotonicMLP model
        U: Input PIT values, shape (N,) or (N, 1)
        batch_size: Batch size for prediction. If None, use full batch.
                   For large datasets, recommend setting batch_size (e.g., 4096 or 8192).
        device: Device to use for prediction ('cuda', 'cuda:0', 'cpu', etc.)
                If None, automatically selects CUDA if available, otherwise CPU.
        verbose: Whether to print progress information.
        
    Returns:
        Predicted calibrated values, shape (N,)
    """
    import time
    
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for Monotonic MLP.")
    
    if verbose:
        print("=" * 60)
        print("[MonotonicMLP Predict] Starting prediction...")
        start_time = time.time()
    
    # Determine device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    if verbose:
        print(f"[MonotonicMLP Predict] Device: {device}")
        if device.type == 'cuda':
            print(f"[MonotonicMLP Predict] GPU: {torch.cuda.get_device_name(device)}")
            print(f"[MonotonicMLP Predict] GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[MonotonicMLP Predict] Model parameters: {total_params:,}")
    
    # Keep data on CPU, move to device batch by batch
    U_array = np.asarray(U).flatten()
    U_tensor = torch.FloatTensor(U_array).reshape(-1, 1)  # Stay on CPU
    n_samples = len(U_tensor)
    
    if verbose:
        print(f"[MonotonicMLP Predict] Input samples: {n_samples:,}")
        print(f"[MonotonicMLP Predict] Input range: [{U_array.min():.6f}, {U_array.max():.6f}]")
        print(f"[MonotonicMLP Predict] Input mean: {U_array.mean():.6f}, std: {U_array.std():.6f}")
    
    # Determine batch size
    if batch_size is None:
        batch_size = n_samples  # Full batch
        if verbose:
            print(f"[MonotonicMLP Predict] Using full batch (batch_size={n_samples:,})")
    else:
        batch_size = min(batch_size, n_samples)
        if verbose:
            print(f"[MonotonicMLP Predict] Using mini-batch (batch_size={batch_size:,})")
    
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    if verbose:
        print(f"[MonotonicMLP Predict] Total batches: {n_batches}")
        print("-" * 60)
    
    # Collect predictions
    predictions_list = []
    
    with torch.no_grad():
        batch_iterator = range(n_batches)
        if n_batches > 1:
            batch_iterator = tqdm(batch_iterator, desc="[MonotonicMLP Predict] Batches", leave=False)
        
        for i in batch_iterator:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            # Move batch to device
            U_batch = U_tensor[start_idx:end_idx].to(device)
            
            # Predict
            outputs = model(U_batch)
            
            # Move results back to CPU and store
            predictions_list.append(outputs.cpu().numpy())
            
            # Clean up GPU memory
            del U_batch, outputs
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    # Move model back to CPU
    model = model.cpu()
    
    # Concatenate all predictions
    predictions = np.concatenate(predictions_list, axis=0).flatten()
    
    if verbose:
        print("-" * 60)
        print(f"[MonotonicMLP Predict] Output range: [{predictions.min():.6f}, {predictions.max():.6f}]")
        print(f"[MonotonicMLP Predict] Output mean: {predictions.mean():.6f}, std: {predictions.std():.6f}")
        elapsed_time = time.time() - start_time
        print(f"[MonotonicMLP Predict] Completed in {elapsed_time:.2f} seconds")
        print(f"[MonotonicMLP Predict] Throughput: {n_samples / elapsed_time:.0f} samples/sec")
        print("=" * 60)
    
    return predictions

def _process_rank_histogram_group(args):
    """
    Worker function for processing a single ensemble group.
    Must be at module level for ProcessPoolExecutor pickling.
    """
    # Unpack arguments
    # New signature: base_id, file_paths, variables, ground_truth_dir, N, n_bins, return_ranks
    if len(args) == 5:
         # Legacy support or simple call
         base_id, file_paths, variables, ground_truth_dir, N = args
         n_bins = 20
         return_ranks = False
    elif len(args) == 6:
         # Ambiguous: could be (..., N, return_ranks) or (..., N, n_bins)
         # We'll assume the caller has been updated to pass n_bins
         if isinstance(args[5], bool):
             base_id, file_paths, variables, ground_truth_dir, N, return_ranks = args
             n_bins = 20
         else:
             base_id, file_paths, variables, ground_truth_dir, N, n_bins = args
             return_ranks = False
    else:
         base_id, file_paths, variables, ground_truth_dir, N, n_bins, return_ranks = args

    local_hist = {var: np.zeros(n_bins, dtype=int) for var in variables}
    local_ranks = {var: [] for var in variables} if return_ranks else None

    # Ensure we use the same N (skip if mismatch)
    if len(file_paths) != N:
        # Try to use the first N files if available, or skip
        if len(file_paths) > N:
            file_paths = sorted(file_paths)[:N]
        else:
            return None
        
    try:
        # 1. Load Ensemble Members
        members = []
        for fp in file_paths:
            if fp.endswith('.nc'):
                ds = xr.open_dataset(fp)
                if 'data' in ds:
                    data = ds['data'].values
                else:
                    # Try to concatenate variables if 'data' not present
                    vars_list = ['thetao', 'so', 'uo', 'vo']
                    present_vars = [v for v in vars_list if v in ds]
                    if present_vars:
                        data_list = [ds[v].values for v in present_vars]
                        processed_list = []
                        for d in data_list:
                            if d.ndim == 3: 
                                 processed_list.append(d[0])
                            else:
                                 processed_list.append(d)
                        data = np.stack(processed_list)
                    else:
                        ds.close()
                        raise ValueError(f"Could not find valid data in .nc file: {fp}")
                ds.close()
                members.append(data)
            else:
                data = np.load(fp)
                if fp.endswith('.npz'):
                    if hasattr(data, 'files'):
                        if 'data' in data.files:
                            data = data['data']
                        else:
                            raise ValueError(f"Key 'data' not found in .npz file: {fp}")
                members.append(data)

        ensemble = np.stack(members, axis=0) # (N, C, H, W)
        
        # 2. Load Ground Truth
        gt_filename = base_id + '.nc'
        gt_path = os.path.join(ground_truth_dir, gt_filename)
    
        if not os.path.exists(gt_path):
            return None
            
        # Load GT Data
        gt_data_list = []
        with xr.open_dataset(gt_path) as ds_gt:
            for var in variables:
                if var not in ds_gt:
                    continue
                data = ds_gt[var].values
                if data.ndim == 3 and data.shape[0] == 1:
                    data = data[0]
                gt_data_list.append(data)
        
        if len(gt_data_list) != len(variables):
            return None
            
        gt = np.array(gt_data_list) # (C, H, W)
        
        # 3. Align shapes (padding)
        if ensemble.shape[2:] != gt.shape[1:]:
                if ensemble.shape[2] > gt.shape[1]: # Height mismatch
                    pad_h = ensemble.shape[2] - gt.shape[1]
                    gt = np.pad(gt, ((0,0), (pad_h, 0), (0,0)), mode='constant', constant_values=np.nan)
        
        if ensemble.shape[2:] != gt.shape[1:]:
            return None

        # 4. Mask Land (NaNs in GT)
        mask = np.isnan(gt) # (C, H, W)
        
        # 5. Compute Stats for Continuous CDF (KDE)
        # We use Gaussian Kernel Density Estimation with Silverman's rule for bandwidth selection.
        # Bandwidth h = 1.06 * std * N^(-1/5)
        
        # Calculate standard deviation per pixel
        # Use ddof=1 for sample standard deviation
        sigma = np.std(ensemble, axis=0, ddof=1) # (C, H, W)
        
        # Calculate bandwidth per pixel
        # N is ensemble size
        h = 1.06 * sigma * (N ** (-0.2))
        
        # Avoid zero bandwidth (for identical ensemble members)
        h = np.maximum(h, 1e-6)
        
        # 6. Accumulate counts for valid points
        for i, var in enumerate(variables):
            # GT must be valid
            valid_mask = ~mask[i]
            
            # Ensemble must not have NaNs at these locations
            # Check if any ensemble member is NaN at each pixel
            ens_has_nan = np.isnan(ensemble[:, i]).any(axis=0)
            valid_mask = valid_mask & (~ens_has_nan)
            
            if not np.any(valid_mask):
                continue
                
            # Extract valid data
            # ensemble[:, i, valid_mask] -> (N, P) where P is number of valid points
            # We index carefully: first select variable i, then mask spatial dims
            ens_valid = ensemble[:, i][:, valid_mask] # (N, P)
            y = gt[i][valid_mask]   # (P,)
            bw = h[i][valid_mask]   # (P,)
            
            # Compute PIT values using vectorized KDE CDF
            # PIT = 1/N * sum(Phi((y - x_i)/h))
            
            # Expand dims for broadcasting
            # y: (P,) -> (1, P)
            # ens_valid: (N, P)
            # bw: (P,) -> (1, P)
            
            z = (y[None, :] - ens_valid) / bw[None, :] # (N, P)
            
            # Compute Gaussian CDF
            cdf_vals = ndtr(z) # (N, P)
            
            # Average over ensemble members to get the KDE estimate of P(Y <= y)
            pit_values = np.mean(cdf_vals, axis=0) # (P,)

            # Final check for NaNs in pit_values (should not happen if inputs are clean, but for safety)
            valid_pit_mask = np.isfinite(pit_values)
            if not np.all(valid_pit_mask):
                pit_values = pit_values[valid_pit_mask]
            
            if len(pit_values) == 0:
                continue

            # Accumulate histogram
            # Bins are 0 to 1
            counts, _ = np.histogram(pit_values, bins=n_bins, range=(0, 1))
            local_hist[var] += counts

            # Collect ranks (PIT values) if requested
            if return_ranks:
                local_ranks[var] = pit_values
        
        return {'hist': local_hist, 'ranks': local_ranks}
            
    except Exception as e:
        print(f"Error processing {base_id}: {e}")
        return None

class ConformalPredictor:
    def __init__(self, sample_dir: str, ground_truth_dir: str = './data/glorys12/test', variables=None, n_bins=200, num_workers=None, sample_ratio=1.0, sample_seed=42):
        self.processor = SamplePostProcessor(sample_dir=sample_dir, variables=variables)
        self.ground_truth_dir = ground_truth_dir
        self.variables = self.processor.variables
        self.sample_dir = sample_dir
        self.n_bins = n_bins
        self.num_workers = num_workers
        self.sample_ratio = sample_ratio
        self.sample_seed = sample_seed
        # Cache for computed results to avoid redundant calculations
        self._rank_histogram_cache = None  # Stores (rank_counts, all_ranks) tuple
        self._cache_metadata = None  # Stores (sample_dir, ground_truth_dir, n_bins, sample_ratio, sample_seed) for cache validation

    def _get_cache_path(self, calibrated=False):
        """Get the path for caching computed results.
        
        Args:
            calibrated (bool): If True, return path for calibrated histogram cache.
                              If False, return path for raw histogram cache.
        """
        cache_dir = os.path.join(self.processor.output_dir, 'conformal', '.cache')
        os.makedirs(cache_dir, exist_ok=True)
        # Create a unique cache key based on sample_dir and ground_truth_dir
        import hashlib
        cache_key = hashlib.md5(f"{self.sample_dir}:{self.ground_truth_dir}:{self.n_bins}".encode()).hexdigest()
        if calibrated:
            return os.path.join(cache_dir, f'calibrated_rank_histogram_cache_{cache_key}.pkl')
        return os.path.join(cache_dir, f'rank_histogram_cache_{cache_key}.pkl')

    def _is_cache_valid(self):
        """Check if the cached data is still valid based on current configuration."""
        current_metadata = (self.sample_dir, self.ground_truth_dir, self.n_bins, self.sample_ratio, self.sample_seed)
        return self._cache_metadata == current_metadata

    def _save_cache(self, rank_counts, all_ranks):
        """Save computed results to disk cache.
        
        Args:
            rank_counts: dict of binned histogram counts for each variable
            all_ranks: dict of raw PIT values, or None if using binned mode only
                      If None and cache already exists with all_ranks, the existing
                      all_ranks will be preserved.
        """
        cache_path = self._get_cache_path()
        
        # If all_ranks is None but cache exists with all_ranks, preserve it
        if all_ranks is None and os.path.exists(cache_path):
            try:
                existing_cache = joblib.load(cache_path)
                cached_metadata = existing_cache.get('metadata')
                current_metadata = (self.sample_dir, self.ground_truth_dir, self.n_bins, self.sample_ratio, self.sample_seed)
                if cached_metadata == current_metadata:
                    existing_ranks = existing_cache.get('all_ranks')
                    if existing_ranks is not None:
                        all_ranks = existing_ranks
                        print(f"Preserving existing all_ranks in cache")
            except Exception:
                pass  # If loading fails, just save without all_ranks
        
        cache_data = {
            'rank_counts': rank_counts,
            'all_ranks': all_ranks,
            'metadata': (self.sample_dir, self.ground_truth_dir, self.n_bins, self.sample_ratio, self.sample_seed)
        }
        joblib.dump(cache_data, cache_path)
        if all_ranks is not None:
            print(f"Cache saved to {cache_path} (with raw ranks)")
        else:
            print(f"Cache saved to {cache_path} (binned data only, memory-efficient)")

    def _load_cache(self):
        """Load computed results from disk cache if available and valid.
        
        Returns:
            tuple: (rank_counts, all_ranks) where either may be None if not in cache
        """
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            try:
                cache_data = joblib.load(cache_path)
                cached_metadata = cache_data.get('metadata')
                current_metadata = (self.sample_dir, self.ground_truth_dir, self.n_bins, self.sample_ratio, self.sample_seed)
                if cached_metadata == current_metadata:
                    print(f"Loaded valid cache from {cache_path}")
                    return cache_data.get('rank_counts'), cache_data.get('all_ranks')
                else:
                    print("Cache metadata mismatch, recomputing...")
            except Exception as e:
                print(f"Error loading cache: {e}, recomputing...")
        return None, None

    def _save_calibrated_cache(self, calibrated_counts, calibrated_all_ranks=None, model_path=None):
        """Save calibrated histogram results to disk cache.
        
        Args:
            calibrated_counts: dict of calibrated histogram counts for each variable
            calibrated_all_ranks: dict of calibrated raw PIT values for each variable (for KDE plotting)
            model_path: path to the calibration model used (for cache validation)
        """
        cache_path = self._get_cache_path(calibrated=True)
        cache_data = {
            'calibrated_rank_counts': calibrated_counts,
            'calibrated_all_ranks': calibrated_all_ranks,
            'metadata': (self.sample_dir, self.ground_truth_dir, self.n_bins, model_path)
        }
        joblib.dump(cache_data, cache_path)
        if calibrated_all_ranks is not None:
            print(f"Calibrated histogram cache saved to {cache_path} (with raw ranks)")
        else:
            print(f"Calibrated histogram cache saved to {cache_path} (binned data only)")

    def _load_calibrated_cache(self, model_path=None):
        """Load calibrated histogram results from disk cache if available and valid.
        
        Args:
            model_path: path to the calibration model used (for cache validation)
            
        Returns:
            tuple: (calibrated_rank_counts, calibrated_all_ranks) if cache is valid, (None, None) otherwise
        """
        cache_path = self._get_cache_path(calibrated=True)
        if os.path.exists(cache_path):
            try:
                cache_data = joblib.load(cache_path)
                cached_metadata = cache_data.get('metadata')
                current_metadata = (self.sample_dir, self.ground_truth_dir, self.n_bins, model_path)
                if cached_metadata == current_metadata:
                    print(f"Loaded valid calibrated histogram cache from {cache_path}")
                    return cache_data.get('calibrated_rank_counts'), cache_data.get('calibrated_all_ranks')
                else:
                    print("Calibrated cache metadata mismatch, recomputing...")
            except Exception as e:
                print(f"Error loading calibrated cache: {e}, recomputing...")
        return None, None

    def plot_rank_histogram(self, save_name="rank_histogram.png", use_cache=True, save_cache=True):
        """
        Calculate and plot the Rank Histogram (PIT Histogram) for each variable.
        Uses Gaussian KDE fit to continuous ensemble distribution.
        Sampling is applied here if sample_ratio < 1.0, and the sampled data is cached.
        All subsequent methods will use the cached sampled data.
        
        Args:
            save_name (str): Output filename for the plot.
            use_cache (bool): Whether to use cached results if available.
            save_cache (bool): Whether to save results to cache after computation.
        """
        import concurrent.futures

        # Try to load from cache first (need both counts and ranks for complete cache)
        if use_cache:
            cached_counts, cached_ranks = self._load_cache()
            if cached_counts is not None and cached_ranks is not None:
                rank_counts = cached_counts
                all_ranks = cached_ranks
                print("Using cached rank histogram data.")
                # Plotting
                from modules.plot.plot_conformal import plot_histograms
                save_path = os.path.join(os.path.dirname(self.sample_dir), 'conformal', save_name)
                plot_histograms(rank_counts, self.n_bins, self.variables, save_path, None)
                return

        # Get ensemble groups using the processor's helper
        ensemble_groups = self.processor._get_ensemble_groups()
        
        if not ensemble_groups:
            print("No ensemble groups found.")
            return

        # Determine ensemble size N from the first group
        first_group_files = next(iter(ensemble_groups.values()))
        N = len(first_group_files)
        print(f"Detected ensemble size: {N}")
        
        # Apply sampling to ensemble groups if sample_ratio < 1.0
        if self.sample_ratio < 1.0:
            import random
            random.seed(self.sample_seed)
            all_group_ids = list(ensemble_groups.keys())
            n_samples = max(1, int(len(all_group_ids) * self.sample_ratio))
            sampled_group_ids = random.sample(all_group_ids, n_samples)
            ensemble_groups = {gid: ensemble_groups[gid] for gid in sampled_group_ids}
            print(f"Sampled {n_samples} out of {len(all_group_ids)} ensemble groups ({self.sample_ratio*100:.1f}%) with seed {self.sample_seed}")
        
        # Define number of bins for PIT histogram
        
        # Rank counts: n_bins
        rank_counts = {var: np.zeros(self.n_bins, dtype=int) for var in self.variables}
        # Also collect ranks for cache (to share with fit_isotonic_calibration)
        all_ranks = {var: [] for var in self.variables}
        
        print(f"Computing PIT Histograms over {len(ensemble_groups)} time steps with multiprocessing...")
        
        # Prepare arguments for multiprocessing (with return_ranks=True to collect ranks)
        process_args = [
            (base_id, file_paths, self.variables, self.ground_truth_dir, N, self.n_bins, True) 
            for base_id, file_paths in ensemble_groups.items()
        ]

        # Execute in parallel using ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(_process_rank_histogram_group, arg) for arg in process_args]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Groups"):
                result = future.result()
                if result is not None:
                    hist_data = result['hist']
                    local_ranks = result['ranks']
                    for var in self.variables:
                        rank_counts[var] += hist_data[var]
                        if local_ranks[var] is not None and len(local_ranks[var]) > 0:
                            all_ranks[var].append(local_ranks[var])
                
        # Plotting
        from modules.plot.plot_conformal import plot_histograms
        save_path = os.path.join(os.path.dirname(self.sample_dir), 'conformal', save_name)
        plot_histograms(rank_counts, self.n_bins, self.variables, save_path, None)

        # Save to cache (with all_ranks for sharing with fit_isotonic_calibration)
        if save_cache:
            self._save_cache(rank_counts, all_ranks)

    def fit_isotonic_calibration(self, save_name="isotonic_calibration.pkl",
                                  use_binned=True, method='sklearn', verbose=True,
                                  batch_size=4096, epochs=1000):
        """
        Fit Isotonic Regression calibration for each variable using the calibration set.
        Follows the method: f* = argmin sum(f(U_(j)) - V_j)^2.
        Saves the model to the 'conformal' directory in the output path.
        This method always uses cached data from plot_rank_histogram().

        Args:
            save_name (str): Output filename for the calibration model.
            use_binned (bool): If True, use binned histogram data for memory-efficient
                              calibration. Recommended for large datasets (billions of points).
                              If False, use raw PIT values (more accurate but memory-intensive).
            method (str): Method to use for isotonic regression. Options:
                         - 'sklearn': Use sklearn.isotonic.IsotonicRegression (default)
                         - 'mlp': Use Monotonic MLP (requires PyTorch)
            verbose (bool): Whether to print detailed progress information.
            batch_size (int): Batch size for MLP training. Default is 4096.
            epochs (int): Number of training epochs for MLP. Default is 1000.
        """
        allowed_methods = {'sklearn', 'mlp', 'xgboost'}

        def _resolve_method(method_spec, var_name):
            if isinstance(method_spec, str):
                return method_spec
            if isinstance(method_spec, dict):
                if var_name in method_spec:
                    return method_spec[var_name]
                if '__default__' in method_spec:
                    return method_spec['__default__']
                if 'default' in method_spec:
                    return method_spec['default']
                if '*' in method_spec:
                    return method_spec['*']
                return 'sklearn'
            raise TypeError("method must be a string or a dict mapping variable names to methods")

        methods_needed = {_resolve_method(method, v) for v in self.variables}
        unknown = sorted(m for m in methods_needed if m not in allowed_methods)
        if unknown:
            raise ValueError(f"Unknown method(s) {unknown}. Choose from 'sklearn', 'mlp', or 'xgboost'.")

        if 'mlp' in methods_needed and not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for method='mlp'. "
                              "Please install it with: pip install torch")
        if 'xgboost' in methods_needed and not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is required for method='xgboost'. "
                              "Please install it with: pip install xgboost")
        
        # Always load from cache - sampling was already done in plot_rank_histogram
        cached_counts, cached_ranks = self._load_cache()
        if cached_counts is None:
            raise ValueError("No cached data found. Please run plot_rank_histogram() first.")
        
        if use_binned:
            if verbose:
                print("Using cached binned data for calibration.")
            self._fit_isotonic_from_binned(cached_counts, save_name, method=method, verbose=verbose, batch_size=batch_size, epochs=epochs)
        else:
            if cached_ranks is None:
                raise ValueError("Cached ranks not available. Please run plot_rank_histogram() with save_cache=True.")
            if verbose:
                print("Using cached ranks for calibration.")
            self._fit_isotonic_from_ranks(cached_ranks, save_name, method=method, verbose=verbose, batch_size=batch_size, epochs=epochs)

    def _fit_monotone_xgboost(self, U, V, sample_weight=None):
        X = np.asarray(U, dtype=np.float32).reshape(-1, 1)
        y = np.asarray(V, dtype=np.float32).reshape(-1)

        n_jobs = self.num_workers if self.num_workers is not None else 1

        model = xgb.XGBRegressor(
            objective="reg:logistic",
            n_estimators=400,
            learning_rate=0.02,
            max_depth=4,
            min_child_weight=0.0,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=0.0,
            reg_alpha=0.0,
            gamma=0.0,
            tree_method="hist",
            max_bin=4096*4,
            monotone_constraints="(1)",
            n_jobs=n_jobs,
            random_state=42,
            verbosity=1,
        )

        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
            model.fit(X, y, sample_weight=sw)
        else:
            model.fit(X, y)

        return model

    def _fit_isotonic_from_ranks(self, all_ranks, save_name, method='sklearn', verbose=True, batch_size=4096, epochs=1000):
        """
        Fit isotonic regression models from collected ranks.
        Separated to allow reuse from cache.
        Note: Sampling is already applied in plot_rank_histogram, so no sampling here.

        Args:
            all_ranks: dict of collected ranks for each variable
            save_name: output filename for the calibration model
            method: 'sklearn' or 'mlp'
            verbose: Whether to print detailed progress information
            batch_size: Batch size for MLP training
            epochs: Number of training epochs for MLP
        """
        self.calibration_models = {}

        if verbose:
            if isinstance(method, str):
                if method == 'mlp':
                    method_name = "Monotonic MLP"
                elif method == 'xgboost':
                    method_name = "Monotone XGBoost"
                else:
                    method_name = "Isotonic Regression"
                print(f"Fitting {method_name}...")
            else:
                print("Fitting per-variable calibration methods...")
        
        for var in self.variables:
            local_method = method if isinstance(method, str) else (
                method[var] if isinstance(method, dict) and var in method else
                method.get('__default__') if isinstance(method, dict) and '__default__' in method else
                method.get('default') if isinstance(method, dict) and 'default' in method else
                method.get('*') if isinstance(method, dict) and '*' in method else
                'sklearn'
            )
            if not all_ranks[var]:
                if verbose:
                    print(f"No valid ranks found for {var}. Skipping.")
                continue

            # Concatenate
            ranks_concat = np.concatenate(all_ranks[var])
            M = len(ranks_concat)

            if M == 0:
                continue

            # Remove NaN and infinite values
            valid_mask = np.isfinite(ranks_concat)
            if not np.all(valid_mask):
                num_invalid = M - np.sum(valid_mask)
                if verbose:
                    print(f"  Warning: Removing {num_invalid} invalid values from {var} ranks.")
                ranks_concat = ranks_concat[valid_mask]
                M = len(ranks_concat)

            if M == 0:
                if verbose:
                    print(f"No valid finite ranks found for {var} after cleaning. Skipping.")
                continue
            
            # Compute U (PIT values in [0, 1])
            # Since we now return PIT values directly, we don't need to divide by N
            U = ranks_concat
            
            # Sort U (ensure no NaN values)
            U_sorted = np.sort(U)
            # Double-check for any remaining NaN values after sorting
            if not np.all(np.isfinite(U_sorted)):
                print(f"  Warning: U_sorted still contains invalid values for {var}, filtering again.")
                U_sorted = U_sorted[np.isfinite(U_sorted)]
                M = len(U_sorted)
                if M == 0:
                    print(f"No valid finite U values for {var} after sorting. Skipping.")
                    continue
            
            # Target V
            # V_j = j / (M + 1)
            V = np.arange(1, M + 1) / (M + 1)
            
            # Fit calibration model based on method
            if local_method == 'mlp':
                # Train Monotonic MLP
                if verbose:
                    print(f"  Training MLP for {var} with {M} samples (batch_size={batch_size})...")
                model = train_monotonic_nn(U_sorted, V, epochs=epochs, lr=0.001, verbose=verbose, batch_size=batch_size)
                self.calibration_models[var] = {'type': 'mlp', 'model': model}
                # Visualize the fitted calibration for MLP
                from modules.plot.plot_conformal import plot_calibration_fit
                output_dir = os.path.join(self.processor.output_dir, 'conformal')
                plot_calibration_fit(var, U_sorted, V, model, method='mlp', output_dir=output_dir, 
                                     predict_fn=lambda m, x: predict_monotonic_nn(m, x, batch_size=1024))
            elif local_method == 'xgboost':
                if verbose:
                    print(f"  Training monotone XGBoost for {var} with {M} samples (n_jobs={self.num_workers})...")
                model = self._fit_monotone_xgboost(U_sorted, V)
                self.calibration_models[var] = {'type': 'xgboost', 'model': model}
                from modules.plot.plot_conformal import plot_calibration_fit
                output_dir = os.path.join(self.processor.output_dir, 'conformal')
                plot_calibration_fit(
                    var,
                    U_sorted,
                    V,
                    model,
                    method='xgboost',
                    output_dir=output_dir,
                    predict_fn=lambda m, x: m.predict(np.asarray(x, dtype=np.float32).reshape(-1, 1)),
                )
            else:
                # Fit sklearn Isotonic Regression
                # We want to find monotonic f such that f(U) ~ V.
                if verbose:
                    print(f"  Fitting sklearn isotonic regression for {var} with {M} samples...")
                ir = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1, increasing=True)
                ir.fit(U_sorted, V)
                self.calibration_models[var] = {'type': 'sklearn', 'model': ir}
                # Visualize the fitted calibration for sklearn
                from modules.plot.plot_conformal import plot_calibration_fit
                output_dir = os.path.join(self.processor.output_dir, 'conformal')
                plot_calibration_fit(var, U_sorted, V, ir, method='sklearn', output_dir=output_dir)

            if verbose:
                print(f"  Fitted calibration for {var} with {M} samples.")

        # Save to conformal directory
        output_dir = os.path.join(self.processor.output_dir, 'conformal')
        os.makedirs(output_dir, exist_ok=True)
        
        # If save_name contains a path separator, treat it as a full path or relative path
        if os.sep in save_name:
            save_path = save_name
        else:
            save_path = os.path.join(output_dir, save_name)
            
        joblib.dump(self.calibration_models, save_path)
        print(f"Calibration models saved to {save_path}")

    def _fit_isotonic_from_binned(self, rank_counts, save_name, method='sklearn', verbose=True, batch_size=4096, epochs=1000):
        """
        Fit isotonic regression models from binned histogram data.
        Memory-efficient alternative to _fit_isotonic_from_ranks for large datasets.

        Uses bin centers as U values and weighted least squares to approximate
        the isotonic regression fit that would be obtained from raw PIT values.

        Args:
            rank_counts: dict of {var: np.array of bin counts}, shape (n_bins,)
            save_name: output filename for the calibration model
            method: 'sklearn' or 'mlp'
            verbose: Whether to print detailed progress information
            batch_size: Batch size for MLP training
            epochs: Number of training epochs for MLP
        """
        self.calibration_models = {}

        # Compute bin centers (U values)
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # shape: (n_bins,)

        if verbose:
            if isinstance(method, str):
                if method == 'mlp':
                    method_name = "Monotonic MLP"
                elif method == 'xgboost':
                    method_name = "Monotone XGBoost"
                else:
                    method_name = "Isotonic Regression"
                print(f"Fitting {method_name} from binned data (n_bins={self.n_bins})...")
            else:
                print(f"Fitting per-variable calibration methods from binned data (n_bins={self.n_bins})...")

        for var in self.variables:
            local_method = method if isinstance(method, str) else (
                method[var] if isinstance(method, dict) and var in method else
                method.get('__default__') if isinstance(method, dict) and '__default__' in method else
                method.get('default') if isinstance(method, dict) and 'default' in method else
                method.get('*') if isinstance(method, dict) and '*' in method else
                'sklearn'
            )
            counts = rank_counts[var]
            total = counts.sum()

            if total == 0:
                if verbose:
                    print(f"No valid data for {var}. Skipping.")
                continue

            # U: bin centers (representative PIT values for each bin)
            U = bin_centers

            # V: target uniform distribution values
            # For each bin, V is the expected position in the uniform CDF
            # V_j = (cumulative_count_before + count/2) / total
            cumsum_counts = np.cumsum(counts)
            V = (cumsum_counts - counts / 2) / total

            # Check for NaN or infinite values in V
            valid_mask = np.isfinite(V)
            if not np.all(valid_mask):
                num_invalid = len(V) - np.sum(valid_mask)
                if verbose:
                    print(f"  Warning: Removing {num_invalid} bins with invalid V values for {var}.")
                # Filter out invalid bins
                U = U[valid_mask]
                V = V[valid_mask]
                counts = counts[valid_mask]

            if len(V) == 0:
                if verbose:
                    print(f"No valid bins for {var} after cleaning. Skipping.")
                continue

            # Fit calibration model based on method
            if local_method == 'mlp':
                # Train Monotonic MLP with sample weights
                if verbose:
                    print(f"  Training MLP for {var} with {total} samples (binned to {len(U)} bins, batch_size={batch_size})...")
                model = train_monotonic_nn(U, V, sample_weight=counts,
                                           epochs=epochs, lr=0.001, verbose=verbose, batch_size=batch_size)
                self.calibration_models[var] = {'type': 'mlp', 'model': model}
            elif local_method == 'xgboost':
                if verbose:
                    print(f"  Training monotone XGBoost for {var} with {total} samples (binned to {len(U)} bins, n_jobs={self.num_workers})...")
                model = self._fit_monotone_xgboost(U, V, sample_weight=counts)
                self.calibration_models[var] = {'type': 'xgboost', 'model': model}
            else:
                # Fit sklearn Isotonic Regression with sample weights
                if verbose:
                    print(f"  Fitting sklearn isotonic regression for {var} with {total} samples...")
                ir = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1, increasing=True)
                ir.fit(U, V, sample_weight=counts)
                self.calibration_models[var] = {'type': 'sklearn', 'model': ir}

            if verbose:
                print(f"  Fitted calibration for {var} with {total} samples (binned to {self.n_bins} bins).")

        # Save to conformal directory
        output_dir = os.path.join(self.processor.output_dir, 'conformal')
        os.makedirs(output_dir, exist_ok=True)
        
        # If save_name contains a path separator, treat it as a full path or relative path
        if os.sep in save_name:
            save_path = save_name
        else:
            save_path = os.path.join(output_dir, save_name)
            
        joblib.dump(self.calibration_models, save_path)
        print(f"Calibration models saved to {save_path}")

    def load_calibration_models(self, load_path):
        """
        Load fitted calibration models from disk.
        
        Supports both sklearn IsotonicRegression and Monotonic MLP models.
        """
        if os.path.exists(load_path):
            self.calibration_models = joblib.load(load_path)
            print(f"Loaded calibration models from {load_path}")
            
            # Check if models are in new format (with 'type' and 'model' keys)
            # or old format (direct model objects)
            for var, model_data in self.calibration_models.items():
                if isinstance(model_data, dict) and 'type' in model_data:
                    model_type = model_data['type']
                    if model_type == 'mlp':
                        print(f"  Variable '{var}': Monotonic MLP model")
                    elif model_type == 'xgboost':
                        print(f"  Variable '{var}': Monotone XGBoost model")
                    else:
                        print(f"  Variable '{var}': sklearn IsotonicRegression model")
                else:
                    # Old format: assume sklearn IsotonicRegression
                    print(f"  Variable '{var}': sklearn IsotonicRegression model (legacy format)")
        else:
            print(f"Calibration model file not found: {load_path}")

    def apply_calibration(self, ranks, N, var, batch_size=8192):
        """
        Apply the fitted calibration to the ranks (PIT values).
        
        Args:
            ranks: raw PIT values in [0, 1]
            N: ensemble size (kept for compatibility)
            var: variable name
            batch_size: Batch size for MLP prediction. Default is 8192.
            
        Returns:
            Calibrated PIT values
        """
        if var not in self.calibration_models:
            print(f"No calibration model for {var}, returning raw ranks.")
            return ranks
        
        model_data = self.calibration_models[var]
        
        # Handle both new format (dict with 'type' and 'model') and old format (direct model)
        if isinstance(model_data, dict) and 'type' in model_data:
            model_type = model_data['type']
            model = model_data['model']
        else:
            # Legacy format: assume sklearn IsotonicRegression
            model_type = 'sklearn'
            model = model_data
        
        # Since ranks are already PIT values in [0, 1], we use them directly
        U = ranks
        
        # Predict new probabilities based on model type
        if model_type == 'mlp':
            U_cal = predict_monotonic_nn(model, U, batch_size=batch_size)
        elif model_type == 'xgboost':
            U_cal = model.predict(np.asarray(U, dtype=np.float32).reshape(-1, 1))
            U_cal = np.clip(U_cal, 0.0, 1.0)
        else:
            U_cal = model.predict(U)
        
        return U_cal

    def _compute_calibrated_histogram_binned(self, rank_counts, var, verbose=False, batch_size=8192):
        """
        Compute calibrated histogram from binned data using memory-efficient method.
        
        Instead of applying calibration to individual PIT values, this method:
        1. Computes the calibration transformation for each bin center
        2. Redistributes counts from original bins to calibrated bins
        
        This avoids storing all raw PIT values in memory.
        
        Args:
            rank_counts: np.array of bin counts, shape (n_bins,)
            var: variable name
            verbose: whether to print progress
            batch_size: Batch size for MLP prediction. Default is 8192.
            
        Returns:
            np.array: calibrated bin counts, shape (n_bins,)
        """
        if verbose:
            print(f"      Computing calibrated histogram for '{var}' using binned method...")
        
        # Compute bin centers (representative PIT values for each bin)
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Apply calibration to bin centers
        # U_cal_centers[i] = f(bin_centers[i])
        U_cal_centers = self.apply_calibration(bin_centers, self.n_bins, var, batch_size=batch_size)
        
        if verbose:
            print(f"        Bin center calibration range: [{U_cal_centers.min():.4f}, {U_cal_centers.max():.4f}]")
        
        # Initialize calibrated histogram
        calibrated_counts = np.zeros(self.n_bins, dtype=int)
        
        # Redistribute counts from original bins to calibrated bins
        # For each original bin i with count c:
        #   - Find which calibrated bin j the center maps to
        #   - Add c to calibrated bin j
        for i in range(self.n_bins):
            count = rank_counts[i]
            if count == 0:
                continue
            
            # Find the calibrated bin index
            # U_cal is in [0, 1], map to [0, n_bins-1]
            cal_bin = int(np.floor(U_cal_centers[i] * self.n_bins))
            cal_bin = np.clip(cal_bin, 0, self.n_bins - 1)
            
            calibrated_counts[cal_bin] += count
        
        if verbose:
            print(f"        Calibrated histogram total: {calibrated_counts.sum()}")
            print(f"        Bin count range: [{calibrated_counts.min()}, {calibrated_counts.max()}]")
        
        return calibrated_counts

    def plot_calibrated_rank_histogram(self, save_name="calibrated_rank_histogram.png", model_path=None, verbose=True, use_binned=False, batch_size=8192, num_workers=None):
        """
        Calculate and plot the Calibrated Rank Histogram (PIT Histogram) for each variable.
        Uses the fitted isotonic regression model to transform PIT values.
        This method always uses cached data from plot_rank_histogram().
        
        Args:
            save_name (str): Output filename.
            model_path (str, optional): Path to load calibration models from. 
                                        If None, uses self.calibration_models.
            verbose (bool): Whether to print detailed progress information. Default is True.
            use_binned (bool): If True, use binned histogram data for memory-efficient 
                              computation. Recommended for large datasets (billions of points).
                              If False, use raw PIT values (more accurate but memory-intensive).
            batch_size (int): Batch size for MLP prediction. Default is 8192.
            num_workers (int): Number of parallel workers for calibration. 
                              If None, process sequentially. For sklearn models, 
                              multi-threading is effective. For MLP models, GPU batching 
                              is already parallelized, so num_workers > 1 may not help much.
        """
        import concurrent.futures
        import time

        start_time = time.time()
        
        if verbose:
            print("=" * 60)
            print("Starting Calibrated Rank Histogram Calculation")
            print("=" * 60)

        # Try to load calibration models if path is provided
        if model_path:
            if verbose:
                print(f"[1/5] Loading calibration models from: {model_path}")
            self.load_calibration_models(model_path)
            if verbose:
                print(f"      Successfully loaded calibration models")
        elif verbose:
            print("[1/5] Using existing calibration models (model_path not provided)")

        if not hasattr(self, 'calibration_models'):
            print("ERROR: Calibration models not found. Please run fit_isotonic_calibration() or provide model_path.")
            return

        # Always load from cache - sampling was already done in plot_rank_histogram
        if verbose:
            print("[2/5] Loading cached rank histogram data...")
        cached_counts, cached_ranks = self._load_cache()
        if cached_counts is None:
            raise ValueError("No cached data found. Please run plot_rank_histogram() first.")
        
        rank_counts = cached_counts
        all_ranks = cached_ranks
        
        if verbose:
            print(f"      Successfully loaded cached rank_counts")
            for var in self.variables:
                print(f"        Variable '{var}': {rank_counts[var].sum()} samples")
            if all_ranks is not None:
                print(f"      Cached all_ranks available")
            else:
                print(f"      Cached all_ranks NOT available")
        
        # Get N from ensemble groups for apply_calibration
        if verbose:
            print("[3/5] Getting ensemble size...")
        ensemble_groups = self.processor._get_ensemble_groups()
        if ensemble_groups:
            first_group_files = next(iter(ensemble_groups.values()))
            N = len(first_group_files)
        else:
            N = self.n_bins  # Fallback, though this shouldn't happen
        
        if verbose:
            print(f"[4/5] Computing calibrated histograms for each variable...")
            for var in self.variables:
                total_count = rank_counts[var].sum()
                if total_count > 0:
                    print(f"      Variable '{var}': {total_count} total samples")
                else:
                    print(f"      Variable '{var}': No data (will be skipped)")
            if num_workers is not None and num_workers > 1:
                print(f"      Using {num_workers} parallel workers for calibration")
                            
        # Compute calibrated histograms
        calibrated_counts = {var: np.zeros(self.n_bins, dtype=int) for var in self.variables}
        calibrated_all_ranks = {var: [] for var in self.variables} if not use_binned else None
        
        def _process_single_variable(var):
            """Process calibration for a single variable."""
            total_count = rank_counts[var].sum()
            if total_count == 0:
                return var, None, None, "skipped"
            
            if use_binned:
                # Memory-efficient: use binned data to compute calibrated histogram
                counts = self._compute_calibrated_histogram_binned(
                    rank_counts[var], var, verbose=False, batch_size=batch_size
                )
                return var, counts, None, "binned"
            elif all_ranks is not None:
                # Original method: use raw PIT values
                ranks_concat = np.concatenate(all_ranks[var])
                
                # Apply calibration
                U_cal = self.apply_calibration(ranks_concat, N, var, batch_size=batch_size)
                
                # Convert to discrete bins for histogram
                R_cal = np.floor(U_cal * self.n_bins).astype(int)
                R_cal = np.clip(R_cal, 0, self.n_bins - 1)
                
                counts = np.bincount(R_cal, minlength=self.n_bins)
                if len(counts) > self.n_bins:
                    counts = counts[:self.n_bins]
                return var, counts, U_cal, "raw"
            else:
                return var, None, None, "error"
        
        # Process variables (parallel or sequential)
        if num_workers is not None and num_workers > 1:
            # Parallel processing using ThreadPoolExecutor
            from concurrent.futures import ThreadPoolExecutor
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_process_single_variable, var): var for var in self.variables}
                
                for future in tqdm(concurrent.futures.as_completed(futures), 
                                   total=len(futures), desc="[4/5] Calibrating variables"):
                    var, counts, U_cal, status = future.result()
                    if status == "skipped":
                        if verbose:
                            print(f"      Skipping variable '{var}' (no data)")
                    elif status == "error":
                        raise ValueError(
                            f"use_binned=False but all_ranks not available in cache for '{var}'. "
                            f"Please run plot_rank_histogram() with save_cache=True."
                        )
                    else:
                        calibrated_counts[var] = counts
                        if U_cal is not None and calibrated_all_ranks is not None:
                            calibrated_all_ranks[var] = U_cal
                        if verbose:
                            print(f"      Variable '{var}': calibrated ({status} method, {counts.sum()} samples)")
        else:
            # Sequential processing (original behavior)
            for var in self.variables:
                total_count = rank_counts[var].sum()
                if total_count == 0:
                    if verbose:
                        print(f"      Skipping variable '{var}' (no data)")
                    continue
                
                if use_binned:
                    # Memory-efficient: use binned data to compute calibrated histogram
                    if verbose:
                        print(f"      Applying calibration to variable '{var}' (binned)...")
                    calibrated_counts[var] = self._compute_calibrated_histogram_binned(
                        rank_counts[var], var, verbose, batch_size=batch_size
                    )
                elif all_ranks is not None:
                    # Original method: use raw PIT values
                    ranks_concat = np.concatenate(all_ranks[var])
                    
                    if verbose:
                        print(f"      Applying calibration to variable '{var}' ({len(ranks_concat)} ranks)...")
                    
                    # Apply calibration
                    U_cal = self.apply_calibration(ranks_concat, N, var, batch_size=batch_size)
                    
                    if verbose:
                        print(f"        Raw PIT range: [{ranks_concat.min():.4f}, {ranks_concat.max():.4f}]")
                        print(f"        Calibrated PIT range: [{U_cal.min():.4f}, {U_cal.max():.4f}]")
                    
                    # Store calibrated ranks for KDE plotting
                    if calibrated_all_ranks is not None:
                        calibrated_all_ranks[var] = U_cal
                    
                    # Convert to discrete bins for histogram
                    R_cal = np.floor(U_cal * self.n_bins).astype(int)
                    R_cal = np.clip(R_cal, 0, self.n_bins - 1)
                    
                    counts = np.bincount(R_cal, minlength=self.n_bins)
                    if len(counts) > self.n_bins:
                        counts = counts[:self.n_bins]
                    calibrated_counts[var] = counts
                    
                    if verbose:
                        print(f"        Histogram total count: {counts.sum()}")
                        print(f"        Bin count range: [{counts.min()}, {counts.max()}]")
                else:
                    # use_binned=False but all_ranks is None (e.g., cache without all_ranks)
                    # Raise error as requested
                    raise ValueError(
                        f"use_binned=False but all_ranks not available in cache. "
                        f"Please run plot_rank_histogram() with save_cache=True."
                    )
        
        if verbose:
            print(f"[5/5] Generating and saving histogram plot...")
            
        from modules.plot.plot_conformal import plot_histograms
        save_path = os.path.join(os.path.dirname(self.sample_dir), 'conformal', save_name)
        
        if verbose:
            print(f"      Save path: {save_path}")
        
        # Pass calibrated_all_ranks to plot_histograms for KDE plotting
        plot_histograms(calibrated_counts, self.n_bins, self.variables, save_path, None)
        
        # Save calibrated data to cache for use by plot_combined_rank_histograms
        if verbose:
            print(f"      Saving calibrated data to cache...")
        self._save_calibrated_cache(calibrated_counts, calibrated_all_ranks)
        if verbose:
            print(f"      Calibrated cache saved successfully")
        
        elapsed_time = time.time() - start_time
        if verbose:
            print("=" * 60)
            print(f"Calibrated Rank Histogram completed in {elapsed_time:.2f} seconds")
            print(f"Output saved to: {save_path}")
            print("=" * 60)

    def visualize_pixel_distributions(self, num_pixels=3, save_name="pixel_distributions.png", 
                                       model_path=None, date_str=None, depth_level=None, 
                                       pixel_coords=None, seed=None):
        """
        Visualize empirical distributions of selected pixels before and after calibration.
        
        Args:
            num_pixels (int): Number of random pixels to select if pixel_coords not provided.
            save_name (str): Filename for the output plot.
            model_path (str, optional): Path to calibration models.
            date_str (str, optional): Date string in format 'YYYYMMDD'. If None, randomly selected.
            depth_level (int, optional): Depth level (number after 'd' in filename). If None, randomly selected.
            pixel_coords (list, optional): List of (h, w) tuples specifying pixel coordinates to plot.
                                          If provided, num_pixels is ignored.
            seed (int, optional): Random seed for reproducibility.
        """
        import random
        import re
        from scipy.special import ndtr
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Load calibration models
        if model_path:
            self.load_calibration_models(model_path)
        
        if not hasattr(self, 'calibration_models'):
             # Try default path
            default_path = os.path.join(os.path.dirname(self.sample_dir), 'conformal', 'isotonic_calibration.pkl')
            if os.path.exists(default_path):
                self.load_calibration_models(default_path)
            else:
                # Also try output_dir from processor if available, though sample_dir parent is usually safer given existing code
                pass

        # Get all groups
        ensemble_groups = self.processor._get_ensemble_groups()
        if not ensemble_groups:
            print("No ensemble groups found.")
            return
        
        # Parse group keys to extract date and depth info
        # Group keys are like: "20230101_d00"
        group_info = []
        for base_id in ensemble_groups.keys():
            match = re.match(r'(\d{8})_d(\d+)', base_id)
            if match:
                date = match.group(1)
                depth = int(match.group(2))
                group_info.append((base_id, date, depth))
        
        if not group_info:
            print("No valid groups found with expected naming pattern 'YYYYMMDD_dXX'.")
            return
        
        # Filter by date if specified
        if date_str is not None:
            matching_groups = [(bid, d, dep) for bid, d, dep in group_info if d == date_str]
            if not matching_groups:
                print(f"No groups found for date {date_str}. Available dates: {sorted(set(d for _, d, _ in group_info))}")
                return
            group_info = matching_groups
        
        # Filter by depth_level if specified
        if depth_level is not None:
            matching_groups = [(bid, d, dep) for bid, d, dep in group_info if dep == depth_level]
            if not matching_groups:
                available_depths = sorted(set(dep for _, _, dep in group_info))
                print(f"No groups found for depth level {depth_level}. Available depths: {available_depths}")
                return
            group_info = matching_groups
        
        # Select a group
        base_id, selected_date, selected_depth = random.choice(group_info)
        file_paths = ensemble_groups[base_id]
        print(f"Visualizing pixels from group: {base_id} (date={selected_date}, depth={selected_depth})")
        
        # Determine N
        N = len(file_paths)
        
        # Load Data
        try:
            # 1. Load Ensemble
            members = []
            for fp in file_paths:
                if fp.endswith('.nc'):
                    ds = xr.open_dataset(fp)
                    if 'data' in ds:
                        data = ds['data'].values
                    else:
                        vars_list = ['thetao', 'so', 'uo', 'vo']
                        present_vars = [v for v in vars_list if v in ds]
                        if present_vars:
                            data_list = [ds[v].values for v in present_vars]
                            processed_list = []
                            for d in data_list:
                                if d.ndim == 3: processed_list.append(d[0])
                                else: processed_list.append(d)
                            data = np.stack(processed_list)
                        else:
                            ds.close()
                            continue
                    ds.close()
                    members.append(data)
                else:
                    data = np.load(fp)
                    if fp.endswith('.npz'):
                        if hasattr(data, 'files') and 'data' in data.files:
                            data = data['data']
                    members.append(data)
            
            if not members:
                print("Could not load members.")
                return

            ensemble = np.stack(members, axis=0) # (N, C, H, W)
            
            # 2. Load GT
            gt_filename = base_id + '.nc'
            gt_path = os.path.join(self.ground_truth_dir, gt_filename)
            if not os.path.exists(gt_path):
                print(f"GT not found: {gt_path}")
                return
                
            gt_data_list = []
            lon_coords = None
            lat_coords = None
            with xr.open_dataset(gt_path) as ds_gt:
                for var in self.variables:
                    if var in ds_gt:
                        data = ds_gt[var].values
                        if data.ndim == 3 and data.shape[0] == 1: data = data[0]
                        gt_data_list.append(data)
                
                # Extract longitude and latitude coordinates if available
                if 'longitude' in ds_gt:
                    lon_coords = ds_gt['longitude'].values
                elif 'lon' in ds_gt:
                    lon_coords = ds_gt['lon'].values
                    
                if 'latitude' in ds_gt:
                    lat_coords = ds_gt['latitude'].values
                elif 'lat' in ds_gt:
                    lat_coords = ds_gt['lat'].values
            
            gt = np.array(gt_data_list) # (C, H, W)

            # 3. Align
            if ensemble.shape[2:] != gt.shape[1:]:
                if ensemble.shape[2] > gt.shape[1]: 
                    pad_h = ensemble.shape[2] - gt.shape[1]
                    gt = np.pad(gt, ((0,0), (pad_h, 0), (0,0)), mode='constant', constant_values=np.nan)

            if ensemble.shape[2:] != gt.shape[1:]:
                print(f"Shape mismatch: Ensemble {ensemble.shape}, GT {gt.shape}")
                return

            # 4. Select Pixels
            if pixel_coords is not None:
                # Use user-specified pixel coordinates
                selected_pixels = np.array(pixel_coords)
                # Validate coordinates
                H, W = gt.shape[1], gt.shape[2]
                valid_coords = []
                for h, w in selected_pixels:
                    if 0 <= h < H and 0 <= w < W:
                        valid_coords.append([h, w])
                    else:
                        print(f"Warning: Pixel coordinate ({h}, {w}) is out of bounds (H={H}, W={W}), skipping.")
                if not valid_coords:
                    print("No valid pixel coordinates provided.")
                    return
                selected_pixels = np.array(valid_coords)
                print(f"Using {len(selected_pixels)} user-specified pixel coordinates: {selected_pixels.tolist()}")
            else:
                # Randomly select pixels
                mask = np.isnan(gt) # (C, H, W)
                ens_has_nan = np.isnan(ensemble).any(axis=0) # (C, H, W)
                valid_mask = (~mask) & (~ens_has_nan) # (C, H, W)
                
                common_valid_mask = valid_mask.all(axis=0) # (H, W)
                valid_indices = np.argwhere(common_valid_mask) # (P, 2)
                
                if len(valid_indices) < num_pixels:
                    print(f"Not enough valid pixels (found {len(valid_indices)}).")
                    return
                    
                selected_indices_idx = np.random.choice(len(valid_indices), num_pixels, replace=False)
                selected_pixels = valid_indices[selected_indices_idx] # (num_pixels, 2)
                print(f"Randomly selected {len(selected_pixels)} pixels: {selected_pixels.tolist()}")
            
            # Plotting
            from modules.plot.plot_conformal import plot_pixel_distributions
            
            save_path = os.path.join(os.path.dirname(self.sample_dir), 'conformal', save_name)
            
            calibration_models = None
            if hasattr(self, 'calibration_models'):
                calibration_models = self.calibration_models
            
            plot_pixel_distributions(
                ensemble=ensemble,
                gt=gt,
                variables=self.variables,
                selected_pixels=selected_pixels,
                calibration_models=calibration_models,
                save_path=save_path,
                lon_coords=lon_coords,
                lat_coords=lat_coords
            )

        except Exception as e:
            print(f"Error visualizing pixels: {e}")
            import traceback
            traceback.print_exc()

    def calculate_coverage(self, confidence_level=0.9, model_path=None, save_name="coverage_results.txt", verbose=True):
        """
        Calculate and print the coverage of confidence intervals before and after calibration.
        This method always uses cached data from plot_rank_histogram().
        
        Args:
            confidence_level (float): Desired confidence level (e.g., 0.9).
            model_path (str, optional): Path to load calibration models.
            save_name (str, optional): Name of the file to save results to.
            verbose (bool): Whether to print detailed progress information. Default is True.
        """
        import time
        start_time = time.time()
        
        if verbose:
            print("=" * 60)
            print(f"Starting Coverage Calculation (Confidence: {confidence_level:.4f})")
            print("=" * 60)
        
        if model_path:
            if verbose:
                print(f"[1/4] Loading calibration models from: {model_path}")
            self.load_calibration_models(model_path)
        elif verbose:
            print("[1/4] Using existing calibration models (model_path not provided)")
        
        has_calibration = hasattr(self, 'calibration_models')
        if not has_calibration:
             print("Warning: Calibration models not found. Only calculating uncalibrated coverage.")

        # Always load from cache - sampling was already done in plot_rank_histogram
        if verbose:
            print("[2/4] Loading cached ranks from plot_rank_histogram...")
        cached_counts, cached_ranks = self._load_cache()
        if cached_ranks is None:
            raise ValueError("No cached ranks found. Please run plot_rank_histogram() with save_cache=True first.")
        
        if verbose:
            print(f"      Successfully loaded cached ranks")
            for var in self.variables:
                if cached_ranks[var]:
                    n_samples = len(cached_ranks[var])
                    total_points = sum(len(r) for r in cached_ranks[var])
                    print(f"        Variable '{var}': {n_samples} sample groups, {total_points} total points")
        
        # Get N from ensemble groups for apply_calibration
        if verbose:
            print("[3/4] Getting ensemble size...")
        ensemble_groups = self.processor._get_ensemble_groups()
        if ensemble_groups:
            first_group_files = next(iter(ensemble_groups.values()))
            N = len(first_group_files)
        else:
            N = self.n_bins  # Fallback
        
        if verbose:
            print(f"      Ensemble size: {N}")
            print(f"[4/4] Computing coverage for confidence level {confidence_level:.4f}...")
        
        all_ranks = cached_ranks

        alpha = 1.0 - confidence_level
        lower_q = alpha / 2.0
        upper_q = 1.0 - alpha / 2.0
        
        output_lines = []
        output_lines.append(f"\nCoverage Results (Target: {confidence_level:.4f}, Interval: [{lower_q:.4f}, {upper_q:.4f}])")
        output_lines.append("-" * 60)
        output_lines.append(f"{'Variable':<10} | {'Uncalibrated':<15} | {'Calibrated':<15}")
        output_lines.append("-" * 60)

        for var in self.variables:
            if not all_ranks[var]:
                output_lines.append(f"{var:<10} | {'N/A':<15} | {'N/A':<15}")
                if verbose:
                    print(f"      Variable '{var}': No data (skipped)")
                continue
                
            ranks = np.concatenate(all_ranks[var])
            total_points = len(ranks)
            
            if verbose:
                print(f"      Variable '{var}': Processing {total_points} points...")
            
            # Uncalibrated Coverage
            # Check if rank is in [lower_q, upper_q]
            in_interval_uncal = np.sum((ranks >= lower_q) & (ranks <= upper_q))
            coverage_uncal = in_interval_uncal / total_points
            
            coverage_cal_str = "N/A"
            if has_calibration and var in self.calibration_models:
                # Apply calibration
                ranks_cal = self.apply_calibration(ranks, N, var)
                in_interval_cal = np.sum((ranks_cal >= lower_q) & (ranks_cal <= upper_q))
                coverage_cal = in_interval_cal / total_points
                coverage_cal_str = f"{coverage_cal:.4f}"
                if verbose:
                    print(f"        Uncalibrated: {coverage_uncal:.4f}, Calibrated: {coverage_cal:.4f}")
            else:
                if verbose:
                    print(f"        Uncalibrated: {coverage_uncal:.4f}, Calibrated: N/A")
            
            output_lines.append(f"{var:<10} | {coverage_uncal:.4f}          | {coverage_cal_str:<15}")
        output_lines.append("-" * 60)
        
        final_output = "\n".join(output_lines)
        print(final_output)
        
        # Save results
        save_dir = os.path.join(os.path.dirname(self.sample_dir), 'conformal')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name)
        
        try:
            with open(save_path, 'w') as f:
                f.write(final_output)
            if verbose:
                print(f"\nResults saved to {save_path}")
        except Exception as e:
            print(f"\nError saving results to {save_path}: {e}")
        
        elapsed_time = time.time() - start_time
        if verbose:
            print("=" * 60)
            print(f"Coverage calculation completed in {elapsed_time:.2f} seconds")
            print("=" * 60)
