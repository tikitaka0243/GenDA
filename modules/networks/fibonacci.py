import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class FibonacciSphere:
    def __init__(self, num_points=1024):
        """
        Initialize the FibonacciSphere generator.
        
        Args:
            num_points (int): The number of points to generate on the sphere.
        """
        self.num_points = num_points
        self.points = None

    def generate_points(self):
        """
        Generate points on a sphere using the Fibonacci lattice algorithm.
        
        Returns:
            np.ndarray: An array of shape (num_points, 3) containing the (x, y, z) coordinates.
        """
        indices = np.arange(0, self.num_points, dtype=float) + 0.5
        
        # phi is the angle from the Z axis (0 to pi)
        # We want z to be uniformly distributed from 1 to -1
        # z = cos(phi) => phi = arccos(z)
        # z = 1 - 2 * indices / num_points
        phi = np.arccos(1 - 2 * indices / self.num_points)
        
        # theta is the azimuthal angle
        # using the golden angle increment
        theta = np.pi * (1 + 5**0.5) * indices

        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        
        self.points = np.stack((x, y, z), axis=1)
        return self.points

    def get_lon_lat(self):
        """
        Convert Cartesian coordinates to Longitude and Latitude.
        
        Returns:
            tuple: (lon, lat) arrays in degrees.
        """
        if self.points is None:
            self.generate_points()
            
        x, y, z = self.points[:, 0], self.points[:, 1], self.points[:, 2]
        
        # Calculate latitude: arcsin(z)
        # z goes from 1 (North Pole) to -1 (South Pole)
        # lat goes from 90 to -90
        lat = np.degrees(np.arcsin(z))
        
        # Calculate longitude: arctan2(y, x)
        # lon goes from -180 to 180
        lon = np.degrees(np.arctan2(y, x))
        
        return lon, lat

    def plot(self, save_path=None, show=True):
        """
        Visualize the generated points on a 3D sphere.
        
        Args:
            save_path (str, optional): Path to save the plot image.
            show (bool): Whether to display the plot.
        """
        if self.points is None:
            self.generate_points()
            
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        x = self.points[:, 0]
        y = self.points[:, 1]
        z = self.points[:, 2]
        
        # Plot points
        sc = ax.scatter(x, y, z, s=20, c=z, cmap='viridis', alpha=0.8)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Fibonacci Sphere ({self.num_points} points)')
        
        # Ensure aspect ratio is equal
        ax.set_box_aspect([1, 1, 1])
        
        # Add colorbar
        plt.colorbar(sc, ax=ax, label='Z coordinate')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D Plot saved to {save_path}")
            
        if show:
            try:
                plt.show()
            except Exception as e:
                print(f"Warning: Could not display plot directly ({e}).")
        plt.close(fig)

    def get_grid_nearest_indices(self, step=0.5):
        """
        Generate a lat/lon grid and find the nearest Fibonacci point for each grid point.
        
        Args:
            step (float): Grid resolution in degrees.
            
        Returns:
            tuple: (glon, glat, nearest_indices) where glon/glat are meshgrids and 
                   nearest_indices are indices of nearest Fibonacci points.
        """
        grid_lon = np.arange(-180, 180, step)
        grid_lat = np.arange(-90, 90.5, step)
        glon, glat = np.meshgrid(grid_lon, grid_lat)
        
        # Convert grid points to Cartesian coordinates on unit sphere
        glat_rad = np.radians(glat.flatten())
        glon_rad = np.radians(glon.flatten())
        
        gx = np.cos(glat_rad) * np.cos(glon_rad)
        gy = np.cos(glat_rad) * np.sin(glon_rad)
        gz = np.sin(glat_rad)
        grid_points = np.stack((gx, gy, gz), axis=1)
        
        # Find nearest Fibonacci point for each grid point
        # Maximizing dot product is equivalent to minimizing Euclidean distance on sphere
        # which is equivalent to minimizing geodesic distance (Great Circle distance)
        dots = grid_points @ self.points.T
        nearest_indices = np.argmax(dots, axis=1)
        
        return glon, glat, nearest_indices

    def plot_grid_distribution(self, step=0.5, bins=30, save_path=None, show=True):
        """
        Plot the distribution histogram of the number of grid points assigned to each Fibonacci point.
        This visualizes how uniform the area of each Voronoi cell is.

        Args:
            step (float): Grid resolution in degrees.
            bins (int): Number of bins for the histogram.
            save_path (str, optional): Path to save the plot image.
            show (bool): Whether to display the plot.
        """
        if self.points is None:
            self.generate_points()
            
        # Get nearest indices for grid points
        _, _, nearest_indices = self.get_grid_nearest_indices(step=step)
        
        # Count occurrences of each index
        counts = np.bincount(nearest_indices, minlength=self.num_points)
        
        # Plot histogram of counts
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Plot histogram
        n, bins_edges, patches = ax.hist(counts, bins=bins, color='skyblue', edgecolor='black', alpha=0.7, log=True)
        
        # Add statistics
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        min_count = np.min(counts)
        max_count = np.max(counts)
        
        stats_text = (f"Mean: {mean_count:.2f}\n"
                      f"Std: {std_count:.2f}\n"
                      f"Min: {min_count}\n"
                      f"Max: {max_count}\n"
                      f"CV: {std_count/mean_count:.4f}")
        
        # Add text box with statistics
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        ax.set_xlabel('Number of Grid Points per Class')
        ax.set_ylabel('Frequency (Number of Classes)')
        ax.set_title(f'Distribution of Grid Point Counts per Class\n(Grid Step: {step}°, Num Classes: {self.num_points})')
        ax.grid(axis='y', alpha=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution Plot saved to {save_path}")
            
        if show:
            try:
                plt.show()
            except Exception as e:
                print(f"Warning: Could not display plot directly ({e}).")
        plt.close(fig)

    def plot_on_map(self, save_path=None, show=True):
        """
        Visualize the generated points on a 2D Earth map using Cartopy.
        
        Args:
            save_path (str, optional): Path to save the plot image.
            show (bool): Whether to display the plot.
        """
        if self.points is None:
            self.generate_points()
            
        lon, lat = self.get_lon_lat()
        
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.2)
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        
        # Generate and plot 0.5 degree grid points under the Fibonacci points
        glon, glat, nearest_indices = self.get_grid_nearest_indices(step=0.5)
        
        # Shuffle indices to ensure adjacent regions have distinct colors
        perm = np.random.permutation(self.num_points)
        shuffled_indices = perm[nearest_indices]
        
        # Use a distinct colormap for classification
        # Plot classified grid points
        ax.scatter(glon.flatten(), glat.flatten(), s=0.5, c=shuffled_indices, 
                   cmap='prism', alpha=1, 
                   transform=ccrs.PlateCarree(), zorder=1, label='Classified 0.5° Grid', linewidth=0)
        
        # Plot points
        sc = ax.scatter(lon, lat, s=10, transform=ccrs.PlateCarree(), alpha=0.8, 
                        edgecolors='k', linewidth=0, color='black', zorder=2, label='Fibonacci Points')
        
        # Add legend
        ax.legend(loc='lower right')
        
        ax.set_title(f'Fibonacci Sphere Points on Earth Map ({self.num_points} points)')
        
        if save_path:
            plt.savefig(save_path, dpi=1800, bbox_inches='tight')
            print(f"Map Plot saved to {save_path}")
            
        if show:
            try:
                plt.show()
            except Exception as e:
                print(f"Warning: Could not display plot directly ({e}).")
        plt.close(fig)


def normalize_deltas(deltas, nearest_indices, num_patches):
    """
    Normalize deltas by dividing by the maximum radius in each patch.
    deltas: (H*W, 3)
    nearest_indices: (H*W,)
    num_patches: int
    """
    # Calculate max radius per patch for normalization
    dist_sq = np.sum(deltas**2, axis=1) # (H*W,)
    max_dist_sq_per_patch = np.zeros(num_patches)
    np.maximum.at(max_dist_sq_per_patch, nearest_indices, dist_sq)
    max_radii = np.sqrt(max_dist_sq_per_patch)
    max_radii = np.maximum(max_radii, 1e-6) # Avoid div by zero
    
    # Normalize deltas to [-1, 1]
    radii_per_point = max_radii[nearest_indices] # (H*W,)
    deltas_normalized = deltas / radii_per_point[:, np.newaxis]
    return deltas_normalized


def generate_grid_points(img_size):
    """
    Generate 3D Cartesian coordinates for a lat/lon grid.
    img_size: (H, W)
    Returns: grid_points (H, W, 3), lat_grid, lon_grid
    """
    H, W = img_size
    # GLORYS12 specific range: -85.5 to 90 lat, -180 to 180 lon (0.5 deg res)
    if H == 352 and W == 720:
        lat = np.linspace(-85.5, 90, H)
        lon = np.linspace(-180, 180, W, endpoint=False)
    else:
        # Fallback for other sizes
        lat = np.linspace(-90, 90, H)
        lon = np.linspace(-180, 180, W)
        
    glat, glon = np.meshgrid(lat, lon, indexing='ij') # (H, W)
    
    # Convert to radians
    rlat = np.radians(glat)
    rlon = np.radians(glon)
    
    # Cartesian coordinates of grid points
    gx = np.cos(rlat) * np.cos(rlon)
    gy = np.cos(rlat) * np.sin(rlon)
    gz = np.sin(rlat)
    grid_points = np.stack((gx, gy, gz), axis=-1) # (H, W, 3)
    
    return grid_points, rlat


class Fib3DPosEmbedder(nn.Module):
    """
    Embeds 3D Cartesian coordinates (x, y, z in [-1, 1]) using sinusoidal embeddings.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, max_freq_power=16):
        super().__init__()
        # We process x, y, z separately and concatenate
        # Each coordinate gets `frequency_embedding_size` features
        self.total_input_dim = frequency_embedding_size * 3
        
        self.mlp = nn.Sequential(
            nn.Linear(self.total_input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_freq_power = max_freq_power

    def forward(self, coords):
        # coords: (N, 3) in [-1, 1]
        
        dim = self.frequency_embedding_size
        half = dim // 2
        
        # Frequencies: 2^0 to 2^max_freq_power
        # Use logspace to cover the range evenly
        freqs = torch.logspace(0, self.max_freq_power, steps=half, base=2.0, device=coords.device)
        # freqs: (half,)
        
        # We want to apply this to each of x, y, z
        # coords: (N, 3) -> (N, 3, 1)
        # freqs: (half,) -> (1, 1, half)
        
        # Map [-1, 1] to [-pi, pi] * freq
        args = coords.unsqueeze(-1) * freqs.view(1, 1, -1) * math.pi # (N, 3, half)
        
        emb_sin = torch.sin(args) # (N, 3, half)
        emb_cos = torch.cos(args) # (N, 3, half)
        
        embedding = torch.cat([emb_sin, emb_cos], dim=-1) # (N, 3, dim) (if dim is even)
        
        if dim % 2:
             # Padding if odd
             padding = torch.zeros_like(embedding[:, :, :1])
             embedding = torch.cat([embedding, padding], dim=-1)
             
        # Flatten: (N, 3*dim)
        embedding = embedding.reshape(coords.shape[0], -1)
        
        return self.mlp(embedding)


class FibonacciPatchEmbed(nn.Module):
    def __init__(self, img_size=(352, 720), in_chans=3, embed_dim=768, fibonacci_n=1024, hidden_dim=None):
        super().__init__()
        self.img_size = img_size
        self.num_patches = fibonacci_n
        self.embed_dim = embed_dim
        
        if hidden_dim is None:
            hidden_dim = embed_dim

        # Initialize Fibonacci Sphere
        self.sphere = FibonacciSphere(num_points=fibonacci_n)
        self.sphere.generate_points() # (N, 3)
        self.register_buffer('sphere_points', torch.from_numpy(self.sphere.points).float())
        
        # Generate Grid and Mapping
        grid_points, rlat = generate_grid_points(img_size)
        
        # Find nearest Fibonacci point
        flat_grid = grid_points.reshape(-1, 3)
        dots = flat_grid @ self.sphere.points.T # (H*W, N)
        nearest_indices = np.argmax(dots, axis=1) # (H*W,)
        
        self.register_buffer('nearest_indices', torch.from_numpy(nearest_indices).long())
        
        # Calculate Delta and Weights
        centers = self.sphere.points[nearest_indices] # (H*W, 3)
        deltas = flat_grid - centers # (H*W, 3)
        
        # Normalize deltas
        deltas = normalize_deltas(deltas, nearest_indices, fibonacci_n)
        
        weights = np.cos(rlat).reshape(-1) # (H*W,)
        
        self.register_buffer('deltas', torch.from_numpy(deltas).float())
        self.register_buffer('weights', torch.from_numpy(weights).float())
        self.register_buffer('cos_lat', torch.from_numpy(weights).float())
        
        # MLP for feature embedding
        mlp_in_dim = in_chans + 4 # C + 3 (delta) + 1 (cos_lat)
        
        # Optimization: Decompose MLP to run heavy projection on patches instead of pixels
        # 1. Pixel-wise MLP: (C+4) -> hidden_dim
        self.pixel_mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, hidden_dim),
            nn.SiLU()
        )
        
        # 2. Patch-wise Projection: hidden_dim -> embed_dim
        # This runs after pooling, so it's much cheaper (N vs H*W)
        self.patch_proj = nn.Linear(hidden_dim, embed_dim)

    @torch.compile
    def forward(self, x):
        B, C, H, W = x.shape
        assert (H, W) == self.img_size, f"Input size {H}x{W} != {self.img_size}"
        
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C) # (B, L, C)
        
        deltas = self.deltas.unsqueeze(0).expand(B, -1, -1) # (B, L, 3)
        cos_lat = self.cos_lat.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1) # (B, L, 1)
        
        features = torch.cat([x, deltas, cos_lat], dim=-1) # (B, L, C+4)
        
        # Pixel-wise projection to hidden_dim
        h = self.pixel_mlp(features) # (B, L, hidden_dim)
        
        w = self.weights.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1) # (B, L, 1)
        h_weighted = h * w # (B, L, hidden_dim)
        
        # Pooling to hidden_dim
        indices = self.nearest_indices.unsqueeze(0).unsqueeze(-1).expand(B, -1, h.shape[-1])
        
        N = self.num_patches
        out_sum = torch.zeros(B, N, h.shape[-1], device=x.device, dtype=x.dtype)
        out_sum.scatter_add_(1, indices, h_weighted)
        
        indices_w = self.nearest_indices.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)
        w_sum = torch.zeros(B, N, 1, device=x.device, dtype=x.dtype)
        w_sum.scatter_add_(1, indices_w, w)
        
        w_sum = torch.clamp(w_sum, min=1e-6)
        patch_features = out_sum / w_sum # (B, N, hidden_dim)
        
        # Project to embed_dim (Patch-wise)
        patch_embeddings = self.patch_proj(patch_features) # (B, N, embed_dim)
        
        return patch_embeddings


class FibonacciDecoder(nn.Module):
    def __init__(self, img_size=(352, 720), out_chans=3, embed_dim=768, fibonacci_n=1024, hidden_dim=256):
        super().__init__()
        self.img_size = img_size
        self.num_patches = fibonacci_n
        self.out_chans = out_chans
        
        self.sphere = FibonacciSphere(num_points=fibonacci_n)
        self.sphere.generate_points()
        
        grid_points, rlat = generate_grid_points(img_size)

        flat_grid = grid_points.reshape(-1, 3)
        dots = flat_grid @ self.sphere.points.T
        nearest_indices = np.argmax(dots, axis=1)
        
        self.register_buffer('nearest_indices', torch.from_numpy(nearest_indices).long())
        
        centers = self.sphere.points[nearest_indices]
        deltas = flat_grid - centers
        
        # Normalize deltas
        deltas = normalize_deltas(deltas, nearest_indices, fibonacci_n)
        
        self.register_buffer('deltas', torch.from_numpy(deltas).float())
        
        # Decomposed MLP for efficiency
        # Original: (embed_dim + 3) -> embed_dim -> out_chans
        # New: Decompose first layer into z_proj and delta_proj
        self.z_proj = nn.Linear(embed_dim, hidden_dim)
        self.delta_proj = nn.Linear(3, hidden_dim, bias=False)
        
        self.final_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, out_chans)
        )
        
    @torch.compile
    def forward(self, z):
        B, N, D = z.shape
        H, W = self.img_size
        
        # 1. Project Patch Embeddings (N is small)
        z_proj = self.z_proj(z) # (B, N, hidden_dim)
        
        # 2. Gather Projected Embeddings
        # Expand indices for hidden_dim
        indices = self.nearest_indices.unsqueeze(0).unsqueeze(-1).expand(B, -1, z_proj.shape[-1])
        z_broadcast = torch.gather(z_proj, 1, indices) # (B, L, hidden_dim)
        
        # 3. Project Deltas (Pixel-wise, but input dim is small)
        # self.deltas is (L, 3)
        delta_proj = self.delta_proj(self.deltas) # (L, hidden_dim)
        
        # 4. Combine and Finish
        # z_broadcast: (B, L, H)
        # delta_proj: (L, H) -> Broadcasts to (1, L, H) -> (B, L, H)
        inp = z_broadcast + delta_proj.unsqueeze(0)
        
        out = self.final_mlp(inp) # (B, L, Out_C)
        
        out = out.transpose(1, 2).reshape(B, -1, H, W)
        
        return out


class SimpleFibonacciPatchEmbed(nn.Module):
    def __init__(self, img_size=(352, 720), in_chans=3, embed_dim=768, fibonacci_n=1024):
        super().__init__()
        self.img_size = img_size
        self.num_patches = fibonacci_n
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        
        # Initialize Fibonacci Sphere
        self.sphere = FibonacciSphere(num_points=fibonacci_n)
        self.sphere.generate_points()
        
        # Generate Grid and Mapping
        grid_points, _ = generate_grid_points(img_size)
        flat_grid = grid_points.reshape(-1, 3)
        dots = flat_grid @ self.sphere.points.T
        nearest_indices = np.argmax(dots, axis=1) # (H*W,)
        
        # Pre-compute gather indices
        patch_pixel_indices = [[] for _ in range(fibonacci_n)]
        for pixel_idx, patch_idx in enumerate(nearest_indices):
            patch_pixel_indices[patch_idx].append(pixel_idx)
            
        max_points = max(len(p) for p in patch_pixel_indices)
        self.max_points = max_points
        
        gather_indices = np.zeros((fibonacci_n, max_points), dtype=np.int64)
        valid_mask = np.zeros((fibonacci_n, max_points), dtype=np.float32)
        
        for i, p in enumerate(patch_pixel_indices):
            length = len(p)
            gather_indices[i, :length] = p
            valid_mask[i, :length] = 1.0
            
        self.register_buffer('gather_indices', torch.from_numpy(gather_indices))
        self.register_buffer('valid_mask', torch.from_numpy(valid_mask))
        
        # MLP Input: max_points * (in_chans + 1) (features + mask)
        mlp_in_dim = max_points * (in_chans + 1)
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    @torch.compile
    def forward(self, x):
        B, C, H, W = x.shape
        assert (H, W) == self.img_size, f"Input size {H}x{W} != {self.img_size}"
        
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H*W, C) # (B, L, C)
        
        # Gather
        flat_indices = self.gather_indices.view(-1) # (N*M)
        x_gathered = x_flat[:, flat_indices, :] # (B, N*M, C)
        x_gathered = x_gathered.view(B, self.num_patches, self.max_points, C)
        
        # Masking
        mask = self.valid_mask.unsqueeze(0).unsqueeze(-1) # (1, N, M, 1)
        x_masked = x_gathered * mask
        
        # Append Mask
        mask_expanded = mask.expand(B, -1, -1, 1) # (B, N, M, 1)
        mlp_input = torch.cat([x_masked, mask_expanded], dim=-1) # (B, N, M, C+1)
        
        # Flatten and Project
        mlp_input = mlp_input.view(B, self.num_patches, -1) # (B, N, M*(C+1))
        return self.mlp(mlp_input)


class SimpleFibonacciPatchEmbedV2(nn.Module):
    def __init__(self, img_size=(352, 720), in_chans=3, embed_dim=768, fibonacci_n=1024):
        super().__init__()
        self.img_size = img_size
        self.num_patches = fibonacci_n
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        
        # Initialize Fibonacci Sphere
        self.sphere = FibonacciSphere(num_points=fibonacci_n)
        self.sphere.generate_points()
        
        # Generate Grid and Mapping
        grid_points, _ = generate_grid_points(img_size)
        flat_grid = grid_points.reshape(-1, 3)
        dots = flat_grid @ self.sphere.points.T
        nearest_indices = np.argmax(dots, axis=1) # (H*W,)
        
        # Pre-compute gather indices
        patch_pixel_indices = [[] for _ in range(fibonacci_n)]
        for pixel_idx, patch_idx in enumerate(nearest_indices):
            patch_pixel_indices[patch_idx].append(pixel_idx)
            
        # Bucketing
        self.buckets = {}
        # (name, max_size) - if max_size is -1, it's dynamic
        bucket_defs = [
            ('256', 256),
            ('512', 512),
            ('1024', 1024),
            ('large', -1)
        ]
        
        self.mlps = nn.ModuleDict()
        self.bucket_names = []
        
        for name, size_limit in bucket_defs:
            indices = []
            for i, p in enumerate(patch_pixel_indices):
                length = len(p)
                if name == '256' and length <= 256:
                    indices.append(i)
                elif name == '512' and 256 < length <= 512:
                    indices.append(i)
                elif name == '1024' and 512 < length <= 1024:
                    indices.append(i)
                elif name == 'large' and length > 1024:
                    indices.append(i)
            
            if not indices:
                continue
                
            self.bucket_names.append(name)
            
            # Determine size
            if size_limit == -1:
                max_len = max(len(patch_pixel_indices[i]) for i in indices)
                size = max_len
            else:
                size = size_limit
            
            self.buckets[name] = size
            num_in_bucket = len(indices)
            
            # Create buffers
            g_indices = np.zeros((num_in_bucket, size), dtype=np.int64)
            v_mask = np.zeros((num_in_bucket, size), dtype=np.float32)
            
            for k, patch_idx in enumerate(indices):
                p = patch_pixel_indices[patch_idx]
                length = len(p)
                g_indices[k, :length] = p
                v_mask[k, :length] = 1.0
                
            self.register_buffer(f'gather_indices_{name}', torch.from_numpy(g_indices))
            self.register_buffer(f'valid_mask_{name}', torch.from_numpy(v_mask))
            self.register_buffer(f'patch_indices_{name}', torch.tensor(indices, dtype=torch.long))
            
            # Create MLP
            # MLP Input: size * (in_chans + 1) (features + mask)
            mlp_in_dim = size * (in_chans + 1)
            
            self.mlps[name] = nn.Sequential(
                nn.Linear(mlp_in_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim)
            )

    @torch.compile
    def forward(self, x):
        B, C, H, W = x.shape
        assert (H, W) == self.img_size, f"Input size {H}x{W} != {self.img_size}"
        
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H*W, C) # (B, L, C)
        
        target_dtype = x.dtype
        if x.device.type == 'cuda' and torch.is_autocast_enabled('cuda'):
            target_dtype = torch.get_autocast_dtype('cuda')
            
        out = torch.zeros(B, self.num_patches, self.embed_dim, device=x.device, dtype=target_dtype)
        
        for name in self.bucket_names:
            module = self.mlps[name]
            g_indices = getattr(self, f'gather_indices_{name}')
            v_mask = getattr(self, f'valid_mask_{name}')
            p_indices = getattr(self, f'patch_indices_{name}')
            size = self.buckets[name]
            
            # Gather
            flat_indices = g_indices.view(-1)
            x_gathered = x_flat[:, flat_indices, :] # (B, N_k*size, C)
            x_gathered = x_gathered.view(B, -1, size, C) # (B, N_k, size, C)
            
            # Masking
            mask = v_mask.unsqueeze(0).unsqueeze(-1) # (1, N_k, size, 1)
            x_masked = x_gathered * mask
            
            # Append Mask
            mask_expanded = mask.expand(B, -1, -1, 1) # (B, N_k, size, 1)
            mlp_input = torch.cat([x_masked, mask_expanded], dim=-1) # (B, N_k, size, C+1)
            
            # Flatten and Project
            mlp_input = mlp_input.view(B, -1, size * (C + 1)) # (B, N_k, size*(C+1))
            res = module(mlp_input) # (B, N_k, D)
            
            # Scatter
            out[:, p_indices, :] = res.to(dtype=out.dtype)
            
        return out


class SimpleFibonacciDecoder(nn.Module):
    def __init__(self, img_size=(352, 720), out_chans=3, embed_dim=768, fibonacci_n=1024):
        super().__init__()
        self.img_size = img_size
        self.num_patches = fibonacci_n
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        
        # Re-compute mapping
        self.sphere = FibonacciSphere(num_points=fibonacci_n)
        self.sphere.generate_points()
        grid_points, _ = generate_grid_points(img_size)
        flat_grid = grid_points.reshape(-1, 3)
        dots = flat_grid @ self.sphere.points.T
        nearest_indices = np.argmax(dots, axis=1)
        
        patch_pixel_indices = [[] for _ in range(fibonacci_n)]
        for pixel_idx, patch_idx in enumerate(nearest_indices):
            patch_pixel_indices[patch_idx].append(pixel_idx)
            
        max_points = max(len(p) for p in patch_pixel_indices)
        self.max_points = max_points
        
        gather_indices = np.zeros((fibonacci_n, max_points), dtype=np.int64)
        valid_mask = np.zeros((fibonacci_n, max_points), dtype=np.float32)
        
        for i, p in enumerate(patch_pixel_indices):
            length = len(p)
            gather_indices[i, :length] = p
            valid_mask[i, :length] = 1.0
            
        self.register_buffer('gather_indices', torch.from_numpy(gather_indices))
        self.register_buffer('valid_mask', torch.from_numpy(valid_mask))
        
        mlp_out_dim = max_points * out_chans
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, mlp_out_dim)
        )
        
    @torch.compile
    def forward(self, z):
        B, N, D = z.shape
        out = self.mlp(z) # (B, N, M*C)
        out = out.view(B, N, self.max_points, self.out_chans)
        
        # Scatter back
        flat_indices = self.gather_indices.view(-1) # (N*M)
        flat_out = out.view(B, N*self.max_points, self.out_chans)
        
        mask_flat = self.valid_mask.view(-1).bool()
        valid_indices = flat_indices[mask_flat]
        valid_values = flat_out[:, mask_flat, :]
        
        L = self.img_size[0] * self.img_size[1]
        
        target_dtype = z.dtype
        if z.device.type == 'cuda' and torch.is_autocast_enabled('cuda'):
            target_dtype = torch.get_autocast_dtype('cuda')
            
        img_flat = torch.zeros(B, L, self.out_chans, device=z.device, dtype=target_dtype)
        img_flat[:, valid_indices, :] = valid_values.to(dtype=img_flat.dtype)
        
        imgs = img_flat.reshape(B, self.img_size[0], self.img_size[1], self.out_chans)
        return imgs.permute(0, 3, 1, 2)


class SimpleFibonacciDecoderV2(nn.Module):
    def __init__(self, img_size=(352, 720), out_chans=3, embed_dim=768, fibonacci_n=1024):
        super().__init__()
        self.img_size = img_size
        self.num_patches = fibonacci_n
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        
        # Re-compute mapping
        self.sphere = FibonacciSphere(num_points=fibonacci_n)
        self.sphere.generate_points()
        grid_points, _ = generate_grid_points(img_size)
        flat_grid = grid_points.reshape(-1, 3)
        dots = flat_grid @ self.sphere.points.T
        nearest_indices = np.argmax(dots, axis=1)
        
        patch_pixel_indices = [[] for _ in range(fibonacci_n)]
        for pixel_idx, patch_idx in enumerate(nearest_indices):
            patch_pixel_indices[patch_idx].append(pixel_idx)
            
        # Bucketing
        self.buckets = {}
        # (name, max_size) - if max_size is -1, it's dynamic
        bucket_defs = [
            ('256', 256),
            ('512', 512),
            ('1024', 1024),
            ('large', -1)
        ]
        
        self.mlps = nn.ModuleDict()
        self.bucket_names = []
        
        for name, size_limit in bucket_defs:
            indices = []
            for i, p in enumerate(patch_pixel_indices):
                length = len(p)
                if name == '256' and length <= 256:
                    indices.append(i)
                elif name == '512' and 256 < length <= 512:
                    indices.append(i)
                elif name == '1024' and 512 < length <= 1024:
                    indices.append(i)
                elif name == 'large' and length > 1024:
                    indices.append(i)
            
            if not indices:
                continue
                
            self.bucket_names.append(name)
            
            # Determine size
            if size_limit == -1:
                max_len = max(len(patch_pixel_indices[i]) for i in indices)
                size = max_len
            else:
                size = size_limit
            
            self.buckets[name] = size
            num_in_bucket = len(indices)
            
            # Create buffers
            g_indices = np.zeros((num_in_bucket, size), dtype=np.int64)
            v_mask = np.zeros((num_in_bucket, size), dtype=np.float32)
            
            for k, patch_idx in enumerate(indices):
                p = patch_pixel_indices[patch_idx]
                length = len(p)
                g_indices[k, :length] = p
                v_mask[k, :length] = 1.0
                
            self.register_buffer(f'gather_indices_{name}', torch.from_numpy(g_indices))
            self.register_buffer(f'valid_mask_{name}', torch.from_numpy(v_mask))
            self.register_buffer(f'patch_indices_{name}', torch.tensor(indices, dtype=torch.long))
            
            mlp_out_dim = size * out_chans
            
            self.mlps[name] = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, mlp_out_dim)
            )
        
    @torch.compile
    def forward(self, z):
        B, N, D = z.shape
        
        L = self.img_size[0] * self.img_size[1]
        
        target_dtype = z.dtype
        if z.device.type == 'cuda' and torch.is_autocast_enabled('cuda'):
            target_dtype = torch.get_autocast_dtype('cuda')
            
        img_flat = torch.zeros(B, L, self.out_chans, device=z.device, dtype=target_dtype)
        
        for name in self.bucket_names:
            module = self.mlps[name]
            g_indices = getattr(self, f'gather_indices_{name}')
            v_mask = getattr(self, f'valid_mask_{name}')
            p_indices = getattr(self, f'patch_indices_{name}')
            size = self.buckets[name]
            
            # Select relevant embeddings
            z_subset = z[:, p_indices, :] # (B, N_k, D)
            
            # Run MLP
            out = module(z_subset) # (B, N_k, size*C)
            out = out.view(B, -1, size, self.out_chans)
            
            # Scatter back
            flat_indices = g_indices.view(-1) # (N_k*size)
            flat_out = out.view(B, -1, self.out_chans)
            
            mask_flat = v_mask.view(-1).bool()
            
            valid_indices = flat_indices[mask_flat] # (num_valid_pixels)
            
            # We need to handle batch dimension correctly
            # Vectorized scatter:
            # We can't just use img_flat[:, valid_indices] because valid_indices are spatial indices
            # But valid_values are (B, num_valid_pixels, C)
            
            valid_values = flat_out[:, mask_flat, :] # (B, num_valid, C)
            
            # Use index_add_ or simple assignment if indices are unique (they are unique per image)
            # img_flat[:, valid_indices, :] = valid_values
            # This works because valid_indices are 1D indices into L, and we broadcast over B and C
            
            img_flat[:, valid_indices, :] = valid_values.to(dtype=img_flat.dtype)
        
        imgs = img_flat.reshape(B, self.img_size[0], self.img_size[1], self.out_chans)
        return imgs.permute(0, 3, 1, 2)


if __name__ == "__main__":
    # Create an instance with 1024 points
    fib_sphere = FibonacciSphere(num_points=1024)
    
    # Generate points
    points = fib_sphere.generate_points()
    print(f"Generated {points.shape[0]} points on the unit sphere.")
    
    # Visualize and save 3D plot
    save_file_3d = "fibonacci_sphere_1024_3d.png"
    fib_sphere.plot(save_path=save_file_3d, show=False)
    
    # Visualize and save Map plot
    save_file_map = "fibonacci_sphere_1024_map.png"
    fib_sphere.plot_on_map(save_path=save_file_map, show=False)

    # Visualize and save Distribution plot
    save_file_dist = "fibonacci_sphere_1024_dist.png"
    fib_sphere.plot_grid_distribution(save_path=save_file_dist, show=False)
