import torch
import torch.nn as nn
import torch.nn.functional as F


def get_lat_weights(H: int, device: torch.device) -> torch.Tensor:
    """
    Generate latitude weights based on the grid specifications.
    Grid: 352 points.
    Indices 0-10: Padding.
    Index 11: -80 degrees.
    Step: 0.5 degrees.
    """
    # Create indices
    indices = torch.arange(H, device=device, dtype=torch.float32)
    
    # Calculate latitudes
    # Index 11 is -80.0
    # lat = -80.0 + (index - 11) * 0.5
    lats = -80.0 + (indices - 11) * 0.5
    
    # Convert to radians
    lats_rad = torch.deg2rad(lats)
    
    # Calculate weights (cosine of latitude)
    weights = torch.cos(lats_rad)
    
    # Reshape for broadcasting: (1, 1, H, 1)
    return weights.view(1, 1, H, 1)


class GradientLoss(nn.Module):
    """
    Physics-aware Gradient Matching Loss (Sobolev Loss).
    Features:
    1. Sobel operator for smooth gradient calculation.
    2. Mask Erosion: Ignores gradients at the coastline to prevent artificial 
       jumps between ocean values and land (0.0).
    3. Replicate Padding: Handles map boundaries correctly.
    4. Latitude Weighting: Accounts for grid cell area changes.
    """
    def __init__(self):
        super().__init__()
        # Define Sobel kernels
        # Shape: (1, 1, 3, 3) for broadcasting
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # Define Erosion kernel (3x3 ones) to detect safe ocean interior
        erode_k = torch.ones(1, 1, 3, 3, dtype=torch.float32)

        # Register as buffers so they move to GPU automatically with the model
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        self.register_buffer('erode_kernel', erode_k)

    def erode_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Erodes the binary mask. 
        A pixel is kept only if itself and all 8 neighbors are valid (Ocean).
        This removes the coastline border where gradients are numerically unstable.
        """
        if mask is None:
            return None
        
        # mask shape: (B, 1, H, W)
        # Sum neighbors using convolution
        # padding=1 ensures we check the 3x3 surrounding area
        with torch.no_grad():
            neighbor_sum = F.conv2d(mask.float(), self.erode_kernel, padding=1, bias=None)
            
            # If sum > 8.5 (approx 9), it means all 9 pixels in the 3x3 block are 1 (Ocean).
            # This is a safe internal zone to calculate gradients.
            safe_mask = (neighbor_sum > 8.5).float()
            
        return safe_mask

    def forward(self, v_target: torch.Tensor, v_pred: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            v_target: Ground truth tensor (B, C, H, W). Assumes indices 2,3 are U,V.
            v_pred: Predicted tensor (B, C, H, W).
            mask: Land mask (B, 1, H, W). 1=Water, 0=Land.
        """
        # Ensure we have enough channels for U and V (indices 2 and 3)
        if v_target.shape[1] < 4:
            return torch.tensor(0.0, device=v_target.device)

        # 1. Select Velocity Channels (U, V)
        # Shape: (B, 2, H, W)
        target_vel = v_target[:, 2:4, :, :]
        pred_vel = v_pred[:, 2:4, :, :]
        
        B, C_vel, H, W = target_vel.shape # C_vel should be 2

        # 2. Prepare Kernels for Grouped Convolution
        # We repeat the kernels to match the number of velocity channels (2)
        # Shape: (2, 1, 3, 3)
        kx = self.sobel_x.repeat(C_vel, 1, 1, 1)
        ky = self.sobel_y.repeat(C_vel, 1, 1, 1)

        # 3. Compute Gradients
        # Pad with 1 pixel on all sides (left, right, top, bottom)
        target_vel_padded = F.pad(target_vel, (1, 1, 1, 1), mode='replicate')
        pred_vel_padded = F.pad(pred_vel, (1, 1, 1, 1), mode='replicate')
        
        # padding=0 because we already padded
        target_dx = F.conv2d(target_vel_padded, kx, padding=0, groups=C_vel)
        target_dy = F.conv2d(target_vel_padded, ky, padding=0, groups=C_vel)
        
        pred_dx = F.conv2d(pred_vel_padded, kx, padding=0, groups=C_vel)
        pred_dy = F.conv2d(pred_vel_padded, ky, padding=0, groups=C_vel)

        # 4. Compute Squared Error of Gradients
        grad_loss = (target_dx - pred_dx) ** 2 + (target_dy - pred_dy) ** 2

        # 5. Apply Latitude Weighting
        lat_weights = get_lat_weights(H, v_target.device)
        grad_loss = grad_loss * lat_weights

        # 6. Apply Masking
        if mask is not None:
            # IMPORTANT: Use the eroded mask.
            # We strictly exclude coastline pixels to avoid the "Water-to-Land" gradient jump.
            safe_mask = self.erode_mask(mask)
            
            # Expand mask to match (B, 2, H, W)
            safe_mask_expanded = safe_mask.expand_as(grad_loss)
            
            # Apply mask
            grad_loss = grad_loss * safe_mask_expanded
            
            # Normalize by the sum of weights in the valid (safe) area
            # We need to weight the mask sum by latitude weights as well
            weights_expanded = lat_weights.expand_as(grad_loss)
            weighted_mask_sum = (safe_mask_expanded * weights_expanded).sum(dim=(1, 2, 3))
            
            # Average over batch
            loss_val = (grad_loss.sum(dim=(1, 2, 3)) / (weighted_mask_sum + 1e-6)).mean()
        else:
            # Weighted mean over all dimensions
            sum_grad_loss = grad_loss.sum()
            total_weights = lat_weights.expand_as(grad_loss).sum()
            loss_val = sum_grad_loss / (total_weights + 1e-6)

        return loss_val


def compute_denoiser_loss(
    v: torch.Tensor,
    x_pred: torch.Tensor,
    z: torch.Tensor,
    t: torch.Tensor,
    t_eps: float,
    mask: torch.Tensor | None,
    grad_loss_weight: float = 0.0,
    grad_criterion: GradientLoss | None = None # Pass the initialized module here
) -> torch.Tensor:
    
    # 1. Recover prediction from noise (Diffusion logic)
    v_pred = (x_pred - z) / (1 - t).clamp_min(t_eps)
    
    # 2. Standard MSE Loss (Values)
    loss = (v - v_pred) ** 2

    if mask is not None:
        loss_masked = loss * mask
        sum_loss = loss_masked.sum(dim=(1, 2, 3))
        
        # Calculate normalization factor (count of valid pixels)
        mask_expanded = mask.expand_as(loss)
        sum_mask = mask_expanded.sum(dim=(1, 2, 3))
        
        mse_loss = (sum_loss / (sum_mask + 1e-6)).mean()
    else:
        mse_loss = loss.mean()

    # 3. Gradient Matching Loss (Physics/Structure)
    grad_loss = torch.tensor(0.0, device=v.device)
    
    if grad_loss_weight > 0.0:
        # Initialize criterion locally if not passed (Fallback)
        # NOTE: For efficiency, instantiate GradientLoss() once outside the training loop 
        # and pass it into this function.
        if grad_criterion is None:
            grad_criterion = GradientLoss().to(v.device)
            
        grad_loss = grad_criterion(v, v_pred, mask)
        total_loss = mse_loss + grad_loss_weight * grad_loss
    else:
        total_loss = mse_loss

    return total_loss, mse_loss, grad_loss