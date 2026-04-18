import torch
import torch.nn as nn
import json
import os
from modules.networks.model import DANet_models
from modules.flow_matching.losses import compute_denoiser_loss, GradientLoss


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        # Support both square and non-square images
        if hasattr(args, 'img_height') and hasattr(args, 'img_width'):
            input_size = (args.img_height, args.img_width)
            self.img_height = args.img_height
            self.img_width = args.img_width
        else:
            input_size = args.img_size
            self.img_height = args.img_size
            self.img_width = args.img_size
            
        self.use_mask_as_channel = getattr(args, 'use_mask_as_channel', False)
        
        in_channels = 4
        if self.use_mask_as_channel:
            in_channels += 1

        self.net = DANet_models[args.model](
            input_size=input_size,
            in_channels=in_channels,
            out_channels=4,  # Always output 4 ocean variables
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            n_depth_levels=getattr(args, 'n_depth_levels', 50),
            zero_land_output=getattr(args, 'zero_land_output', False),
            patch_embed_method=getattr(args, 'patch_embed_method', 'standard'),
            fibonacci_n=getattr(args, 'fibonacci_n', 1024),
            fibonacci_pos_embed_type=getattr(args, 'fibonacci_pos_embed_type', 'fourier'),
            argo_fusion_method=getattr(args, 'argo_fusion_method', 'crossattention'),
            use_time_embed=getattr(args, 'use_time_embed', False)
        )

        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale
        self.grad_loss_weight = getattr(args, 'grad_loss_weight', 0.0)
        
        if self.grad_loss_weight > 0.0:
            self.grad_criterion = GradientLoss()
        else:
            self.grad_criterion = None

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps

        # Classifier-Free Guidance (CFG) params
        self.cfg_scale = getattr(args, 'cfg_scale', 1.0)
        self.cfg_dropout_prob = getattr(args, 'cfg_dropout_prob', 0.0)


    def sample_t(self, n: int, device=None):
        # z = torch.randn(n, device=device) * self.P_std + self.P_mean
        # return torch.sigmoid(z)
        return torch.rand(n, device=device)

    def forward(self, x, argo_data=None, depth_index=None, mask=None, time_encoding=None):
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)
        
        # Prepare input for the network
        net_input = z
        if self.use_mask_as_channel:
            if mask is None:
                raise ValueError("Mask is required but not provided")
            net_input = torch.cat([z, mask], dim=1)

        # Classifier-Free Guidance: Randomly drop conditions during training
        if self.training and self.cfg_dropout_prob > 0.0:
            argo_data, depth_index, time_encoding = self._apply_cfg_dropout(
                argo_data, depth_index, time_encoding, x.device
            )

        x_pred = self.net(net_input, t.flatten(), argo_data=argo_data, depth_index=depth_index, mask=mask, time_encoding=time_encoding)
        
        return compute_denoiser_loss(v, x_pred, z, t, self.t_eps, mask, self.grad_loss_weight, self.grad_criterion)

    def _apply_cfg_dropout(self, argo_data, depth_index, time_encoding, device):
        """
        Apply CFG dropout: randomly drop conditions during training.
        Only drop Argo data and time encoding, keep depth_index and mask.
        """
        batch_size = 1  # Will be determined from actual data
        if argo_data is not None and 'temperature' in argo_data:
            batch_size = argo_data['temperature'].shape[0]
        elif depth_index is not None:
            batch_size = depth_index.shape[0]
        elif time_encoding is not None:
            batch_size = time_encoding.shape[0]
        else:
            return argo_data, depth_index, time_encoding
        
        # Create dropout mask
        dropout_mask = torch.rand(batch_size, device=device) < self.cfg_dropout_prob
        
        if dropout_mask.any():
            # Drop argo_data for masked samples
            if argo_data is not None:
                argo_data = self._mask_argo_data(argo_data, dropout_mask)
            
            # Drop time_encoding for masked samples (only drop if all samples are dropped)
            if time_encoding is not None:
                if dropout_mask.all():
                    time_encoding = None
        
        return argo_data, depth_index, time_encoding

    def _mask_argo_data(self, argo_data, dropout_mask):
        """Mask out Argo data for samples where dropout is applied."""
        masked_data = {}
        for key, value in argo_data.items():
            if isinstance(value, torch.Tensor):
                masked_value = value.clone()
                # For dropout samples, set mask to False (no valid Argo data)
                if key == 'mask':
                    masked_value[dropout_mask] = False
                else:
                    # Zero out other features for dropped samples
                    masked_value[dropout_mask] = 0.0
                masked_data[key] = masked_value
            else:
                masked_data[key] = value
        return masked_data

    @torch.no_grad()
    def generate(self, batch_size, device, argo_data=None, depth_index=None, mask=None, time_encoding=None):
        z = self.noise_scale * torch.randn(batch_size, 4, self.img_height, self.img_width, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, batch_size, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, argo_data=argo_data, depth_index=depth_index, mask=mask, time_encoding=time_encoding)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], argo_data=argo_data, depth_index=depth_index, mask=mask, time_encoding=time_encoding)
        return z

    @torch.no_grad()
    def generate_cfg(self, batch_size, device, argo_data=None, depth_index=None, mask=None, time_encoding=None):
        """
        Generate samples with Classifier-Free Guidance (CFG).
        
        CFG formula: v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
        """
        # If cfg_scale is 1.0, no guidance needed
        if self.cfg_scale <= 1.0:
            return self.generate(batch_size, device, argo_data, depth_index, mask, time_encoding)
        
        z = self.noise_scale * torch.randn(batch_size, 4, self.img_height, self.img_width, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, batch_size, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step_cfg
        elif self.method == "heun":
            stepper = self._heun_step_cfg
        else:
            raise NotImplementedError

        # ode with CFG
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, argo_data=argo_data, depth_index=depth_index, mask=mask, time_encoding=time_encoding)
        # last step euler with CFG
        z = self._euler_step_cfg(z, timesteps[-2], timesteps[-1], argo_data=argo_data, depth_index=depth_index, mask=mask, time_encoding=time_encoding)
        return z

    @torch.no_grad()
    def _forward_sample_cfg(self, z, t, argo_data=None, depth_index=None, mask=None, time_encoding=None):
        """
        Forward pass with CFG: compute both conditional and unconditional predictions.
        Returns guided velocity prediction.
        Only drop Argo data and time encoding for unconditional prediction, keep depth_index and mask.
        """
        net_input = z
        if self.use_mask_as_channel:
            if mask is None:
                raise ValueError("Mask is required but not provided")
            net_input = torch.cat([z, mask], dim=1)
        
        # Conditional prediction
        x_pred_cond = self.net(net_input, t.flatten(), argo_data=argo_data, depth_index=depth_index, mask=mask, time_encoding=time_encoding)
        v_pred_cond = (x_pred_cond - z) / (1.0 - t).clamp_min(self.t_eps)
        
        # Unconditional prediction: only drop Argo data and time encoding, keep depth_index and mask
        x_pred_uncond = self.net(net_input, t.flatten(), argo_data=None, depth_index=depth_index, mask=mask, time_encoding=None)
        v_pred_uncond = (x_pred_uncond - z) / (1.0 - t).clamp_min(self.t_eps)
        
        # CFG guidance: v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
        v_pred = v_pred_uncond + self.cfg_scale * (v_pred_cond - v_pred_uncond)
        return v_pred

    @torch.no_grad()
    def _euler_step_cfg(self, z, t, t_next, argo_data=None, depth_index=None, mask=None, time_encoding=None):
        v_pred = self._forward_sample_cfg(z, t, argo_data=argo_data, depth_index=depth_index, mask=mask, time_encoding=time_encoding)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step_cfg(self, z, t, t_next, argo_data=None, depth_index=None, mask=None, time_encoding=None):
        v_pred_t = self._forward_sample_cfg(z, t, argo_data=argo_data, depth_index=depth_index, mask=mask, time_encoding=time_encoding)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample_cfg(z_next_euler, t_next, argo_data=argo_data, depth_index=depth_index, mask=mask, time_encoding=time_encoding)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _forward_sample(self, z, t, argo_data=None, depth_index=None, mask=None, time_encoding=None):
        net_input = z
        if self.use_mask_as_channel:
            if mask is None:
                raise ValueError("Mask is required but not provided")
            net_input = torch.cat([z, mask], dim=1)
            
        x_pred = self.net(net_input, t.flatten(), argo_data=argo_data, depth_index=depth_index, mask=mask, time_encoding=time_encoding)
        v_pred = (x_pred - z) / (1.0 - t).clamp_min(self.t_eps)
        return v_pred

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, argo_data=None, depth_index=None, mask=None, time_encoding=None):
        v_pred = self._forward_sample(z, t, argo_data=argo_data, depth_index=depth_index, mask=mask, time_encoding=time_encoding)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, argo_data=None, depth_index=None, mask=None, time_encoding=None):
        v_pred_t = self._forward_sample(z, t, argo_data=argo_data, depth_index=depth_index, mask=mask, time_encoding=time_encoding)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, argo_data=argo_data, depth_index=depth_index, mask=mask, time_encoding=time_encoding)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
