import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from modules.util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm
from modules.networks.fibonacci import (
    FibonacciSphere, Fib3DPosEmbedder, FibonacciPatchEmbed, FibonacciDecoder,
    SimpleFibonacciPatchEmbed, SimpleFibonacciPatchEmbedV2,
    SimpleFibonacciDecoder, SimpleFibonacciDecoderV2
)

class IdentityRotary(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)





class BottleneckPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        # Support both single int and tuple for img_size
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        elif isinstance(img_size, (tuple, list)):
            img_size = tuple(img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DateEmbedder(nn.Module):
    """
    Embeds cyclical date encoding (sin, cos) into vector representations.
    Input is already in frequency domain [sin, cos], so we directly apply MLP.
    """
    def __init__(self, hidden_size):
        super().__init__()
        # Input is 2D [sin, cos], directly map to hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, date_encoding):
        """
        date_encoding: (N, 2) tensor with [sin_val, cos_val]
        """
        return self.mlp(date_encoding)


class CoordsEmbedder(nn.Module):
    """
    Embeds spatial coordinates (lat/lon in degrees) using sinusoidal embeddings.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, max_freq_power=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_freq_power = max_freq_power

    def forward(self, x):
        # x: (N,) tensor in degrees
        # Convert to radians
        x_rad = x * math.pi / 180.0
        
        dim = self.frequency_embedding_size
        half = dim // 2
        
        # Create frequencies: from 2^0 to 2^max_freq_power
        # Use logspace to cover the range evenly
        freqs = torch.logspace(0, self.max_freq_power, steps=half, base=2.0, device=x.device)
        
        args = x_rad.unsqueeze(-1) * freqs.unsqueeze(0) # (N, half)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return self.mlp(embedding)





class ArgoEncoder(nn.Module):
    """
    Encodes Argo profiles into tokens with spatio-temporal embeddings.
    """
    def __init__(self, hidden_size, n_depth_levels=50, frequency_embedding_size=256):
        super().__init__()
        print(f"ArgoEncoder init: hidden_size={hidden_size}, n_depth_levels={n_depth_levels}")
        # Input features: T, S, MaskT, MaskS per depth level
        self.input_dim = n_depth_levels * 4 
        self.profile_mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Spatio-temporal embeddings
        self.lat_embedder = CoordsEmbedder(hidden_size, frequency_embedding_size)
        self.lon_embedder = CoordsEmbedder(hidden_size, frequency_embedding_size)
        self.time_embedder = TimestepEmbedder(hidden_size, frequency_embedding_size)
    
    @torch.compile  
    def forward(self, argo_data):
        """
        argo_data: dict with batched tensors
        """
        # Flatten profile data: (B, N, D) -> (B, N, D*4)
        B, N, D = argo_data['temperature'].shape
        
        # Stack features: T, S, MaskT, MaskS
        features = torch.cat([
            argo_data['temperature'],
            argo_data['salinity'],
            argo_data['temperature_mask'],
            argo_data['salinity_mask']
        ], dim=2) # (B, N, D*4)
        
        # Encode profiles
        tokens = self.profile_mlp(features) # (B, N, Hidden)
        
        # Add embeddings
        # Reshape scalars to (B*N) for embedder then reshape back
        lat = argo_data['latitude'].flatten()
        lon = argo_data['longitude'].flatten()
        time = argo_data['relative_time'].flatten()
        
        lat_emb = self.lat_embedder(lat).reshape(B, N, -1)
        lon_emb = self.lon_embedder(lon).reshape(B, N, -1)
        time_emb = self.time_embedder(time).reshape(B, N, -1)
        
        tokens = tokens + lat_emb + lon_emb + time_emb
        
        return tokens


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context, context_mask=None):
        B, N, C = x.shape
        # x is query
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # context is key/value (Argo tokens)
        # context: (B, S, C) where S is num profiles
        B_ctx, S, C_ctx = context.shape
        kv = self.kv(context).reshape(B_ctx, S, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Scaled dot product attention
        scale = (C // self.num_heads) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale # (B, Heads, N, S)
        
        if context_mask is not None:
            # context_mask: (B, S) - True for valid, False for padding
            # We want to mask out padding (set to -inf)
            # Mask shape needs to be broadcastable to (B, Heads, N, S)
            # mask: (B, 1, 1, S)
            mask = context_mask.view(B_ctx, 1, 1, S)
            attn = attn.masked_fill(~mask, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype).cuda()

    with torch.cuda.amp.autocast(enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """
    The final layer of DANet.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
    
    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DANetBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, argo_fusion_method='crossattention'):
        super().__init__()
        self.argo_fusion_method = argo_fusion_method
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        
        # Cross Attention
        if self.argo_fusion_method == 'crossattention':
            self.norm_cross = RMSNorm(hidden_size, eps=1e-6)
            self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                                             attn_drop=attn_drop, proj_drop=proj_drop)
            modulation_factor = 9
        else:
            modulation_factor = 6
        
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        
        # Add gates for Cross Attention (gate_cross)
        # 3 (Self) + 3 (Cross) + 3 (MLP) = 9
        # Or 3 (Self) + 3 (MLP) = 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, modulation_factor * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c, argo_tokens=None, argo_mask=None, feat_rope=None):
        if self.argo_fusion_method == 'crossattention':
            shift_msa, scale_msa, gate_msa, shift_cross, scale_cross, gate_cross, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=-1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        
        # Self Attention
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        
        # Cross Attention (only if argo_tokens provided)
        if self.argo_fusion_method == 'crossattention' and argo_tokens is not None:
            x = x + gate_cross.unsqueeze(1) * self.cross_attn(modulate(self.norm_cross(x), shift_cross, scale_cross), argo_tokens, argo_mask)
            
        # MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x








class DANet(nn.Module): 

    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        out_channels=None,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        bottleneck_dim=128,
        n_depth_levels=50,
        zero_land_output=False,
        patch_embed_method='standard', # 'standard', 'fibonacci', 'simple_fibonacci', 'simple_fibonacci_v2'
        fibonacci_n=1024,
        fibonacci_pos_embed_type='fourier', # 'fourier' or 'mlp'
        argo_fusion_method='crossattention', # 'crossattention' or 'geo_fusion'
        use_time_embed=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.zero_land_output = zero_land_output
        self.patch_embed_method = patch_embed_method
        self.fibonacci_pos_embed_type = fibonacci_pos_embed_type
        self.argo_fusion_method = argo_fusion_method
        self.use_time_embed = use_time_embed

        # GLORYS12 Grid Parameters
        self.glorys_lat_min = -80.0
        self.glorys_lat_step = 0.5
        self.glorys_lat_pad = 11
        self.glorys_lon_min = -180.0
        self.glorys_lon_step = 0.5

        # time embed
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # depth embed
        self.depth_embedder = TimestepEmbedder(hidden_size)
        
        # date (cyclical time) embed - input is 2D (sin, cos)
        if use_time_embed:
            self.date_embedder = DateEmbedder(hidden_size)
        
        # argo encoder
        self.argo_encoder = ArgoEncoder(hidden_size, n_depth_levels=n_depth_levels)

        # linear embed
        if patch_embed_method != 'standard':
            if isinstance(input_size, int):
                img_size = (input_size, input_size)
            else:
                img_size = input_size
            
            if patch_embed_method == 'simple_fibonacci_v2':
                self.x_embedder = SimpleFibonacciPatchEmbedV2(img_size, in_channels, hidden_size, fibonacci_n)
                self.fibonacci_decoder = SimpleFibonacciDecoderV2(img_size, self.out_channels, hidden_size, fibonacci_n)
            elif patch_embed_method == 'simple_fibonacci':
                self.x_embedder = SimpleFibonacciPatchEmbed(img_size, in_channels, hidden_size, fibonacci_n)
                self.fibonacci_decoder = SimpleFibonacciDecoder(img_size, self.out_channels, hidden_size, fibonacci_n)
            elif patch_embed_method == 'fibonacci':
                self.x_embedder = FibonacciPatchEmbed(img_size, in_channels, hidden_size, fibonacci_n, hidden_dim=256)
                self.fibonacci_decoder = FibonacciDecoder(img_size, self.out_channels, hidden_size, fibonacci_n, hidden_dim=256)
            else:
                raise ValueError(f"Unknown patch_embed_method: {patch_embed_method}")

            num_patches = fibonacci_n
            
            # Coordinate-based position embedding for Fibonacci
            if self.fibonacci_pos_embed_type == 'fourier':
                self.fib_pos_mlp = Fib3DPosEmbedder(hidden_size)
            else:
                self.fib_pos_mlp = nn.Sequential(
                    nn.Linear(3, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size)
                )
            
            # Register sphere points as buffer for position embedding
            self.register_buffer('sphere_points', torch.from_numpy(self.x_embedder.sphere.points).float())
            
        else:
            self.x_embedder = BottleneckPatchEmbed(input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)
            num_patches = self.x_embedder.num_patches
            self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
            # use fixed sin-cos embedding
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
            
            # Initialize patch coordinates for geo_fusion
            # self._init_patch_coordinates()

        # rope - support non-square images
        half_head_dim = hidden_size // num_heads // 2
        # For non-square images, pass the actual (h, w) dimensions to RoPE
        if patch_embed_method != 'standard':
            # Use Identity Rotary (No RoPE) for Fibonacci
            self.feat_rope = IdentityRotary()
        else:
            if isinstance(input_size, int):
                hw_seq_len = input_size // patch_size
            else:
                hw_seq_len = (input_size[0] // patch_size, input_size[1] // patch_size)
        
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
                num_cls_token=0
            )

        # transformer
        self.blocks = nn.ModuleList([
            DANetBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     argo_fusion_method=argo_fusion_method)
            for i in range(depth)
        ])

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        if self.patch_embed_method == 'standard':
            # Initialize (and freeze) pos_embed by sin-cos embedding:
            # For non-square images, we need to generate position embeddings accordingly
            h_patches = self.x_embedder.img_size[0] // self.patch_size
            w_patches = self.x_embedder.img_size[1] // self.patch_size
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], 
                (h_patches, w_patches)
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
            w1 = self.x_embedder.proj1.weight.data
            nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
            w2 = self.x_embedder.proj2.weight.data
            nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj2.bias, 0)
            
            # Zero-out output layers:
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

            nn.init.constant_(self.final_layer.linear.weight, 0)
            nn.init.constant_(self.final_layer.linear.bias, 0)
        else:
            # Fibonacci Init
            if self.patch_embed_method == 'simple_fibonacci_v2':
                # SimpleFibonacciPatchEmbedV2 has self.mlps
                for name, mlp in self.x_embedder.mlps.items():
                    nn.init.xavier_uniform_(mlp[0].weight)
                    nn.init.xavier_uniform_(mlp[2].weight)
                
                # SimpleFibonacciDecoderV2 has self.mlps
                for name, mlp in self.fibonacci_decoder.mlps.items():
                    nn.init.xavier_uniform_(mlp[0].weight)
                    
                    # Zero-out last layer of decoder (mlp[2])
                    nn.init.constant_(mlp[2].weight, 0)
                    nn.init.constant_(mlp[2].bias, 0)
            elif self.patch_embed_method == 'simple_fibonacci':
                # SimpleFibonacciPatchEmbed has self.mlp
                nn.init.xavier_uniform_(self.x_embedder.mlp[0].weight)
                nn.init.xavier_uniform_(self.x_embedder.mlp[2].weight)
                
                # SimpleFibonacciDecoder has self.mlp
                nn.init.xavier_uniform_(self.fibonacci_decoder.mlp[0].weight)
                nn.init.constant_(self.fibonacci_decoder.mlp[2].weight, 0)
                nn.init.constant_(self.fibonacci_decoder.mlp[2].bias, 0)
            elif self.patch_embed_method == 'fibonacci':
                nn.init.xavier_uniform_(self.x_embedder.pixel_mlp[0].weight)
                nn.init.xavier_uniform_(self.x_embedder.patch_proj.weight)
                
                # Init Decoder (Decomposed)
                nn.init.xavier_uniform_(self.fibonacci_decoder.z_proj.weight)
                nn.init.xavier_uniform_(self.fibonacci_decoder.delta_proj.weight)
                
                # Zero-out last layer of decoder
                nn.init.constant_(self.fibonacci_decoder.final_mlp[1].weight, 0)
                nn.init.constant_(self.fibonacci_decoder.final_mlp[1].bias, 0)
            
            # Init pos embed MLP
            if self.fibonacci_pos_embed_type == 'fourier':
                nn.init.normal_(self.fib_pos_mlp.mlp[0].weight, std=0.02)
                nn.init.normal_(self.fib_pos_mlp.mlp[2].weight, std=0.02)
            else:
                nn.init.normal_(self.fib_pos_mlp[0].weight, std=0.02)
                nn.init.normal_(self.fib_pos_mlp[2].weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        nn.init.normal_(self.depth_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.depth_embedder.mlp[2].weight, std=0.02)
        
        if self.use_time_embed:
            nn.init.normal_(self.date_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.date_embedder.mlp[2].weight, std=0.02)

        # Initialize Argo Encoder
        nn.init.xavier_uniform_(self.argo_encoder.profile_mlp[0].weight)
        nn.init.xavier_uniform_(self.argo_encoder.profile_mlp[2].weight)
        nn.init.normal_(self.argo_encoder.lat_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.argo_encoder.lat_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.argo_encoder.lon_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.argo_encoder.lon_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.argo_encoder.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.argo_encoder.time_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    @torch.compile
    def unpatchify(self, x, p):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        # For non-square images, calculate h and w from the image size
        h = self.x_embedder.img_size[0] // p
        w = self.x_embedder.img_size[1] // p
        assert h * w == x.shape[1], f"Patch count mismatch: {h}*{w}={h*w} != {x.shape[1]}"

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    @torch.compile
    def fuse_argo_geo(self, x, argo_data):
        """
        Fuse Argo data into patch tokens based on geographic coordinates.
        x: (B, N, D) - Patch tokens
        argo_data: dict - Argo data
        """
        B, N, D = x.shape
        
        # Get Argo info
        argo_lat = argo_data['latitude'] # (B, M)
        argo_lon = argo_data['longitude'] # (B, M)
        argo_mask = argo_data['mask'] # (B, M)
        
        # Get Argo tokens
        argo_tokens = self.argo_encoder(argo_data) # (B, M, D)
        
        # Compute patch indices from lat/lon
        # 1. Map lat/lon to pixel coordinates (including padding)
        pixel_h = (argo_lat - self.glorys_lat_min) / self.glorys_lat_step + self.glorys_lat_pad
        pixel_w = (argo_lon - self.glorys_lon_min) / self.glorys_lon_step
        
        # 2. Map pixel coordinates to patch indices
        patch_h = torch.floor(pixel_h / self.patch_size).long()
        patch_w = torch.floor(pixel_w / self.patch_size).long()
        
        # 3. Filter valid indices
        H_patches = self.x_embedder.img_size[0] // self.patch_size
        W_patches = self.x_embedder.img_size[1] // self.patch_size
        
        valid_idx = (patch_h >= 0) & (patch_h < H_patches) & \
                    (patch_w >= 0) & (patch_w < W_patches) & \
                    argo_mask
        
        # 4. Compute linear patch index
        patch_idx = patch_h * W_patches + patch_w # (B, M)
        
        # 5. Scatter sum
        # Flatten batch and patch dims for scatter
        # Global index: b * N + patch_idx
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand_as(patch_idx)
        flat_idx = batch_idx * N + patch_idx
        
        # Filter valid tokens and indices
        flat_idx_valid = flat_idx[valid_idx] # (K,)
        tokens_valid = argo_tokens[valid_idx] # (K, D)
        
        if flat_idx_valid.numel() > 0:
            # Prepare buffers
            x_flat = x.reshape(-1, D)
            delta_sum = torch.zeros_like(x_flat)
            delta_count = torch.zeros(B*N, 1, device=x.device)
            
            # Accumulate tokens and counts
            delta_sum.index_add_(0, flat_idx_valid, tokens_valid)
            delta_count.index_add_(0, flat_idx_valid, torch.ones(flat_idx_valid.size(0), 1, device=x.device))
            
            # Compute average
            # Avoid division by zero
            delta_count = delta_count.clamp(min=1.0)
            delta_avg = delta_sum / delta_count
            
            # Add to x
            x = x + delta_avg.view(B, N, D)
            
        return x

    def forward(self, x, t, argo_data=None, depth_index=None, mask=None, time_encoding=None):
        """
        x: (N, C, H, W)
        t: (N,)
        argo_data: dict containing padded Argo tensors
        depth_index: (N,)
        mask: (N, 1, H, W) or None. 1 for Ocean, 0 for Land.
        time_encoding: (N, 2) or None. Cyclical time encoding [sin, cos] for date.
        """
        # time embeddings
        t_emb = self.t_embedder(t)
        
        # depth embeddings
        if depth_index is not None:
            d_emb = self.depth_embedder(depth_index)
            c = t_emb + d_emb
        else:
            c = t_emb
        
        # date (cyclical time) embeddings
        if self.use_time_embed and time_encoding is not None:
            # time_encoding is (N, 2) with [sin_val, cos_val]
            # We treat it similarly to depth_index - a scalar-like conditioning
            date_emb = self.date_embedder(time_encoding)
            c = c + date_emb
            
        # argo embeddings
        argo_tokens = None
        argo_mask = None
        if argo_data is not None:
            if self.argo_fusion_method == 'geo_fusion':
                pass # Will be handled after x_embedder
            else:
                # Standard cross-attention
                argo_tokens = self.argo_encoder(argo_data)
                argo_mask = argo_data['mask'] if 'mask' in argo_data else None

        # forward DANet
        x = self.x_embedder(x)
        
        if self.patch_embed_method != 'standard':
            # Add coordinate-based position embedding
            # sphere_points: (N, 3)
            pos_emb = self.fib_pos_mlp(self.sphere_points) # (N, D)
            x = x + pos_emb.unsqueeze(0)
        else:
            x += self.pos_embed

        # Apply Geo Fusion if enabled
        if argo_data is not None and self.argo_fusion_method == 'geo_fusion':
             x = self.fuse_argo_geo(x, argo_data)

        for i, block in enumerate(self.blocks):
            x = block(x, c, argo_tokens, argo_mask, self.feat_rope)

        if self.patch_embed_method != 'standard':
            output = self.fibonacci_decoder(x)
        else:
            x = self.final_layer(x, c)
            output = self.unpatchify(x, self.patch_size)

        if self.zero_land_output:
            if mask is None:
                raise ValueError("Mask is required when zero_land_output is enabled")
            output = output * mask

        return output


def DANet_B_16(**kwargs):
    return DANet(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, patch_size=16, **kwargs)

def DANet_B_32(**kwargs):
    return DANet(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, patch_size=32, **kwargs)

def DANet_L_16(**kwargs):
    return DANet(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, patch_size=16, **kwargs)

def DANet_L_32(**kwargs):
    return DANet(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, patch_size=32, **kwargs)

def DANet_H_16(**kwargs):
    return DANet(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, patch_size=16, **kwargs)

def DANet_H_32(**kwargs):
    return DANet(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, patch_size=32, **kwargs)


DANet_models = {
    'DANet-B/16': DANet_B_16,
    'DANet-B/32': DANet_B_32,
    'DANet-L/16': DANet_L_16,
    'DANet-L/32': DANet_L_32,
    'DANet-H/16': DANet_H_16,
    'DANet-H/32': DANet_H_32,
}
