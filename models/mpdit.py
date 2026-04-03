# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Downsampling DiT: A DiT variant with spatial downsampling, processing, and upsampling stages.
Architecture: PatchEmbed (reduces spatial resolution) -> DiT blocks -> Upsample -> DiT blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class FNOTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, modes=16, width=32):
        super().__init__()
        self.modes = modes
        self.width = width

        # Lifting layer - project time to higher dimensional space
        self.fc0 = nn.Linear(1, self.width)

        # Fourier layers
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes)

        # Local convolutions
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)

        # Projection layer
        self.fc1 = nn.Linear(self.width, hidden_size)

    def forward(self, t):
        batch_size = t.shape[0]
        t = t.unsqueeze(-1)  # [batch, 1]

        # Create a temporal grid around each timestep
        grid = torch.linspace(-1, 1, 32, device=t.device).unsqueeze(0).repeat(batch_size, 1)
        # Center grid around timestep value
        grid = grid + t

        # Lift to higher dimension
        x = self.fc0(grid.unsqueeze(-1))  # [batch, grid_size, width]
        x = x.permute(0, 2, 1)  # [batch, width, grid_size]

        # Fourier layers
        x = self.conv0(x) + self.w0(x)
        x = F.gelu(x)

        x = self.conv1(x) + self.w1(x)
        x = F.gelu(x)

        x = self.conv2(x) + self.w2(x)

        # Global average pooling and projection
        x = x.mean(dim=-1)  # [batch, width]
        x = self.fc1(x)
        return x

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.weights_real = nn.Parameter(
            torch.randn(in_channels, out_channels, modes)
        )
        self.weights_imag = nn.Parameter(
            torch.randn(in_channels, out_channels, modes)
        )

    def forward(self, x):
        # Store original dtype and convert to float32 for FFT
        original_dtype = x.dtype
        x_float32 = x.float()

        # Compute FFT in float32
        x_ft = torch.fft.rfft(x_float32, dim=-1)
        x_ft_real = x_ft.real
        x_ft_imag = x_ft.imag

        # Rest of computation in float32
        out_ft_real = torch.zeros(
            x.shape[0], self.out_channels, x.shape[-1]//2 + 1, 
            device=x.device, dtype=torch.float32
        )
        out_ft_imag = torch.zeros(
            x.shape[0], self.out_channels, x.shape[-1]//2 + 1, 
            device=x.device, dtype=torch.float32
        )

        if self.modes <= x_ft_real.shape[-1]:
            # Complex multiplication
            x_real_modes = x_ft_real[:, :, :self.modes]
            x_imag_modes = x_ft_imag[:, :, :self.modes]

            # Ensure weights are in float32
            weights_real = self.weights_real.float()
            weights_imag = self.weights_imag.float()

            # Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
            real_result = torch.einsum("bim,iom->bom", x_real_modes, weights_real) - \
                         torch.einsum("bim,iom->bom", x_imag_modes, weights_imag)
            imag_result = torch.einsum("bim,iom->bom", x_real_modes, weights_imag) + \
                         torch.einsum("bim,iom->bom", x_imag_modes, weights_real)

            out_ft_real[:, :, :self.modes] = real_result
            out_ft_imag[:, :, :self.modes] = imag_result

        # Combine and inverse FFT
        out_ft = torch.complex(out_ft_real, out_ft_imag)
        result = torch.fft.irfft(out_ft, n=x.shape[-1], dim=-1)

        # Convert back to original dtype
        return result.to(original_dtype)
    

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
            # nn.SiLU(),
            # nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        """
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


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    """
    def __init__(self, num_classes, hidden_size, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size * num_tokens)

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        embeddings = embeddings.view(embeddings.shape[0], self.num_tokens, -1)
        return embeddings


#################################################################################
#                                 Core DiT Block                                #
#################################################################################


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, time_modulation):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = time_modulation
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

#################################################################################
#                         Token Upsampling Module.                              #
#################################################################################

class TokenUpsampler(nn.Module):
    """
    Upsamples tokens spatially using pixel shuffle operation.
    Increases spatial resolution by redistributing channel dimension.
    """
    def __init__(self, hidden_size, upsample_factor=4):
        super().__init__()
        self.upsample_factor = upsample_factor
        # Project to expanded dimension for upsampling
        self.proj = nn.Linear(hidden_size, hidden_size * upsample_factor * upsample_factor, bias=True)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.refine = nn.Linear(hidden_size, hidden_size, bias=True)
        

    def forward(self, x, num_condition_tokens=0):
        """
        x: (B, num_condition_tokens + H*W, D)
        returns: (B, num_condition_tokens + H'*W', D) where H' = H * upsample_factor, W' = W * upsample_factor
        """
        B, N, D = x.shape

        # Split condition tokens and spatial tokens
        if num_condition_tokens > 0:
            cond_tokens = x[:, :num_condition_tokens, :]
            spatial_tokens = x[:, num_condition_tokens:, :]
        else:
            cond_tokens = None
            spatial_tokens = x

        # Calculate spatial dimensions
        num_spatial = spatial_tokens.shape[1]
        h = w = int(num_spatial ** 0.5)
        assert h * w == num_spatial, f"Number of spatial tokens ({num_spatial}) must be a perfect square"

        # Project to expanded dimension
        spatial_tokens = self.proj(spatial_tokens)  # (B, H*W, D * upsample_factor^2)

        # Reshape to spatial grid with expanded channels
        spatial_tokens = spatial_tokens.view(B, h, w, D, self.upsample_factor, self.upsample_factor)

        # Pixel shuffle: rearrange channels into spatial dimensions
        spatial_tokens = spatial_tokens.permute(0, 1, 4, 2, 5, 3).contiguous()
        spatial_tokens = spatial_tokens.view(B, h * self.upsample_factor, w * self.upsample_factor, D)
        spatial_tokens = spatial_tokens.view(B, -1, D)

        spatial_tokens = F.gelu(spatial_tokens, approximate="tanh")

        # Recombine with condition tokens
        if cond_tokens is not None:
            x = torch.cat([cond_tokens, spatial_tokens], dim=1)
        else:
            x = spatial_tokens

        # Refinement Linear
        x = self.norm(x)
        x = self.refine(x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def forward(self, x, time_modulation):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = time_modulation
        shift, scale = shift_mlp, scale_mlp
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


#################################################################################
#                                 MPDiT Model.                                  #
#################################################################################


class MPDiT(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth_downsample=24,
        depth_upsample=4,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=1000,
        joint=False,
        num_condition_tokens=16,
        time_emb="fno",
        skip_connection = True
    ):
        """This is MPDiT for 1-level downsample.

        Args:
            input_size (int, optional): Spatial size of input (assumes square). Defaults to 32.
            patch_size (int, optional): Size of each patch for patch embedding. Defaults to 2.
            in_channels (int, optional): Number of input channels. Defaults to 4.
            hidden_size (int, optional): Hidden size of the model. Defaults to 1152.
            depth_downsample (int, optional): Number of downsample blocks. Defaults to 24.
            depth_upsample (int, optional): Number of upsample blocks. Defaults to 4.
            num_heads (int, optional): Number of attention heads. Defaults to 16.
            mlp_ratio (float, optional): Ratio of MLP hidden dimension to embedding dimension. Defaults to 4.0.
            num_classes (int, optional): Number of classes for classification. Defaults to 1000.
            joint (bool, optional): Whether to use joint modeling. Defaults to False.
            num_condition_tokens (int, optional): Number of condition tokens. Defaults to 16.
            time_emb (str, optional): Type of time embedding. Defaults to "fno".
            skip_connection (bool, optional): Whether to use skip connections. Defaults to True.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if not joint else 2 * in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_condition_tokens = num_condition_tokens
        self.num_classes = num_classes
        self.time_emb = time_emb
        self.skip_connection = skip_connection

        # Standard patch embedding
        self.x_embedder = PatchEmbed(
            input_size,
            patch_size,
            in_channels,
            hidden_size,
            bias=True
        )

        if skip_connection:
            self.x_embedder_skip = PatchEmbed(
                input_size,
                2,
                in_channels,
                hidden_size,
                bias=True
            )
            self.linear_skip = nn.Linear(hidden_size, hidden_size, bias=True)

        if time_emb == "mlp":
            self.t_embedder = TimestepEmbedder(hidden_size)
        elif time_emb == "fno":
            self.t_embedder = FNOTimestepEmbedder(hidden_size, modes=16, width=32)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, num_condition_tokens)
        num_patches = self.x_embedder.num_patches

        # Positional embeddings at full resolution (after patch embedding)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # Positional embeddings for downsampled resolution
        num_patches = self.x_embedder.num_patches
        self.pos_embed_downsampled = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, (input_size//2)**2, hidden_size), requires_grad=False)

        # Time modulation for all blocks
        self.time_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # Downsample blocks (majority of computation at reduced resolution)
        self.downsample_blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth_downsample)
        ])

        # Upsampler module
        self.upsampler = TokenUpsampler(hidden_size, upsample_factor=patch_size//2)

        # Upsample blocks (final processing at full resolution)
        self.upsample_blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth_upsample)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size=2, out_channels=self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize pos_embed by sin-cos embedding (full resolution)
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.pos_embed.shape[1] ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize downsampled pos_embed
        pos_embed_downsampled = get_2d_sincos_pos_embed(
            self.pos_embed_downsampled.shape[-1],
            int(self.pos_embed_downsampled.shape[1] ** 0.5)
        )
        self.pos_embed_downsampled.data.copy_(torch.from_numpy(pos_embed_downsampled).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        if self.skip_connection:
            w = self.x_embedder_skip.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.x_embedder_skip.proj.bias, 0)
            nn.init.xavier_uniform_(self.linear_skip.weight)
            nn.init.constant_(self.linear_skip.bias, 0)

        # Initialize label embedding table
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP
        if self.time_emb == "mlp":
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out time modulation layer
        nn.init.constant_(self.time_modulation[-1].weight, 0)
        nn.init.constant_(self.time_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # Initialize downsampler and upsampler projections
        nn.init.xavier_uniform_(self.upsampler.proj.weight)
        nn.init.constant_(self.upsampler.proj.bias, 0)
        nn.init.xavier_uniform_(self.upsampler.refine.weight)
        nn.init.constant_(self.upsampler.refine.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = 2
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y=None):
        """
        Forward pass of MPDiT.
        x: (N, C, H, W) tensor of spatial inputs
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # Embed patches (standard patch embedding)
        x_inp = x
        x = self.x_embedder(x)  # (N, T, D) where T = (H*W) / patch_size^2

        # Add positional embeddings at full resolution
        x = x + self.pos_embed_downsampled
        t_embed = self.t_embedder(t)  # (N, D)

        # Class condition as tokens
        c = self.y_embedder(y)  # (N, down_num_tokens, D)

        # Concatenate condition tokens
        x = torch.cat([c, x], dim=1)  # (N, down_num_tokens + T, D)

        # Compute time modulation once for all blocks
        time_modulation = self.time_modulation(t_embed).chunk(6, dim=1)

        # Process at downsampled resolution (majority of computation)
        for block in self.downsample_blocks:
            x = block(x, time_modulation)

        # Upsample tokens (using pixel shuffle)
        x = self.upsampler(x, num_condition_tokens=self.num_condition_tokens)

        # Add positional embeddings at full resolution (only to spatial tokens)
        spatial_tokens = x[:, self.num_condition_tokens:, :]
        if self.skip_connection:
            x_skip = self.x_embedder_skip(x_inp)
            x_skip = self.linear_skip(x_skip)
            spatial_tokens = spatial_tokens + x_skip
            
        spatial_tokens = spatial_tokens + self.pos_embed
        x = torch.cat([x[:, :self.num_condition_tokens, :], spatial_tokens], dim=1)

        # Process at full resolution
        for block in self.upsample_blocks:
            x = block(x, time_modulation)

        # Extract only the patch tokens for final layer
        x_patches = x[:, self.num_condition_tokens:, :]
        cls_token = x[:, :self.num_condition_tokens, :]
        x = self.final_layer(x_patches, time_modulation)

        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass with classifier-free guidance.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y=y)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    


class MPDiTv2(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_sizes=[8, 4, 2],
        in_channels=4,
        hidden_size=1152,
        depth_list=[16, 6, 6],
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=1000,
        joint=False,
        num_condition_tokens=16,
        time_emb="fno",
        skip_connection=True
    ):
        """MPDiT with 2-level downsample

        Args:
            input_size (int, optional): Spatial size of input (assumes square). Defaults to 32.
            patch_sizes (list, optional): List of patch sizes for each level. Defaults to [8, 4, 2].
            in_channels (int, optional): Number of input channels. Defaults to 4.
            hidden_size (int, optional): Hidden size of the model. Defaults to 1152.
            depth_list (list, optional): List of depths for each level. Defaults to [16, 6, 6].
            num_heads (int, optional): Number of attention heads. Defaults to 16.
            mlp_ratio (float, optional): Ratio of MLP hidden dimension to embedding dimension. Defaults to 4.0.
            num_classes (int, optional): Number of classes for classification. Defaults to 1000.
            joint (bool, optional): Whether to use joint modeling. Defaults to False.
            num_condition_tokens (int, optional): Number of condition tokens. Defaults to 16.
            time_emb (str, optional): Type of time embedding. Defaults to "fno".
            skip_connection (bool, optional): Whether to use skip connections. Defaults to True.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if not joint else 2 * in_channels
        self.patch_sizes = patch_sizes
        self.num_heads = num_heads
        self.num_condition_tokens = num_condition_tokens
        self.num_classes = num_classes
        self.time_emb = time_emb
        self.skip_connection = skip_connection

        # Patch embedding
        self.x_embedder = PatchEmbed(input_size, patch_sizes[0], in_channels, hidden_size, bias=True)
        if skip_connection:
            self.x_embedder_lvl1 = PatchEmbed(input_size, patch_sizes[1], in_channels, hidden_size, bias=True)
            self.x_embedder_lvl2 = PatchEmbed(input_size, patch_sizes[2], in_channels, hidden_size, bias=True)
            self.linear_skip_lvl1 = nn.Linear(hidden_size, hidden_size, bias=True)
            self.linear_skip_lvl2 = nn.Linear(hidden_size, hidden_size, bias=True)

        if time_emb == "mlp":
            self.t_embedder = TimestepEmbedder(hidden_size)
        elif time_emb == "fno":
            self.t_embedder = FNOTimestepEmbedder(hidden_size, modes=16, width=32)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, num_condition_tokens)

        # Positional embeddings for downsampled resolution
        self.pos_embed = nn.Parameter(torch.zeros(1, (input_size//8)**2, hidden_size), requires_grad=False)
        self.pos_embed_lvl1 = nn.Parameter(torch.zeros(1, (input_size//4)**2, hidden_size), requires_grad=False)
        self.pos_embed_lvl2 = nn.Parameter(torch.zeros(1, (input_size//2)**2, hidden_size), requires_grad=False)

        # Time modulation for all blocks
        self.time_modulation = nn.Sequential(nn.SiLU(), 
                                             nn.Linear(hidden_size, 6 * hidden_size, bias=True),
                                            )

        # Downsample blocks (majority of computation at reduced resolution)
        self.downsample_blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth_list[0])])

        # Upsampler module lvl1
        self.upsampler_lvl1 = TokenUpsampler(hidden_size, upsample_factor=2)
        self.upsample_blocks_lvl1 = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth_list[1])])

        # Upsampler module lvl2
        self.upsampler_lvl2 = TokenUpsampler(hidden_size, upsample_factor=2)
        self.upsample_blocks_lvl2 = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth_list[2])])

        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size=2, out_channels=self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize pos_embed by sin-cos embedding (full resolution)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.pos_embed.shape[1] ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        pos_embed_lvl1 = get_2d_sincos_pos_embed(self.pos_embed_lvl1.shape[-1], int(self.pos_embed_lvl1.shape[1] ** 0.5))
        self.pos_embed_lvl1.data.copy_(torch.from_numpy(pos_embed_lvl1).float().unsqueeze(0))

        pos_embed_lvl2 = get_2d_sincos_pos_embed(self.pos_embed_lvl2.shape[-1], int(self.pos_embed_lvl2.shape[1] ** 0.5))
        self.pos_embed_lvl2.data.copy_(torch.from_numpy(pos_embed_lvl2).float().unsqueeze(0))


        # Initialize patch_embed like nn.Linear
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        if self.skip_connection:
            w = self.x_embedder_lvl1.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.x_embedder_lvl1.proj.bias, 0)

            w = self.x_embedder_lvl2.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.x_embedder_lvl2.proj.bias, 0)
            
            nn.init.xavier_uniform_(self.linear_skip_lvl1.weight)
            nn.init.constant_(self.linear_skip_lvl1.bias, 0)
            nn.init.xavier_uniform_(self.linear_skip_lvl2.weight)
            nn.init.constant_(self.linear_skip_lvl2.bias, 0)

        # Initialize label embedding table
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP
        if self.time_emb == "mlp":
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out time modulation layer
        nn.init.constant_(self.time_modulation[-1].weight, 0)
        nn.init.constant_(self.time_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # Initialize downsampler and upsampler projections
        nn.init.xavier_uniform_(self.upsampler_lvl1.proj.weight)
        nn.init.constant_(self.upsampler_lvl1.proj.bias, 0)

        nn.init.xavier_uniform_(self.upsampler_lvl2.refine.weight)
        nn.init.constant_(self.upsampler_lvl2.refine.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = 2
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y=None):
        """
        Forward pass of DownsampleDiT.
        x: (N, C, H, W) tensor of spatial inputs
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # Embed patches (standard patch embedding)
        x_inp = x
        x = self.x_embedder(x)  # (N, T, D) where T = (H*W) / patch_size^2
        x = x + self.pos_embed
        c = self.y_embedder(y)  # (N, down_num_tokens, D)
        x = torch.cat([c, x], dim=1)  # (N, down_num_tokens + T, D)

        t_embed = self.t_embedder(t)  # (N, D)
        time_modulation = self.time_modulation(t_embed).chunk(6, dim=1)

        # Process at downsampled resolution (majority of computation)
        for block in self.downsample_blocks:
            x = block(x, time_modulation)

        ##############################################################################
        # First upsampling stage
        ##############################################################################
        x = self.upsampler_lvl1(x, num_condition_tokens=self.num_condition_tokens)

        # Add positional embeddings at full resolution (only to spatial tokens)
        spatial_tokens = x[:, self.num_condition_tokens:, :]
        if self.skip_connection:
            x_skip = self.x_embedder_lvl1(x_inp)
            x_skip = self.linear_skip_lvl1(x_skip)
            spatial_tokens = spatial_tokens + x_skip
        spatial_tokens = spatial_tokens + self.pos_embed_lvl1
        x = torch.cat([x[:, :self.num_condition_tokens, :], spatial_tokens], dim=1)
        for block in self.upsample_blocks_lvl1:
            x = block(x, time_modulation)

        ##############################################################################
        # Second upsampling stage
        ##############################################################################
        x = self.upsampler_lvl2(x, num_condition_tokens=self.num_condition_tokens)

        # Add positional embeddings at full resolution (only to spatial tokens)
        spatial_tokens = x[:, self.num_condition_tokens:, :]
        if self.skip_connection:
            x_skip = self.x_embedder_lvl2(x_inp)
            x_skip = self.linear_skip_lvl2(x_skip)
            spatial_tokens = spatial_tokens + x_skip
        spatial_tokens = spatial_tokens + self.pos_embed_lvl2
        x = torch.cat([x[:, :self.num_condition_tokens, :], spatial_tokens], dim=1)
        for block in self.upsample_blocks_lvl2:
            x = block(x, time_modulation)

        # Extract only the patch tokens for final layer
        x_patches = x[:, self.num_condition_tokens:, :]
        x = self.final_layer(x_patches, time_modulation)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass with classifier-free guidance.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y=y)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return: pos_embed: [grid_size*grid_size, embed_dim]
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


#################################################################################
#                              Model Configs                                    #
#################################################################################


def MPDiT_XL_4x(**kwargs):
    """XL model with 4x spatial downsampling (16x token reduction)"""
    return MPDiT(
        depth_downsample=22,
        depth_upsample=6,
        hidden_size=1152,
        patch_size=4,
        num_heads=16,
        **kwargs
    )

def MPDiT_L_4x(**kwargs):
    """L model with 4x spatial downsampling (16x token reduction)"""
    return MPDiT(
        depth_downsample=18,
        depth_upsample=6,
        hidden_size=1024,
        patch_size=4,
        num_heads=16,
        **kwargs
    )


def MPDiT_B_4x(**kwargs):
    """B model with 4x spatial downsampling (16x token reduction)"""
    return MPDiT(
        depth_downsample=6,
        depth_upsample=6,
        hidden_size=768,
        patch_size=4,
        num_heads=12,
        **kwargs
    )

def MPDiTv2_XL(**kwargs):
    return MPDiTv2(
        depth_list=[14, 8, 6],
        hidden_size=1152,
        patch_sizes=[8,4,2],
        num_heads=16,
        **kwargs
    )

def MPDiTv2_L(**kwargs):
    return MPDiTv2(
        depth_list=[14, 6, 4],
        hidden_size=1024,
        patch_sizes=[8,4,2],
        num_heads=16,
        **kwargs
    )

def MPDiTv2_B(**kwargs):
    return MPDiTv2(
        depth_list=[6, 4, 2],
        hidden_size=768,
        patch_sizes=[8,4,2],
        num_heads=12,
        **kwargs
    )



MPDiT_models = {
    'MPDiT-XL/4x': MPDiT_XL_4x,
    'MPDiT-L/4x': MPDiT_L_4x,
    'MPDiT-B/4x': MPDiT_B_4x,
    'MPDiTv2-XL': MPDiTv2_XL,
    'MPDiTv2-L': MPDiTv2_L,
    'MPDiTv2-B': MPDiTv2_B,
}
