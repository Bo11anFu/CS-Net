import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import warnings

try:
    from SFS_MSDeformAttn.ops.modules import MSDeformAttn_for_sfs as _MSDeformAttnCUDA
    _HAS_CUDA_SFS = True
except ImportError:
    _HAS_CUDA_SFS = False
    warnings.warn("SFS_MSDeformAttn CUDA module not found, using CPU fallback.")


def generate_structured_grid(n_heads, n_points, n_levels=1, base_radius=1.0, radius_step=1.0):
    """
    Initialization of spiral-aware sampling pattern.

    parameters:
    - n_heads: number of attention heads
    - n_points: number of sampling points of each head
    - n_levels: number of feature levels, default=1
    - base_radius: initial radius of sampling point
    - radius_step: radial step between consecutive points of each head

    return:
    - grid: Tensor, [n_heads, n_levels, n_points, 2]
    """
    offsets = []
    for h in range(n_heads):
        head_offsets = []
        delta_theta = 2 * math.pi * h / n_heads
        for i in range(n_points):
            theta = 2 * math.pi * i / n_points + delta_theta
            r = base_radius + i * radius_step
            dx = r * math.cos(theta)
            dy = r * math.sin(theta)
            head_offsets.append([dx, dy])
        offsets.append(head_offsets)

    grid = torch.tensor(offsets, dtype=torch.float32)
    grid = grid.unsqueeze(1).repeat(1, n_levels, 1, 1)
    return grid


class MSDeformAttnCPU(nn.Module):
    """CPU-compatible fallback for MSDeformAttn_for_sfs using grid_sample."""

    def __init__(self, d_model=256, n_levels=1, n_heads=8, n_points=4, ratio=1.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads')
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.ratio = ratio
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, int(d_model * ratio))
        self.output_proj = nn.Linear(int(d_model * ratio), d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes,
                input_level_start_index, sampling_offsets, input_padding_mask=None):
        N, Len_q, C = query.shape
        _, Len_in, _ = input_flatten.shape
        H2, W2 = input_spatial_shapes[0]
        V_dim = int(self.ratio * self.d_model)
        d_head = V_dim // self.n_heads

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, d_head)

        value_2d = value.reshape(N, H2, W2, self.n_heads, d_head)
        value_2d = value_2d.permute(0, 3, 4, 1, 2).reshape(N, self.n_heads, d_head, H2, W2)

        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_points)
        attention_weights = F.softmax(attention_weights, dim=-1)

        sampling_offsets = sampling_offsets.view(N, Len_q, self.n_heads, self.n_points, 2)

        ref_xy = reference_points[:, :, 0, :]
        sampling_locations = ref_xy.unsqueeze(2).unsqueeze(3) + sampling_offsets
        sampling_locations = sampling_locations * 2.0 - 1.0

        sampled_list = []
        for h in range(self.n_heads):
            head_value = value_2d[:, h]
            head_value_exp = head_value.unsqueeze(1).expand(N, Len_q, d_head, H2, W2)
            head_value_exp = head_value_exp.reshape(N * Len_q, d_head, H2, W2)

            head_locs = sampling_locations[:, :, h, :, :]
            head_locs = head_locs.reshape(N * Len_q, self.n_points, 1, 2)

            head_sampled = F.grid_sample(head_value_exp, head_locs, mode='bilinear',
                                          padding_mode='zeros', align_corners=True)
            head_sampled = head_sampled.squeeze(-1)
            head_sampled = head_sampled.view(N, Len_q, d_head, self.n_points)
            head_sampled = head_sampled.permute(0, 1, 3, 2)
            sampled_list.append(head_sampled)

        sampled_feat = torch.stack(sampled_list, dim=2)

        attn_weight = attention_weights.unsqueeze(-1)
        output = (sampled_feat * attn_weight).sum(dim=3)
        output = output.reshape(N, Len_q, V_dim)

        output = self.output_proj(output)
        return output


if _HAS_CUDA_SFS:
    MSDeformAttn_for_sfs = _MSDeformAttnCUDA
else:
    MSDeformAttn_for_sfs = MSDeformAttnCPU


class SpiralAware_CrossDeformAttn2D(nn.Module):
    """
    Spiral-Aware MSDeformAttn.

    Inputs:
        - query_feat: [B, C, H1, W1], larger scale feature maps
        - key_feat:   [B, C, H2, W2], smaller scale feature maps
    Output:
        - out:   [B, C, H1, W1]
    """
    def __init__(self, dim, n_heads=8, n_points=4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_points = n_points

        self.query_Conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.key_Conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        self.shared_offsets_residual = nn.Parameter(torch.zeros(n_heads, n_points, 2))
        fixed_bias = generate_structured_grid(n_heads, n_points, n_levels=1, base_radius=1.0, radius_step=1.0)
        self.register_buffer("offset_base", fixed_bias.view(1, 1, n_heads, 1, n_points, 2))

        self.query_norm = nn.LayerNorm(dim)
        self.key_norm = nn.LayerNorm(dim)
        self.out_norm = nn.LayerNorm(dim)

        self.attn = MSDeformAttn_for_sfs(
            d_model=dim,
            n_levels=1,
            n_heads=n_heads,
            n_points=n_points
        )

    def forward(self, query_feat: Tensor, key_feat: Tensor) -> Tensor:
        B, C, H1, W1 = query_feat.shape
        _, _, H2, W2 = key_feat.shape

        query_feat = self.query_Conv(query_feat)
        key_feat = self.key_Conv(key_feat)

        offsets_residual = self.shared_offsets_residual

        shared_offsets = self.offset_base.view(self.n_heads, 1, self.n_points, 2) + offsets_residual.view(self.n_heads, 1, self.n_points, 2)
        offsets = shared_offsets.view(1, 1, self.n_heads, 1, self.n_points, 2).expand(B, H1 * W1, -1, -1, -1, -1)

        query = query_feat.flatten(2).transpose(1, 2)
        kv = key_feat.flatten(2).transpose(1, 2)
        query = self.query_norm(query)
        kv = self.key_norm(kv)

        spatial_shapes = torch.tensor([[H2, W2]], device=key_feat.device, dtype=torch.long)
        level_start_index = torch.tensor([0], device=key_feat.device, dtype=torch.long)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.5 / H1, 1 - 0.5 / H1, H1, device=query_feat.device),
            torch.linspace(0.5 / W1, 1 - 0.5 / W1, W1, device=query_feat.device),
            indexing='ij'
        )
        reference_points = torch.stack((grid_x, grid_y), -1)
        reference_points = reference_points.view(1, H1 * W1, 1, 2).repeat(B, 1, 1, 1)

        attn = self.attn(
            query=query,
            reference_points=reference_points,
            input_flatten=kv,
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start_index,
            sampling_offsets=offsets
        )

        out = query + query * attn
        out = self.out_norm(out).transpose(1, 2).reshape(B, C, H1, W1)
        return out