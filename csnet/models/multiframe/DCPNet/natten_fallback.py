"""Pure-PyTorch re-implementation of the two NATTEN 2D primitives used by DCPNet.

DCPNet (``model_DCPNet.py``) imports

    from natten.functional import NATTEN2DQKRPBFunction, NATTEN2DAVFunction

NATTEN is a CUDA extension and no prebuilt wheel matches this environment
(torch 2.5.1+cu118), so this module provides an equivalent implementation built
from standard PyTorch ops. The call signature and the numerics are identical.
When a real NATTEN install is present the model uses it and this module is never
imported.

Semantics were taken from the NATTEN 0.14.6 sources vendored in the project
(``MSmsd/models/multiframe/DCPNet-main/NATTEN-0.14.6``):

* ``src/natten/csrc/cuda/natten2dqkrpb_cuda_kernel.cu``
* ``src/natten/csrc/cuda/natten2dav_cuda_kernel.cu``
* ``src/natten/csrc/cuda/natten_commons.cuh`` (``get_window_start``, ``get_pb_start``)

For a query at ``(i, j)`` and neighbourhood offset ``(ki, kj)``, with
``n = ki * K + kj``::

    r = get_window_start(i, H, K, (K-1)//2, d) + ki * d
    c = get_window_start(j, W, K, (K-1)//2, d) + kj * d

    attn[b, h, i, j, n] = sum_dim q[b, h, i, j, :] * k[b, h, r, c, :]
                          + rpb[h, get_pb_start(i) + ki, get_pb_start(j) + kj]
    out [b, h, i, j, :] = sum_{ki, kj} attn[b, h, i, j, n] * v[b, h, r, c, :]

``get_window_start`` shifts the window inward at the borders, so NATTEN never
reads out of bounds and never masks; the port reproduces that exactly.

Both primitives are ``torch.autograd.Function`` with an explicit backward, and
both are organised so that only the "gather table" and the "bias table" describe
the neighbourhood. Those two tables depend solely on ``(H, W, K, d)``, so they
are built once per geometry and cached: a training step calls these operators
about twenty times with identical geometry, and rebuilding the tables per chunk
was costing more host time than the kernels themselves (the profiler showed tens
of thousands of ``slice`` / ``as_strided`` entries per step).

The spatial extent is processed in row chunks sized to ``_CHUNK_BYTES`` (override
with the ``DCPNET_NATTEN_CHUNK_MB`` environment variable) so transient
allocations stay bounded on small GPUs. Chunking is done on the *row* axis so
that ``table[start:stop]`` stays a contiguous view and needs no copy, which is
why the output tensors are concatenated on the axis with size ``H``.
"""

import os

import torch

__all__ = [
    "NATTEN2DQKRPBFunction",
    "NATTEN2DAVFunction",
    "natten2dqkrpb",
    "natten2dav",
]

_CHUNK_BYTES = int(float(os.environ.get("DCPNET_NATTEN_CHUNK_MB", "64")) * 1024 * 1024)

_GEOMETRY_CACHE = {}


# --------------------------------------------------------------------------- #
# index helpers
# --------------------------------------------------------------------------- #
def _window_start(length, kernel_size, dilation, device):
    """Vectorised port of ``get_window_start`` from ``natten_commons.cuh``."""
    n = torch.arange(length, device=device)
    neighbourhood = (kernel_size - 1) // 2
    if dilation <= 1:
        start = torch.clamp(n - neighbourhood, min=0)
        return start + (n + neighbourhood >= length).long() * (length - n - neighbourhood - 1)
    nbd = neighbourhood * dilation
    mod = n % dilation
    aligned = (length // dilation) * dilation
    tail = length - aligned
    far_end = torch.where(
        mod < tail,
        length - tail + mod - 2 * nbd,
        aligned + mod - kernel_size * dilation,
    )
    return torch.where(n - nbd < 0, mod, torch.where(n + nbd >= length, far_end, n - nbd))


def _position_bias_start(length, kernel_size, dilation, device):
    """Vectorised port of ``get_pb_start`` from ``natten_commons.cuh``."""
    n = torch.arange(length, device=device)
    neighbourhood = (kernel_size - 1) // 2
    if dilation <= 1:
        return (neighbourhood
                + (n < neighbourhood).long() * (neighbourhood - n)
                + (n + neighbourhood >= length).long() * (length - n - 1 - neighbourhood))
    nbd = neighbourhood * dilation
    near_end = kernel_size - 1 - (n // dilation)
    far_end = (length - n - 1) // dilation
    interior = torch.full_like(n, neighbourhood)
    return torch.where(n - nbd < 0, near_end, torch.where(n + nbd >= length, far_end, interior))


def _geometry(height, width, kernel_size, dilation, device):
    """Gather and bias tables for one geometry, cached across calls.

    ``gather[i, j, n]``   flat index of the key/value pixel of query ``(i, j)`` at offset ``n``
    ``bias[i, j, n]``     flat index of the same offset inside the ``(2K-1) x (2K-1)`` rpb grid

    Both are laid out as ``[H, W, K*K]`` with ``n = ki * K + kj``, so slicing the
    first axis yields a contiguous chunk.
    """
    key = (height, width, kernel_size, dilation, str(device))
    entry = _GEOMETRY_CACHE.get(key)
    if entry is not None:
        return entry

    steps = torch.arange(kernel_size, device=device)
    row_start = _window_start(height, kernel_size, dilation, device)
    col_start = _window_start(width, kernel_size, dilation, device)
    # [H, K] and [W, K] source coordinates per offset
    rows = (row_start[:, None] + steps[None, :] * dilation)[:, None, :, None]
    cols = (col_start[:, None] + steps[None, :] * dilation)[None, :, None, :]

    gather = ((rows * width + cols)
              .expand(height, width, kernel_size, kernel_size)
              .reshape(height, width, kernel_size * kernel_size)
              .contiguous())

    rpb_width = 2 * kernel_size - 1
    pb_rows = (_position_bias_start(height, kernel_size, dilation, device)[:, None]
               + steps[None, :])[:, None, :, None]
    pb_cols = (_position_bias_start(width, kernel_size, dilation, device)[:, None]
               + steps[None, :])[None, :, None, :]
    bias = ((pb_rows * rpb_width + pb_cols)
            .expand(height, width, kernel_size, kernel_size)
            .reshape(height, width, kernel_size * kernel_size)
            .contiguous())

    entry = {"gather": gather, "bias": bias}
    _GEOMETRY_CACHE[key] = entry
    return entry


def _chunks(length, bytes_per_row):
    """Row ranges limiting each transient neighbourhood tensor to ``_CHUNK_BYTES``."""
    chunk = max(1, int(_CHUNK_BYTES // max(1, bytes_per_row)))
    if chunk >= length:
        return [(0, length)]
    bounds = []
    position = 0
    while position < length:
        step = min(chunk, length - position)
        bounds.append((position, step))
        position += step
    return bounds


# --------------------------------------------------------------------------- #
# NATTEN2DQKRPB
# --------------------------------------------------------------------------- #
def _qkrpb_forward(query, key, rpb, kernel_size, dilation):
    batch, heads, height, width, dim = query.shape
    assert rpb is not None and rpb.shape[0] == 1, "the port requires a shared RPB"
    gather = _geometry(height, width, kernel_size, dilation, query.device)["gather"]
    bias = _geometry(height, width, kernel_size, dilation, query.device)["bias"]
    rpb_flat = rpb.reshape(-1)
    key_flat = key.reshape(batch, heads, height * width, dim)

    neighbourhood = kernel_size * kernel_size
    bytes_per_row = batch * heads * width * neighbourhood * dim * 4

    outputs = []
    for start, chunk in _chunks(height, bytes_per_row):
        index = gather[start:start + chunk].reshape(-1)
        neighbours = key_flat.index_select(2, index).reshape(
            batch, heads, chunk, width, neighbourhood, dim)
        bias_chunk = rpb_flat.index_select(
            0, bias[start:start + chunk].reshape(-1)).reshape(chunk, width, neighbourhood)
        attention = torch.matmul(
            neighbours, query[:, :, start:start + chunk].unsqueeze(-1)).squeeze(-1)
        outputs.append(attention + bias_chunk)

    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=2)


def _qkrpb_backward(grad_attn, query, key, rpb, kernel_size, dilation):
    batch, heads, height, width, dim = query.shape
    gather = _geometry(height, width, kernel_size, dilation, query.device)["gather"]
    bias = _geometry(height, width, kernel_size, dilation, query.device)["bias"]

    grad_attn = grad_attn.contiguous()
    d_query = torch.empty_like(query)
    key_flat = key.reshape(batch, heads, height * width, dim)
    d_key_flat = torch.zeros_like(key_flat)
    # rpb has a leading singleton head axis ([1, 2K-1, 2K-1]); the kernel shares it
    # across heads, so the whole flat buffer is the (2K-1)**2 bias grid.
    d_rpb_flat = torch.zeros_like(rpb.reshape(-1))

    neighbourhood = kernel_size * kernel_size
    bytes_per_row = batch * heads * width * neighbourhood * dim * 4

    for start, chunk in _chunks(height, bytes_per_row):
        index = gather[start:start + chunk].reshape(-1)
        neighbours = key_flat.index_select(2, index).reshape(
            batch, heads, chunk, width, neighbourhood, dim)
        grad_chunk = grad_attn[:, :, start:start + chunk]
        query_chunk = query[:, :, start:start + chunk]

        d_query[:, :, start:start + chunk] = torch.matmul(
            neighbours.transpose(-1, -2), grad_chunk.unsqueeze(-1)).squeeze(-1)

        # scatter d_attn * query back onto the key pixels
        contribution = query_chunk.unsqueeze(4) * grad_chunk.unsqueeze(-1)
        d_key_flat.index_add_(2, index, contribution.reshape(batch, heads, -1, dim))

        d_rpb_flat.index_add_(0, bias[start:start + chunk].reshape(-1),
                              grad_chunk.sum(dim=(0, 1)).reshape(-1))

    return d_query, d_key_flat.view_as(key), d_rpb_flat.view_as(rpb), None, None


class NATTEN2DQKRPBFunction(torch.autograd.Function):
    """Drop-in stand-in for ``natten.functional.NATTEN2DQKRPBFunction``."""

    @staticmethod
    def forward(ctx, query, key, rpb, kernel_size, dilation):
        query = query.contiguous()
        key = key.contiguous()
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.save_for_backward(query, key, rpb)
        with torch.no_grad():
            return _qkrpb_forward(query, key, rpb, kernel_size, dilation)

    @staticmethod
    def backward(ctx, grad_attn):
        query, key, rpb = ctx.saved_tensors
        d_query, d_key, d_rpb, _, _ = _qkrpb_backward(
            grad_attn, query, key, rpb, ctx.kernel_size, ctx.dilation)
        return d_query, d_key, d_rpb, None, None


# --------------------------------------------------------------------------- #
# NATTEN2DAV
# --------------------------------------------------------------------------- #
def _av_forward(attention, value, kernel_size, dilation):
    batch, heads, height, width, dim = value.shape
    gather = _geometry(height, width, kernel_size, dilation, value.device)["gather"]
    value_flat = value.reshape(batch, heads, height * width, dim)

    neighbourhood = kernel_size * kernel_size
    bytes_per_row = batch * heads * width * neighbourhood * dim * 4

    outputs = []
    for start, chunk in _chunks(height, bytes_per_row):
        index = gather[start:start + chunk].reshape(-1)
        neighbours = value_flat.index_select(2, index).reshape(
            batch, heads, chunk, width, neighbourhood, dim)
        weight = attention[:, :, start:start + chunk].unsqueeze(-2)
        outputs.append(torch.matmul(weight, neighbours).squeeze(-2))

    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=2)


def _av_backward(grad_out, attention, value, kernel_size, dilation):
    batch, heads, height, width, dim = value.shape
    gather = _geometry(height, width, kernel_size, dilation, value.device)["gather"]

    grad_out = grad_out.contiguous()
    d_attention = torch.empty_like(attention)
    value_flat = value.reshape(batch, heads, height * width, dim)
    d_value_flat = torch.zeros_like(value_flat)

    neighbourhood = kernel_size * kernel_size
    bytes_per_row = batch * heads * width * neighbourhood * dim * 4

    for start, chunk in _chunks(height, bytes_per_row):
        index = gather[start:start + chunk].reshape(-1)
        neighbours = value_flat.index_select(2, index).reshape(
            batch, heads, chunk, width, neighbourhood, dim)

        grad_chunk = grad_out[:, :, start:start + chunk]
        d_attention[:, :, start:start + chunk] = torch.matmul(
            neighbours, grad_chunk.unsqueeze(-1)).squeeze(-1)

        contribution = (attention[:, :, start:start + chunk].unsqueeze(-1)
                        * grad_chunk.unsqueeze(-2))
        d_value_flat.index_add_(2, index, contribution.reshape(batch, heads, -1, dim))

    return d_attention, d_value_flat.view_as(value)


class NATTEN2DAVFunction(torch.autograd.Function):
    """Drop-in stand-in for ``natten.functional.NATTEN2DAVFunction``."""

    @staticmethod
    def forward(ctx, attention, value, kernel_size, dilation):
        attention = attention.contiguous()
        value = value.contiguous()
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.save_for_backward(attention, value)
        with torch.no_grad():
            return _av_forward(attention, value, kernel_size, dilation)

    @staticmethod
    def backward(ctx, grad_out):
        attention, value = ctx.saved_tensors
        d_attention, d_value = _av_backward(
            grad_out, attention, value, ctx.kernel_size, ctx.dilation)
        return d_attention, d_value, None, None


def natten2dqkrpb(query, key, rpb, kernel_size, dilation):
    return NATTEN2DQKRPBFunction.apply(query, key, rpb, kernel_size, dilation)


def natten2dav(attention, value, kernel_size, dilation):
    return NATTEN2DAVFunction.apply(attention, value, kernel_size, dilation)
