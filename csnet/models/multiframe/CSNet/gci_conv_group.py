# gci_conv_group_agg.py
import torch
from torch.autograd import Function
import triton
import triton.language as tl
# from torch.amp import custom_fwd, custom_bwd
import math
import torch.nn as nn

# --------------------------
# grid helper
# --------------------------
def _grid(numel: int, bs: int):
    return (triton.cdiv(numel, bs),)


@triton.jit
def _idx(i, n: int, c: int, h: int, w: int):
    ni = i // (c * h * w)
    ci = (i // (h * w)) % c
    hi = (i // w) % w
    wi = i % w
    m = i < (n * c * h * w)
    return ni, ci, hi, wi, m


# ============================================================
# Forward Triton Kernel
# ============================================================
@triton.jit
def gci_conv_group_fwd(
    x_ptr, w_ptr, o_ptr,
    n, ic, oc, h, w, ks, pad, groups,
    BS: tl.constexpr, CT: tl.constexpr, AT: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BS
    offs = start + tl.arange(0, BS)

    ni, co, hi, wi, m = _idx(offs, n, oc, h, w)
    val = tl.zeros((BS,), dtype=AT)

    group_in = ic // groups
    group_out = oc // groups
    gi = co // group_out
    ci_base = gi * group_in

    for kh in range(ks):
        hin = hi - pad + kh
        hb = (hin >= 0) & (hin < h)
        for kw in range(ks):
            win = wi - pad + kw
            b = hb & (win >= 0) & (win < w)

            w_k_idx = kh * ks + kw
            w_off = ((ni * groups + gi) * ks * ks + w_k_idx) * h * w + hi * w + wi
            w_val = tl.load(w_ptr + w_off, mask=m, other=0.0).to(CT)

            for s in range(group_in):
                ci = ci_base + s
                x_off = ((ni * ic + ci) * h + hin) * w + win
                x_val = tl.load(x_ptr + x_off, mask=m & b, other=0.0).to(CT)
                val += tl.where(m & b, x_val * w_val, 0.0).to(AT)

    o_off = ((ni * oc + co) * h + hi) * w + wi
    tl.store(o_ptr + o_off, val.to(CT), mask=m)


# ============================================================
# Backward: w.r.t x
# ============================================================
@triton.jit
def gci_conv_group_bwd_x(
    go_ptr, w_ptr, gx_ptr,
    n, ic, oc, h, w, ks, pad, groups,
    BS: tl.constexpr, CT: tl.constexpr, AT: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BS
    offs = start + tl.arange(0, BS)

    ni, ci, hi, wi, m = _idx(offs, n, ic, h, w)
    val = tl.zeros((BS,), dtype=AT)

    group_in = ic // groups
    group_out = oc // groups
    gi = ci // group_in
    co_base = gi * group_out

    for kh in range(ks):
        ho = hi + pad - kh
        hb = (ho >= 0) & (ho < h)
        for kw in range(ks):
            wo = wi + pad - kw
            b = hb & (wo >= 0) & (wo < w)

            w_k_idx = kh * ks + kw
            w_off = ((ni * groups + gi) * ks * ks + w_k_idx) * h * w + ho * w + wo
            w_val = tl.load(w_ptr + w_off, mask=m, other=0.0).to(CT)

            for s in range(group_out):
                co = co_base + s
                go_off = ((ni * oc + co) * h + ho) * w + wo
                go_val = tl.load(go_ptr + go_off, mask=m & b, other=0.0).to(CT)
                val += tl.where(m & b, go_val * w_val, 0.0).to(AT)

    gx_off = ((ni * ic + ci) * h + hi) * w + wi
    tl.store(gx_ptr + gx_off, val.to(CT), mask=m)


# ============================================================
# Backward: w.r.t w
# ============================================================
@triton.jit
def gci_conv_group_bwd_w(
    go_ptr, x_ptr, gw_ptr,
    n, ic, oc, h, w, ks, pad, groups,
    BS: tl.constexpr, CT: tl.constexpr, AT: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BS
    offs = start + tl.arange(0, BS)

    ni, gi, hi, wi, m = _idx(offs, n, groups, h, w)

    group_in = ic // groups
    group_out = oc // groups

    ci_base = gi * group_in
    co_base = gi * group_out

    for kh in range(ks):
        hin = hi - pad + kh
        hb = (hin >= 0) & (hin < h)
        for kw in range(ks):
            win = wi - pad + kw
            b = hb & (win >= 0) & (win < w)

            w_k_idx = kh * ks + kw
            w_off = ((ni * groups + gi) * ks * ks + w_k_idx) * h * w + hi * w + wi

            s_go = tl.zeros((BS,), dtype=AT)
            for s in range(group_out):
                co = co_base + s
                go_off = ((ni * oc + co) * h + hi) * w + wi
                go_val = tl.load(go_ptr + go_off, mask=m, other=0.0).to(CT)
                s_go += tl.where(m, go_val, 0.0).to(AT)

            s_x = tl.zeros((BS,), dtype=AT)
            for s in range(group_in):
                ci = ci_base + s
                x_off = ((ni * ic + ci) * h + hin) * w + win
                x_val = tl.load(x_ptr + x_off, mask=m & b, other=0.0).to(CT)
                s_x += tl.where(m & b, x_val, 0.0).to(AT)

            tl.store(gw_ptr + w_off, (s_x * s_go).to(CT), mask=m)


# ============================================================
# Autograd wrapper
# ============================================================
class GCIConvGroupAggFn(Function):
    @staticmethod
    # @custom_fwd(device_type="cuda")
    def forward(ctx, x, w, out_channels):
        n, ic, h, width = x.shape
        groups = w.shape[1]  # wc

        ks = int(math.sqrt(w.shape[2]))
        pad = (ks - 1) // 2

        ctx.ks = ks
        ctx.pad = pad
        ctx.groups = groups
        ctx.out_channels = out_channels

        o = torch.empty(n, out_channels, h, width, device=x.device, dtype=x.dtype)
        numel = o.numel()

        grid = lambda meta: _grid(numel, meta["BS"])

        ct = tl.float32 if x.dtype == torch.float32 else tl.float16
        at = tl.float32

        gci_conv_group_fwd[grid](
            x, w, o,
            n, ic, out_channels, h, width, ks, pad, groups,
            BS=1024,
            CT=ct,
            AT=at
        )

        ctx.save_for_backward(x, w)
        ctx.ct = ct
        ctx.at = at

        return o

    @staticmethod
    # @custom_bwd(device_type="cuda")
    def backward(ctx, go):
        x, w = ctx.saved_tensors
        ks, pad = ctx.ks, ctx.pad
        n, ic, h, width = x.shape
        oc = ctx.out_channels
        groups = ctx.groups

        go = go.contiguous()

        ct = ctx.ct
        at = ctx.at

        gx = gw = None

        if ctx.needs_input_grad[0]:
            gx = torch.empty_like(x)
            numel = gx.numel()
            gci_conv_group_bwd_x[lambda meta: _grid(numel, meta["BS"])](
                go, w, gx,
                n, ic, oc, h, width, ks, pad, groups,
                BS=1024, CT=ct, AT=at
            )

        if ctx.needs_input_grad[1]:
            gw = torch.empty_like(w)
            numel = gw.numel() // w.shape[2]
            gci_conv_group_bwd_w[lambda meta: _grid(numel, meta["BS"])](
                go, x, gw,
                n, ic, oc, h, width, ks, pad, groups,
                BS=1024, CT=ct, AT=at
            )

        return gx, gw, None


class GCIConvGroupAgg(nn.Module):
    def forward(self, x, w, out_channels):
        return GCIConvGroupAggFn.apply(x, w, out_channels)
