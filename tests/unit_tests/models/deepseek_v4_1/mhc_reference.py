# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FP64, per-token reference arithmetic for the V4.1 mHC CPU tests."""

import torch


def hc_post_reference(y, residual, post, comb):
    y, residual, post, comb = (value.double() for value in (y, residual, post, comb))
    branches = residual.shape[-2]
    return torch.stack(
        [
            post[..., q, None] * y + sum(comb[..., m, q, None] * residual[..., m, :] for m in range(branches))
            for q in range(branches)
        ],
        dim=-2,
    )


def hc_pre_reference(x, incoming, state, *, sinkhorn_iters, hc_eps, norm_eps):
    weight, base, scale = (state[name].double() for name in ("hc_fn", "hc_base", "hc_scale"))
    branches = x.shape[-2]
    outputs = []
    for token, previous in zip(x.double(), incoming.double(), strict=True):
        flat = token.reshape(-1)
        rms = (sum(value * value for value in flat) / flat.numel() + norm_eps).sqrt()
        logits = torch.stack([sum(a * b for a, b in zip(row, flat, strict=True)) for row in weight]) / rms
        pre = torch.sigmoid(logits[:branches] * scale[0] + base[:branches]) + hc_eps
        post = 2 * torch.sigmoid(logits[branches : 2 * branches] * scale[1] + base[branches : 2 * branches])
        comb = (logits[2 * branches :] * scale[2] + base[2 * branches :]).reshape(branches, branches)
        comb = torch.stack([(row - row.max()).exp() for row in comb])
        comb = torch.stack([row / row.sum() + hc_eps for row in comb])
        comb = torch.stack([column / (column.sum() + hc_eps) for column in comb.unbind(1)], dim=1)
        for _ in range(sinkhorn_iters - 1):
            comb = torch.stack([row / (row.sum() + hc_eps) for row in comb])
            comb = torch.stack([column / (column.sum() + hc_eps) for column in comb.unbind(1)], dim=1)
        collapsed = sum(previous[m] * token[m] for m in range(branches))
        outputs.append((collapsed, pre, post, comb))
    return tuple(torch.stack(values) for values in zip(*outputs, strict=True))
