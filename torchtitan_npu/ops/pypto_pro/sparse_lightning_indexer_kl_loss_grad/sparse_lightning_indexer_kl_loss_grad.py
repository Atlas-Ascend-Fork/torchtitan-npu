# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# This component's License is provided in torchtitan_npu/ops/pypto_pro/LICENSE.

# Keep the qualified DSL control flow and compile-time bindings unchanged.
# ruff: noqa: E501, RUF059, SIM102

"""A5 sparse Lightning Indexer KL gradients and student softmax.

Ported from pypto-gym commit 7ce340d64f46c0ef1025772ffb72abb5ca2be232,
custom/frozen_li_lig_20261018_v1. Kernel arithmetic and synchronization are
retained; development-only platform/build inspection helpers are excluded.

The qualified runtime is cann/pypto commit
e28aca0837d642333ad8b9fffc1b1e379fab9b1a. Its ``pypto`` distribution provides
the ``pypto_pro`` import package; ``pypto==0.2.1`` alone does not identify this
ABI. Build from that revision with ``build_ci.py`` following its
``docs/zh/install/build_and_install.md``, then install the matching package in
the active Python environment. Mismatch errors include the fixed-revision
build guide and the loaded JIT path to identify packages shadowed by CANN.
"""

import hashlib
import importlib
import inspect
import re
import threading
from pathlib import Path
from typing import cast

import torch

# Operator-owned launch configuration for mixed SIMT LIG kernels.
# Unrelated kernels, including LI, keep the native PyPTO launch path.
KERNEL_NAMES = frozenset(
    {
        "sparse_li_klgrad_pro_kernel_opt20261013finalatomicwide_public",
        "sparse_li_klgrad_pro_kernel_opt20261013finalatomicwidet_public",
        "sparse_li_klgrad_pro_kernel_opt20261013finaldwds_public",
        "sparse_li_klgrad_pro_kernel_opt20261013finaldwdst_public",
    }
)
_QUALIFIED_RUNTIME_COMMIT = "e28aca0837d642333ad8b9fffc1b1e379fab9b1a"
_QUALIFIED_RUNTIME_BUILD_GUIDE = (
    f"https://gitcode.com/cann/pypto/blob/{_QUALIFIED_RUNTIME_COMMIT}/docs/zh/install/build_and_install.md"
)
_QUALIFIED_JIT_SHA256 = "8584faadd0148c613f251f01fe8aaebb7d8e18e2216751aff7a696bcbe1dacca"
_QUALIFIED_CALLER_SHA256 = "302bedbf332e61630fd06ff371f6a5fd5d296320e753d28783f2554d1c78eedb"
_QUALIFIED_COMPILE_SHA256 = "e386d422a183f3edbc271dd339a849ec3c1d52fde86da0ace6e518a3c0150c06"
_STATE_KEY = "_li_lig_public_final_simt_launch_v1"
_LOCK_KEY = "_li_lig_public_final_simt_launch_lock_v1"


def install_simt_launch():
    """Install only the four named final-candidate launch configurations, once.

    No tensor values are read and no tensor operation is issued. Installation
    occurs at operator-module import, before the public wrapper is called.
    Unknown runtime versions fail before compilation instead of reusing an ABI
    that has not been verified. Unrelated kernels delegate to the original JIT.
    """
    jit = importlib.import_module("pypto_pro.runtime.jit")
    lock = jit.__dict__.setdefault(_LOCK_KEY, threading.RLock())
    with lock:
        previous = getattr(jit, _STATE_KEY, None)
        if previous is not None:
            if (
                previous["names"] != KERNEL_NAMES
                or jit._generate_caller_cpp is not previous["caller"]
                or jit._run_bisheng is not previous["compile"]
            ):
                raise RuntimeError("The LI/LIG launch configuration was changed after registration")
            return
        jit_file = getattr(jit, "__file__", None)
        if jit_file is None:
            raise RuntimeError(
                "PyPTO JIT source path is unavailable, so the LI/LIG SIMT launch ABI cannot be qualified: "
                f"jit.__file__={jit_file!r}. "
                f"Build and install the matching runtime from cann/pypto@{_QUALIFIED_RUNTIME_COMMIT}: "
                f"{_QUALIFIED_RUNTIME_BUILD_GUIDE}. "
                "pypto==0.2.1 alone does not guarantee the qualified ABI; use the matching file-backed JIT."
            )
        jit_path = Path(jit_file).resolve()
        actual_jit_sha256 = hashlib.sha256(jit_path.read_bytes()).hexdigest()
        if actual_jit_sha256 != _QUALIFIED_JIT_SHA256:
            raise RuntimeError(
                "PyPTO JIT is not qualified for the LI/LIG SIMT launch ABI: "
                f"jit={jit_path}, SHA256 actual={actual_jit_sha256}, expected={_QUALIFIED_JIT_SHA256}. "
                f"Build and install the matching runtime from cann/pypto@{_QUALIFIED_RUNTIME_COMMIT}: "
                f"{_QUALIFIED_RUNTIME_BUILD_GUIDE}. "
                "pypto==0.2.1 alone does not guarantee the qualified ABI; check the loaded JIT path."
            )
        for hook_name, function, expected in (
            ("_generate_caller_cpp", jit._generate_caller_cpp, _QUALIFIED_CALLER_SHA256),
            ("_run_bisheng", jit._run_bisheng, _QUALIFIED_COMPILE_SHA256),
        ):
            actual_hook_sha256 = hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
            if actual_hook_sha256 != expected:
                raise RuntimeError(
                    f"PyPTO launch hook {hook_name} differs from the qualified LI/LIG SIMT runtime: "
                    f"jit={jit_path}, hook SHA256 actual={actual_hook_sha256}, expected={expected}. "
                    f"Use an unmodified matching runtime from cann/pypto@{_QUALIFIED_RUNTIME_COMMIT}; "
                    f"build/install guidance: {_QUALIFIED_RUNTIME_BUILD_GUIDE}. "
                    "pypto==0.2.1 alone does not guarantee the qualified ABI; check for other launch-hook patches."
                )
        _install(jit)


def _install(jit):
    shared_ub = 221184
    original_caller = jit._generate_caller_cpp
    original_compile = jit._run_bisheng
    widths = {"int32_t": 4, "uint32_t": 4, "int64_t": 8, "uint64_t": 8, "float": 4, "double": 8, "bool": 1}

    def caller(
        kernel_params,
        kernel_cpp_name,
        kernel_name,
        has_cross_core_sync=False,
        enable_print_debug=False,
        global_entry="",
    ):
        text = original_caller(
            kernel_params, kernel_cpp_name, kernel_name, has_cross_core_sync, enable_print_debug, global_entry
        )
        if kernel_name not in KERNEL_NAMES:
            return text
        mixed = True
        # Current mixed candidates implement system event synchronization themselves;
        # their official compiler-generated caller has no extra FFTS pointer.
        if has_cross_core_sync:
            raise ValueError("FFTS argument ABI is not qualified by this operator")
        fields = []
        checks = []
        initializers = []
        layout = []
        end = 0
        for i, (typ, name, ptr) in enumerate(kernel_params):
            ctype = "void*" if ptr else typ
            size = 8 if ptr else widths.get(typ)
            if size is None:
                raise ValueError("Unverified scalar ABI type: " + typ)
            offset = (end + size - 1) // size * size
            fields.append(f"        {ctype} a{i};")
            checks.append(f'    static_assert(offsetof(SimtArgs,a{i}) == {offset}, "SIMT field offset");')
            initializers.append(name)
            layout.append({"name": name, "type": ctype, "offset": offset, "bytes": size})
            end = offset + size
        assert fields
        # Compiler evidence: original LIG=152B; one STATIC extent gives148B, despite
        # sizeof(SimtArgs)==152. Runtime argsSize excludes final struct padding.
        # smatomic2 binds dk instead of key; kernel_params preserves actual order.
        if mixed:
            assert len(layout) == 24 and end == 148, (kernel_name, layout, end)
            assert [x["offset"] for x in layout] == list(range(0, 96, 8)) + [96] + list(range(104, 148, 4))  # noqa: RUF005
        last = len(fields) - 1
        checks.append(
            f'    static_assert(offsetof(SimtArgs,a{last}) + sizeof(((SimtArgs*)nullptr)->a{last}) == {end}, "SIMT args size");'
        )
        struct = "    struct SimtArgs {\n" + "\n".join(fields) + "\n    };\n"
        call = struct + "\n".join(checks) + "\n"
        call += "    SimtArgs simt_args{" + ", ".join(initializers) + "};\n"
        call += (
            f"    __cce_rtKernelLaunchWithFlagV2((void*)&{kernel_name}, blockDim, "
            f'&simt_args, {end}, stream, {shared_ub}, (char*)"{kernel_name}", '
            f"{len(kernel_name)}, {1 if mixed else 0}, {2 if mixed else 1}, 0);"
        )
        lines = text.splitlines()
        hits = [i for i, line in enumerate(lines) if "<<<blockDim, nullptr, stream>>>" in line]
        assert len(hits) == 1
        lines[hits[0]] = call
        declaration = '#include <cstddef>\nextern "C" void __cce_rtKernelLaunchWithFlagV2(void*,unsigned,void*,unsigned,void*,unsigned,char*,unsigned long,int,int,int);\n'
        generated = declaration + "\n".join(lines) + "\n"
        return generated

    def compile_hook(bisheng_path, flags, arch, paths, link_args, compile_timeout, output_path):
        source = Path(paths.final_kernel).read_text()
        body_path = Path(paths.final_kernel).parent / "kernel.cpp"
        body = body_path.read_text() if body_path.exists() else ""
        selected = any(
            re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"\s*\(", source + "\n" + body) for name in KERNEL_NAMES
        )
        if (
            selected
            and "cce::async_invoke" in body
            and ("SimtArgs simt_args" not in source or f"stream, {shared_ub}," not in source)
        ):
            raise RuntimeError(
                "The LI/LIG SIMT kernel has no qualified launch metadata; check actual global function name"
            )
        extra = ["-cce-launch-with-flagv2-impl"] if selected and "SimtArgs simt_args" in source else []
        return original_compile(bisheng_path, [*flags, *extra], arch, paths, link_args, compile_timeout, output_path)

    jit._generate_caller_cpp = caller
    jit._run_bisheng = compile_hook
    setattr(jit, _STATE_KEY, {"names": KERNEL_NAMES, "caller": caller, "compile": compile_hook})


# Configure the qualified mixed SIMT launch before importing/decorating kernels.
install_simt_launch()

# The optional PyPTO-Pro SDK is unavailable in generic type-check CI.
import pypto_pro.language as pl  # pyrefly: ignore [missing-import]
from pypto_pro.language import Vf as vf  # noqa: N813  # pyrefly: ignore [missing-import]
from pypto_pro.runtime.tilingkey import TilingKeyField  # pyrefly: ignore [missing-import]

try:  # delivery-safe: NPU runtime is optional
    import torch_npu  # noqa: F401

    _HAS_TORCH_NPU = True
except ImportError:  # pragma: no cover - host without torch_npu
    _HAS_TORCH_NPU = False


# =====================================================================================
# DESIGN.md section 2.1 -- operator constants (copied verbatim; the coder may not change them)
# =====================================================================================
N1 = 64  # query.shape[1], head count -- NEVER split (SPEC 7.2(C))
D = 128  # head dim
TS = 64  # s-chunk width = columns of one pipeline task
CMP_RATIO_DEFAULT = 4
PRELOAD = 1  # cv_fusion_pipeline.md 4.2
MAX_DELAY = 2  # = max(stage delay)
CTX_DEPTH = 4  # >= MAX_DELAY + 1, pl.struct_array depth
# pl.struct_array's size argument must be an integer LITERAL: _struct_parser.py:81 tests
# isinstance(call.args[0], ast.Constant), so a module-level name is rejected.  Both rings below
# therefore spell "4" literally, and this assert is what stops the two from drifting apart.
assert CTX_DEPTH == 4, "pl.struct_array(4, ...) is spelled literally in both sections"

K_CHOICES = (512, 1024)  # KMAX, one compile variant per value (pl.STATIC, Tensor.md:57-66)
MASK_MODES = (0, 3)

# DESIGN section 3.1 -- UB(Vec) addresses and vector-domain mutex ids of the Module-1 tiles.
VA_KROW = 0x00000
VA_SI_CHUNK = 0x00200
VA_KK_UB = 0x00400
VA_NZ_STAGE = 0x0C400
MUTEX_KROW = [0, 1]
MUTEX_SI_CHUNK = [2, 3]
MUTEX_KK_UB = [4, 5]
MUTEX_NZ_STAGE = [7]

# STAGED-ONLY tile (removed at finalize): the int32 row that carries g_s out to the staged g_ids
# output.  It is placed PAST the end of the frozen DESIGN section 3.1 UB map of the WIDEST variant
# (K = 1024 ends at 0x35C00 = 220160 B), so it can never collide with a frozen address in either
# variant, and it reuses vector-domain mutex id 19 -- whose owner (si_chunk_out, Module 5) does not
# exist in this file (make_tile_group.md:55 "不同Tile间ID可重复").
VA_GID_UB_STAGED = 0x35C00
MUTEX_GID_UB_STAGED = [19]
# STAGED-ONLY landing tile for the section-17 VF copy of s_vec ([N1,TS] fp32 = 16384 B).  Placed
# after gid_ub, still past the end of the widest frozen variant, and reusing vector-domain mutex
# id 20 whose owner (dkrow_ub, Module 5) does not exist in this file.
VA_SCOPY_UB_STAGED = 0x35D00
MUTEX_SCOPY_UB_STAGED = [20]

# -------------------------------------------------------------------------------------
# Module 3 additions -- DESIGN section 3.1 frozen addresses (klg_plan.tile_table).
# z_row / y_row / pdi_row are [1, K], so their addresses depend on the variant; every tile
# BEFORE them in the UB list is K-independent, which is why one base constant suffices and
# the three rows simply stride by kmax*4.  Checked against the K=512 table:
#   z 205824  y 207872 (+2048)  pdi 209920 (+2048)  w 211968 (+2048)  dw 212224  red 212480
# -------------------------------------------------------------------------------------
VA_ROWS_BASE = 0x32400  # 205824 -- z_row, first of the [1,K] resident rows
VA_DS_UB = 0x14400  # 82944  -- ds_ub [64,64] bf16
MUTEX_Z_ROW = [12]
MUTEX_Y_ROW = [13]
MUTEX_PDI_ROW = [14]
MUTEX_W_ROW = [15]
MUTEX_DW_ROW = [16]
MUTEX_RED_ROW = [17]
MUTEX_DS_UB = [8]
LANES = 64  # fp32 vector register width, and TS == LANES here
NEG_INF = -1e30  # reduce_max identity, official sample test_softmax_tile_group_vf.py:56

# si3 -- PROMOTED TO DELIVERED by the BP-9 architect round (2026-09-25).  It is no longer a
# staged-only tile and no longer a divergence.  The ruling: `si_chunk_g`'s `next()` sits INSIDE the
# valid-task gate, so it fails requirement (ii) of the `previous()` invariant (the reader's gate must
# be the producer's gate shifted one tick) and `si_chunk_g.previous()` would be wrong on the first
# drain tick.  s3 therefore keeps its own GM load, and it ALIASES `zero_grp`: same address, same
# mutex 18.  Justified on lifetime -- `zero_grp`'s only use is the section 1.1 prologue, fenced off
# by the INTER_BLOCK barrier before any tick -- and sharing id 18 is also what orders those prologue
# stores against this load.  The alias was FORCED, not chosen: the per-domain mutex plan fills
# [0, 31] exactly (cube 0-10, vector 0-20, xcore 21-31), so there is no free id for a 22nd vector
# tile.  byte_budget is byte-identical before and after, so section 3.2's figures do not move.
# The address is K-dependent because the three [1, K] rows precede it, so it is computed at the
# declaration rather than frozen as a literal: VA_ROWS_BASE + 3*kmax*4 + 3*N1*4
#   K = 512  -> 0x33F00     K = 1024 -> 0x35700
# The former staged placement (0x39D00, past the end of the widest variant) is what freed the room
# the Module-4 staged dkg landing tile now occupies.
#
# HISTORICAL NOTE, kept because the record is the point.  DESIGN section 1.3 V10 needs this chunk's sparse_indices to form v_s, and in the
# delivered kernel s3 reads them as ``si_chunk_g.previous()`` -- s1 writes that slot one tick
# earlier.  ``previous()`` is BP-9 and BP-9 is unresolved (an architect item), so here s3 loads its
# OWN index segment straight from GM instead.  The design already has exactly this pattern for the
# other consuming stage: ``si_chunk_out`` is "s5's own index chunk; g/v recomputed in scalar code"
# (design_checks/klg_plan.py tile table).
#   WHAT IT COSTS: one 256 B DMA per pass-0 chunk, and -- the part that matters -- this module does
#   NOT exercise the DELIVERED si path.  If the delivered kernel later reads si_chunk_g.previous(),
#   the slot arithmetic of that read is UNVALIDATED by these tests.  That gap is real, it belongs in
#   the record next to BP-9, and it does not preempt the (a)/(b) call for the delivered kernel.
#   WHAT WOULD REMOVE IT: a resolution of BP-9 that makes previous() usable, after which this tile
#   and its pl.load are deleted and V10's si argument becomes ``si_chunk_g.previous()``.
# Placed past the end of the widest frozen variant (K = 1024 ends at 0x35C00), after gid_ub
# (0x35C00, 256 B) and scopy_ub (0x35D00, 16384 B), and reusing vector-domain mutex id 18 whose
# owner (zero_grp, a Module-5 kernel-prologue tile) does not exist in this file
# (make_tile_group.md:55 "不同Tile间ID可重复").
MUTEX_SI3 = [18]  # = zero_grp's id; see the note above


def _va_si3(kmax):
    """si3's address: it aliases zero_grp, which follows the three [1, K] rows."""
    return VA_ROWS_BASE + 3 * kmax * 4 + 3 * N1 * 4


# -------------------------------------------------------------------------------------
# Module 4 additions -- frozen DESIGN section 3.1 addresses (design_checks/klg_plan.py::tile_table)
# and the section 2.3 per-domain mutex ids.  Nothing here is re-derived.
# -------------------------------------------------------------------------------------
VA_KKT_UB = 0x08400  # Vec,   pl.transpose(kk_ub) result, ND   [D, TS] bf16
VA_DST_UB = 0x16400  # Vec,   pl.transpose(ds_ub) result, ND   [TS, N1] bf16
VA_DQP_VEC = 0x18400  # Vec,   dq chunk partial, Acc->Vec (B4)  [N1, D] fp32
VA_DKG_VEC = 0x20400  # Vec,   dkg chunk, Acc->Vec (B4)         [TS, D] fp32
VA_DQ_ROW = 0x28400  # Vec,   Module-5 fp32 dq accumulator -- reused STAGED here
VA_DQ_HALF = 0x30400  # Vec,   Module-5 bf16 dq half -- reused STAGED here
MUTEX_KKT_UB = [6]
MUTEX_DST_UB = [9]
MUTEX_DQP_VEC = [22]  # xcore domain
MUTEX_DKG_VEC = [21]  # xcore domain
MUTEX_DQ_ROW = [10]
MUTEX_DQ_HALF = [11]
# qt_l1 is DEPTH 2 in this file: see consequence (2) in the module docstring.  Slot 0 keeps the
# frozen address; slot 1 sits at the first byte past the frozen L1 end (0x20000 = 131072 of
# 524288), so no tile in the frozen map moves.  _tile_plan() still carries the frozen depth 1.
# -------------------------------------------------------------------------------------
# Module 5 additions.  Every address and every mutex id below is the FROZEN DESIGN section 3.1 /
# design_checks/klg_plan.py value, not re-derived: zero_grp [18] at _va_si3(kmax), then
# si_chunk_out [19], then dkrow_ub [20].  dq_row [10] and dq_half_bf16 [11] were already reserved
# and were borrowed by the staged B4 consumer; Module 5 is their real owner.
# -------------------------------------------------------------------------------------
DROW_REGS = D // LANES  # 2 registers cover one [1, D] fp32 dkg row
DQ_HALF_ROWS = N1 // 2  # 32: dq goes out in two [32, D] bf16 halves (V27)


def _va_si_chunk_out(kmax):
    """s5's own index chunk, immediately after zero_grp (which si3 aliases)."""
    return _va_si3(kmax) + D * 4


def _va_dkrow(kmax):
    """the single-slot [1, D] fp32 row the atomic store reads (section 17: never dkg_vec itself)."""
    return _va_si_chunk_out(kmax) + TS * 4


MUTEX_ZERO_GRP = [18]  # shared with si3 by design: disjoint lifetimes, same bytes
MUTEX_SI_CHUNK_OUT = [19]
MUTEX_DKROW = [20]
MA_QT_L1 = [0x04000, 0x20000]  # Mat,   query[t]^T NZ, depth 2                     (B3)
MA_KKT_L1 = [0x10000, 0x14000]  # Mat,  gathered keys^T NZ, depth 2      (B3)
MA_DS_L1 = [0x18000, 0x1A000]  # Mat,   dS NZ, depth 2                   (B3)
MA_DST_L1 = [0x1C000, 0x1E000]  # Mat,  dS^T NZ, depth 2                 (B3)
LA_M2 = 0x04000  # Left,  M2 lhs = dS
LA_M3 = 0x06000  # Left,  M3 lhs = dS^T
RA_M2 = 0x04000  # Right, M2 rhs = KK   (bytes of kkt_l1)
RA_M3 = 0x08000  # Right, M3 rhs = Q    (bytes of qt_l1)
CA_DQ_ACC = 0x04000  # Acc,   dq chunk partial, fractal 1024
CA_DKG_ACC = 0x0C000  # Acc,   dkg chunk, fractal 1024
# qt_l1 is now a CROSS-CORE tile, so its ids must come from the xcore band -- which is full.
# DESIGN :631's authorized fallback is id MERGING; kkt_l1 is the right partner because the two
# are touched in strict program order on both cores.  See consequence (3).  Cube id 10, which
# qt_l1 used to hold, is now UNUSED in this file.
MUTEX_QT_L1 = [26, 27]  # xcore band, SHARED with kkt_l1 (DESIGN :631 id merge)
MUTEX_KKT_L1 = [26, 27]
MUTEX_DS_L1 = [28, 29]
MUTEX_DST_L1 = [30, 31]
MUTEX_L_M2 = [1]
MUTEX_L_M3 = [2]
MUTEX_R_M2 = [4]
MUTEX_R_M3 = [5]
MUTEX_DQ_ACC = [7]
MUTEX_DKG_ACC = [8]

# STAGED-ONLY landing tile for the section-17 VF copy of dkg_vec, half a chunk at a time
# ([32, D] fp32 = 16384 B).  Placed at the address si3 vacated when the BP-9 round aliased it onto
# zero_grp, which is exactly the room this needs; mutex 11 is dq_half_bf16's, a Module-5 tile that
# does not exist in this file.  Half a chunk rather than a whole one because the whole would need
# 32768 B and only 33792 B are free above the frozen map, which gid_ub and scopy_ub already share.
VA_DKG_HALF_STAGED = 0x39D00
MUTEX_DKG_HALF_STAGED = [11]
DKG_HALF_ROWS = 32
WIDE_REGS = N1 * D // LANES  # 128 registers cover a [64, 128] fp32 tile
HALF_REGS = DKG_HALF_ROWS * D // LANES  # 64 registers cover [32, 128] fp32

# DESIGN section 2.1 A5 platform table: pinned-doc values, ALL BOX-PENDING (BP-1).  L0C is the one
# disputed reading (131072 B in api/config/pypto-set_conv_tile_shapes.md:128 vs the 256 KB of the
# matmul sample comment and KB arch-a5.md); the budget passes under both (design_checks/
# byte_budget.out.txt).  These numbers are used ONLY by _check_platform_constants(), which re-reads
# the box values at test time and compares.
A5_DOC_CAPACITY = {"UB": 253952, "L1": 524288, "L0A": 65536, "L0B": 65536, "L0C": 131072}
A5_L0C_ALT = 262144
A5_SPACE_KEYS = {"UB": "ub_size", "L1": "l1_size", "L0A": "l0_a_size", "L0B": "l0_b_size", "L0C": "l0_c_size"}
SPACE_ALIGN = {"UB": 32, "L1": 32, "L0A": 512, "L0B": 512, "L0C": 64}  # make_tile.md:44

# -------------------------------------------------------------------------------------
# Module 2 additions -- DESIGN section 3.1 frozen addresses, read from
# design_checks/klg_plan.py::tile_table (NOT re-derived).  Cube-domain mutex ids 0-10 and
# xcore-domain 21-31 per DESIGN section 2.3.
# -------------------------------------------------------------------------------------
MA_Q_L1 = 0x00000  # Mat, query[t] NZ, resident for the token   (plan addr 0)
MA_KK_L1 = [0x08000, 0x0C000]  # Mat, gathered keys, depth 2                (plan addr 32768)
LA_M1 = 0x00000  # Left,  M1 lhs = Q                          (plan addr 0)
RA_M1 = 0x00000  # Right, M1 rhs = KK^T                       (plan addr 0)
CA_S_ACC = 0x00000  # Acc,   S chunk accumulator, fractal 1024   (plan addr 0)
VA_S_VEC = 0x10400  # Vec,   S chunk after Acc->Vec drain        (plan addr 66560)
MUTEX_Q_L1 = [9]
MUTEX_KK_L1 = [24, 25]
MUTEX_L_M1 = [0]
MUTEX_R_M1 = [3]
MUTEX_S_ACC = [6]

# -------------------------------------------------------------------------------------
# TOKEN SPLIT (DESIGN section 5.1 alternative (b), adopted 2026-09-26): the two AIVs each
# hold one token and the Cube drains stream A with SingleModeVec0 and stream B with
# SingleModeVec1.  Stream B mirrors stream A: every L1 tile at +0x24000 (L1 top 0x24000 ->
# 0x48000 = 294912 of 524288) and every accumulator at +0x14000 (L0C top 0x18000 -> 0x28000
# = 163840 of 262144, measured capacity).  Stream A's addresses are UNCHANGED, so a bisect
# against the frozen control ``module12345 e6539a48da5fceec`` stays comparable.
# -------------------------------------------------------------------------------------
TS_L1_OFF = 0x24000  # L1 top of the single-stream layout = stream B's base
TS_ACC_OFF = 0x14000  # L0C top of the single-stream layout = stream B's base
MA_Q_L1_B = MA_Q_L1 + TS_L1_OFF
MA_QT_L1_B = [a + TS_L1_OFF for a in MA_QT_L1]
MA_KK_L1_B = [a + TS_L1_OFF for a in MA_KK_L1]
MA_KKT_L1_B = [a + TS_L1_OFF for a in MA_KKT_L1]
MA_DS_L1_B = [a + TS_L1_OFF for a in MA_DS_L1]
MA_DST_L1_B = [a + TS_L1_OFF for a in MA_DST_L1]
CA_S_ACC_B = CA_S_ACC + TS_ACC_OFF
CA_DQ_ACC_B = CA_DQ_ACC + TS_ACC_OFF
CA_DKG_ACC_B = CA_DKG_ACC + TS_ACC_OFF
# Stream B's cross-core L1 groups SHARE stream A's xcore mutex ids.  Sharing an id is
# conservative -- auto_mutex can only add ordering, never miscompute -- and it keeps the id
# budget at the 32 it already was (xcore 11 + max(cube 14, vector 21) = 32), so NO id merge
# is needed anywhere: not on l_m1/l_m2, not on r_m1/r_m2, not on any vector scratch tile.
MUTEX_KK_L1_B = MUTEX_KK_L1
MUTEX_KKT_L1_B = MUTEX_KKT_L1
MUTEX_QT_L1_B = MUTEX_QT_L1
MUTEX_DS_L1_B = MUTEX_DS_L1
MUTEX_DST_L1_B = MUTEX_DST_L1
# Stream B's cube-private tiles take fresh cube-domain ids from the unused 10..20 range.
MUTEX_Q_L1_B = [10]
MUTEX_S_ACC_B = [11]
MUTEX_DQ_ACC_B = [12]
MUTEX_DKG_ACC_B = [13]
MUTEX_S_VEC = [23]

# -------------------------------------------------------------------------------------
# CUBE TRANSPOSE (2026-09-30).  pl.transpose is 411.08 us for a [64,128] bf16 tile -- 90.3
# cycles/element, MEASURED (probe_tr_cost.py, NT sweep 1/8/32, 5/5 within 1%), and 47x more than
# the same tile's whole GM round trip (8.823 us).  It is a scalar loop in disguise.  The identity
# matmul does the same reordering on the 98.2%-idle cube for 9.156 us, bit-exact
# (probe_cubetr.py: ctr_mm 26.676 - ctr_no 17.520, EXACT=True), i.e. 44.9x cheaper.
#
# Addresses append past every frozen tile, so NO existing address moves (verified by
# _plan_conflicts + a byte-for-byte address diff against the control).
MA_I_L1 = 0x44000  # Mat, I[D,D] bf16, ONE copy for BOTH streams  (L1 278528 -> 311296)
LA_I = 0x08000  # Left, identity -> L0A, loaded once           (L0A 32768 -> 65536)
CA_QT_ACC = 0x28000  # Acc,  transpose result, stream A             (L0C 163840 -> 196608)
CA_QT_ACC_B = 0x30000  # Acc,  stream B                               (        -> 229376)
# VA_QT_VEC is COMPUTED, not a constant: the UB end moves with kmax (z_row / y_row / pdi_row are
# kmax-sized), 214016 at K=512 but 220160 at K=1024.  Same reason _va_si3(kmax) is a function.
# SINGLE slot: SingleModeVec0/1 write the two AIVs' OWN UB at this one address, exactly as s_vec
# does ("one handle, two drain MODES"), so it costs 32768 B and not 65536.

# All 32 mutex ids were already allocated (vector 0-20 + xcore 21-31), the same exhaustion that
# forced si3 to alias zero_grp's id 18.  These SHARE, following the precedent already running in
# the delivered kernel: MUTEX_QT_L1 == MUTEX_KKT_L1 == [26,27] with qt_l1 and kkt_l1 at DIFFERENT
# addresses (DESIGN :631 id merge).  The file's own rule applies: "sharing an id is conservative --
# auto_mutex can only add ordering, never miscompute".
MUTEX_I_L1 = MUTEX_KK_L1  # xcore band: i_l1 is built by a VF and read by the cube
MUTEX_LA_I = MUTEX_L_M1  # cube band
MUTEX_QT_ACC = MUTEX_S_ACC  # cube band
MUTEX_QT_ACC_B = MUTEX_S_ACC_B
MUTEX_QT_VEC = MUTEX_S_VEC  # xcore band, same role as s_vec; tr_vec reuses this slot for kkt

# ---- MILESTONE 3 (2026-10-07): kkt (V19) by Cube transpose + a dedicated TR window ----
RA_I = 0x0C000  # Right, identity rhs = KK^T free alias -> L0B  (L0B 49152 -> 65536 full)
MUTEX_RA_I = MUTEX_R_M1  # cube band; serialised reuse
MUTEX_I_NZ = MUTEX_DKG_VEC  # i_nz aliases dkg_vec (prologue-only); xcore band
QT_CAST_REGS = D * N1 // 128  # klg_qtcast_vf emits 128 contiguous bf16 per output register


# Two L1 generations, each containing five key blocks per stream.
# These declarations supersede the historical layout constants above.
MA_Q_L1 = 0x00000
MA_QT_L1 = [0x04000, 0x08000]
MA_Q_L1_B = 0x0C000
MA_QT_L1_B = [0x10000, 0x14000]
MA_DS_L1 = [0x18000, 0x1A000]
MA_DS_L1_B = [0x1C000, 0x1E000]
MA_KKCACHE_A = [0x20000, 0x24000, 0x28000, 0x2C000, 0x30000, 0x48000, 0x4C000, 0x50000, 0x54000, 0x58000]
MA_KKCACHE_B = [0x34000, 0x38000, 0x3C000, 0x40000, 0x44000, 0x5C000, 0x60000, 0x64000, 0x68000, 0x6C000]
# Same-address aliases always use the same id. Repeated generations and
# streams share ids conservatively, as the original two streams already do.
MUTEX_KKCACHE = [24, 25, 26, 27, 30, 24, 25, 26, 27, 30]

# DESIGN section 6 event table.  Every pair is INTRA_BLOCK; the number of pre-fired RELEASE
# credits equals the boundary depth (B1 2, B2 1) -- "credit != slot", DESIGN section 6.1.
EV_B1_READY = [0, 1]  # V->C  set after pl.insert / MTE3 ; wait before pl.move / MTE1
EV_B1_RELEASE = [2, 3]  # C->V  set after pl.move / MTE1   ; wait before insert / MTE3
EV_B2_READY = 4  # C->V  set after Acc->Vec / FIX   ; wait before reading s_vec / V
EV_B2_RELEASE = 5  # V->C  set after reading s_vec / V; wait before Acc->Vec / FIX
EV_B3_READY = [6, 7]  # V->C  set after the 3rd pl.insert / MTE3 ; wait before pl.move / MTE1
EV_B3_RELEASE = [8, 9]  # C->V  set after the 4 pl.move -> L0 / MTE1 ; wait before insert / MTE3
EV_B4_READY = 10  # C->V  set after both Acc->Vec / FIX ; wait before reading dqp_vec / V
EV_B4_RELEASE = 11
EV_TR_READY = 13  # C->V  cube set after Acc->Vec(kkt)/FIX ; vector wait before cast
EV_TR_RELEASE = 14  # V->C  vector set after reading tr_vec/V ; cube wait before next drain
EV_DK_ZERO = 12  # DESIGN :1099 event table, row DK_ZERO: one-shot, never reused             # V->C  set after the last staged store / V ; wait before Acc->Vec / FIX
# DESIGN :178 / :648: twelve ids, 0-11, one READY/RELEASE pair per boundary per slot
# (B1 2/2, B2 1/1, B3 2/2, B4 1/1) -- "credit != slot", no extra credits.


# =====================================================================================
# Pure compile-time planning helpers (DESIGN section 3.1 / 3.2).  No platform read, no assert
# and no dev-only import at module level (stage4-check #13).
# =====================================================================================
def _tile_plan(kmax):
    """Reproduce the frozen DESIGN section 3.1 on-chip address map for one ``K`` variant.

    Returns ``{"space": {name: (addr, slot_bytes, depth, lifetime)}, "totals": ..., "ends": ...}``.
    Modules 2-5 tiles are planned here as well even though this staged file instantiates only the
    Module-1 ones: the frozen addresses stay reserved, ``_check_platform_constants`` can validate
    the whole budget on the box, and the staged-only ``gid_ub`` can be placed past the end.
    """
    ub = []
    ub.append(("krow_g", 1 * D * 2, 2, "rotating"))
    ub.append(("si_chunk_g", TS * 4, 2, "rotating"))
    ub.append(("kk_ub_g", TS * D * 2, 2, "rotating"))
    ub.append(("kkt_ub", D * TS * 2, 1, "scratch"))
    ub.append(("nz_stage", TS * D * 2, 1, "scratch"))  # alias group: _t / _s share it
    ub.append(("s_vec_g", N1 * TS * 4, 1, "rotating"))
    ub.append(("ds_ub", N1 * TS * 2, 1, "scratch"))
    ub.append(("dst_ub", TS * N1 * 2, 1, "scratch"))
    ub.append(("dqp_vec_g", N1 * D * 4, 1, "rotating"))
    ub.append(("dkg_vec_g", TS * D * 4, 1, "rotating"))
    ub.append(("dq_row", N1 * D * 4, 1, "resident"))
    ub.append(("dq_half_bf16", 32 * D * 2, 1, "scratch"))
    ub.append(("z_row", kmax * 4, 1, "resident"))
    ub.append(("y_row", kmax * 4, 1, "resident"))
    ub.append(("pdi_row", kmax * 4, 1, "resident"))
    ub.append(("w_row", N1 * 4, 1, "resident"))
    ub.append(("dw_row", N1 * 4, 1, "resident"))
    ub.append(("red_row", N1 * 4, 1, "scratch"))
    ub.append(("zero_grp", D * 4, 1, "scratch"))
    ub.append(("si_chunk_out", TS * 4, 1, "resident"))
    ub.append(("dkrow_ub", D * 4, 1, "scratch"))

    mat = [
        ("q_l1", TS * D * 2, 1, "resident"),
        ("qt_l1", D * TS * 2, 1, "resident"),
        ("kk_l1_g", TS * D * 2, 2, "rotating"),
        ("kkt_l1_g", D * TS * 2, 2, "rotating"),
        ("ds_l1_g", N1 * TS * 2, 2, "rotating"),
        ("dst_l1_g", TS * N1 * 2, 2, "rotating"),
    ]
    left = [
        ("l_m1", N1 * D * 2, 1, "scratch"),
        ("l_m2", N1 * TS * 2, 1, "scratch"),
        ("l_m3", TS * N1 * 2, 1, "scratch"),
    ]
    right = [("r_m1", D * TS * 2, 1, "scratch"), ("r_m2", TS * D * 2, 1, "scratch"), ("r_m3", TS * D * 2, 1, "scratch")]
    acc = [
        ("s_acc", N1 * TS * 4, 1, "scratch"),
        ("dq_acc", N1 * D * 4, 1, "scratch"),
        ("dkg_acc", TS * D * 4, 1, "scratch"),
    ]

    plan = {}
    totals = {}
    ends = {}
    for space, items in (("UB", ub), ("L1", mat), ("L0A", left), ("L0B", right), ("L0C", acc)):
        align = SPACE_ALIGN[space]
        addr = 0
        table = {}
        total = 0
        for name, slot, depth, life in items:
            addr = (addr + align - 1) // align * align
            table[name] = (addr, slot, depth, life)
            addr += slot * depth
            total += slot * depth
        plan[space] = table
        totals[space] = total
        ends[space] = addr
    return {"plan": plan, "totals": totals, "ends": ends}


_UB_QT_BASE = _tile_plan(K_CHOICES[0])["ends"]["UB"] - 3 * K_CHOICES[0] * 4
for _k in K_CHOICES:
    assert _UB_QT_BASE + 3 * _k * 4 == _tile_plan(_k)["ends"]["UB"], "qt_vec closed form drifted"


def _va_qt_vec(kmax):
    """UB address of tr_vec (= qt_vec slot), CLOSED FORM in kmax (symbolic-safe in the @pl.jit body;
    calling _tile_plan there makes the frontend parse its python tuples as IR, F00003 on 238)."""
    return _UB_QT_BASE + 3 * kmax * 4


# =====================================================================================
# Runtime gates.  Called ONLY from a test body -- never at import time (stage4-check #13).
# =====================================================================================
_GATE_CACHE = {}


# =====================================================================================
# TilingKey (deviation D1) -- DESIGN section 5.4 launches with a tiling key; ``mask_mode`` is the
# one operator parameter that changes a code path, so it is the key field (tiling_key.md:16-27,
# sample pro_ops/fa/test_fa_tilingkey_attn_mask.py:129-132,443,750).  ``MaskMode`` is referenced in
# the kernel as a bare compile-time name (tiling_key.md:98-120), hence the file-level ruff F821.
# PyPTO injects MaskMode per specialization; it must not be a Python module global.
# =====================================================================================
class KlgTilingKey:
    MaskMode = TilingKeyField(bits=1, values=[0, 3])

    def is_valid(self, key):
        (mask_mode,) = key
        return mask_mode in MASK_MODES


# =====================================================================================
# VF helper (DESIGN section 1.7 step V4).  ``vector_selection: vf`` -- see deviation D2 for the
# ``select`` spelling of the ``* v_s`` multiply.
# =====================================================================================
@pl.vector_function
def klg_diagonal64(src, dst):
    pred = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    lanes = vf.arange(0, dtype=pl.DT_UINT16)
    mask31 = vf.full(31, pred, dtype=pl.DT_UINT16)
    mask1 = vf.full(1, pred, dtype=pl.DT_UINT16)
    mask63 = vf.full(63, pred, dtype=pl.DT_UINT16)
    mask_all = vf.full(64 * 64 - 1, pred, dtype=pl.DT_UINT16)
    lane = vf.and_(lanes, mask31, pred)
    twice_lane = vf.shift_left(lane, 1, pred)
    parity_src = vf.shift_right(lanes, 5, pred)
    parity_src = vf.and_(parity_src, mask1, pred)
    parity_dst = vf.shift_right(lanes, 6, pred)
    row0 = vf.add(twice_lane, parity_src, pred)
    col0 = vf.add(twice_lane, parity_dst, pred)
    for db in pl.range(0, 64 // 64):
        col = vf.adds(col0, 64 * db, pred)
        row = row0
        source = vf.shift_left(row, 6, pred)
        source = vf.add(source, col, pred)
        dest_col = vf.shift_left(col, 6, pred)
        for shift in pl.range(0, 32):
            dest = vf.add(dest_col, row, pred)
            values = vf.gather(src, source, pred)
            vf.scatter(dst, values, dest, pred)
            source = vf.adds(source, 2 * 64, pred)
            source = vf.and_(source, mask_all, pred)
            row = vf.adds(row, 2, pred)
            row = vf.and_(row, mask63, pred)


@pl.vector_function
def klg_diagonal128(src, dst):
    pred = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    lanes = vf.arange(0, dtype=pl.DT_UINT16)
    mask31 = vf.full(31, pred, dtype=pl.DT_UINT16)
    mask1 = vf.full(1, pred, dtype=pl.DT_UINT16)
    mask63 = vf.full(63, pred, dtype=pl.DT_UINT16)
    mask_all = vf.full(64 * 128 - 1, pred, dtype=pl.DT_UINT16)
    lane = vf.and_(lanes, mask31, pred)
    twice_lane = vf.shift_left(lane, 1, pred)
    parity_src = vf.shift_right(lanes, 5, pred)
    parity_src = vf.and_(parity_src, mask1, pred)
    parity_dst = vf.shift_right(lanes, 6, pred)
    row0 = vf.add(twice_lane, parity_src, pred)
    col0 = vf.add(twice_lane, parity_dst, pred)
    for db in pl.range(0, 128 // 64):
        col = vf.adds(col0, 64 * db, pred)
        row = row0
        source = vf.shift_left(row, 7, pred)
        source = vf.add(source, col, pred)
        dest_col = vf.shift_left(col, 6, pred)
        for shift in pl.range(0, 32):
            dest = vf.add(dest_col, row, pred)
            values = vf.gather(src, source, pred)
            vf.scatter(dst, values, dest, pred)
            source = vf.adds(source, 2 * 128, pred)
            source = vf.and_(source, mask_all, pred)
            row = vf.adds(row, 2, pred)
            row = vf.and_(row, mask63, pred)


@pl.vector_function
def klg_ident_vf(dst_tile, nrow):
    """Build I[nrow, D] bf16 in UB.  VERIFIED EXACT on the box (probe_cubetr_prims.py 8f042cb3):
    nonzero=128 (want 128), diag_ok=True.

    NEW BOX FACT this cost a round to find: ``vf.full``'s value must be a COMPILE-TIME constant.
    Passing a ``pl.range`` loop variable fails inside ``pypto::ir::...::GenerateBody`` with a
    generic F0FFFF -- a CODEGEN failure, not a device fault, and only legible with
    ASCEND_GLOBAL_LOG_LEVEL=1.  So the vf.eq/vf.select form (which needs ``vf.full(d)``) is dead;
    this is the zero-fill + one FIRST_ELEMENT store per diagonal element form, and every
    ``vf.full`` here takes a constant.  ``dist=FIRST_ELEMENT`` needs only dtype-width alignment
    (store_align.md:82), which is what makes the element at linear offset ``d * D + d`` reachable.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    zero = vf.full(0, dtype=pl.DT_BF16)
    one = vf.full(1, dtype=pl.DT_BF16)
    for r in pl.range(0, nrow):
        vf.store_align(dst_tile + r * D, zero, preg)
    for d in pl.range(0, nrow):
        vf.store_align(dst_tile + d * D + d, one, preg, dist=pl.StoreDist.FIRST_ELEMENT)


@pl.vector_function
def klg_qtcast_vf(dst_tile, src_tile, nreg):
    """fp32 -> bf16, 128 CONTIGUOUS bf16 per output register.  VERIFIED EXACT (0/8192 mismatched).

    The Acc must be fp32 and the drain cannot narrow it: TMatmul.hpp:145 "Acc Type support int32_t
    or float" and common.hpp:147 "Src data type only support float or int32_t" on the Acc->Vec
    TMOV.  So this cast is unavoidable.  ``vf.astype`` runs at the WIDER type's element count, so
    ``layout=CastLayout.ZERO`` places 64 results at the EVEN b16 lanes -- two casts plus a
    ``vf.de_interleave`` are needed for 128 contiguous bf16.  That is klg_ds_vf's shape, copied
    rather than re-derived.
    """
    p32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    p16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    for r in pl.range(0, nreg):
        v0 = vf.load_align(src_tile, r * 128)
        v1 = vf.load_align(src_tile, r * 128 + 64)
        c0 = vf.astype(v0, p32, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO)
        c1 = vf.astype(v1, p32, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO)
        lo, hi = vf.de_interleave(c0, c1, dtype=pl.DT_BF16)
        vf.store_align(dst_tile + r * 128, lo, p16)


@pl.vector_function
def klg_zero_key_tail_vf(kk_tile, first_row):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    zero = vf.full(0, dtype=pl.DT_BF16)
    for row in pl.range(first_row, TS):
        vf.store_align(kk_tile + row * D, zero, preg)


@pl.vector_function
def klg_zero_key_row_vf(kk_tile, off):
    """Initialize a tail or invalid row without fetching a discarded key from GM."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    zero = vf.full(0.0, dtype=pl.DT_BF16)
    vf.store_align(kk_tile + off, zero, preg)


@pl.simt.function(max_threads=1024)
def klg_simt_dk_atomic(
    values, indices, output, base: pl.DT_INT32, real_k: pl.DT_INT32, lim: pl.DT_INT32, ks: pl.DT_INT32
):
    tid = pl.simt.linear_thread_idx()
    live = real_k - base
    if live > TS:
        live = TS
    if live < 0:
        live = 0
    # Each 1024-thread iteration covers eight complete 128-element rows.
    # Keep the original last-row and invalid-index guards below.
    for i in pl.range(0, (live + 7) // 8):
        off = tid + i * 1024
        s = off // D
        d = off % D
        if base + s < real_k:
            si = indices[0, s]
            if si >= 0 and si < lim:
                pl.simt.atomic_add(output[ks + si, 0, d], values[s, d])


@pl.vector_function
def klg_place_vf(kk_tile, krow_tile, off, v_lanes):
    """Place one gathered key row at ``kk_ub + off``; an invalid column becomes an EXACT zero row.

    ``v_lanes`` is ``v_s * D``: 128 (all b16 lanes, update_mask.md:24-40) when the column is valid,
    0 when it is not, so ``vf.select`` picks the loaded row or the zero register lane by lane.
    """
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    preg_sel = vf.update_mask(v_lanes, dtype=pl.DT_BF16)
    row = vf.load_align(krow_tile, 0)  # [1,128] bf16 = one 256 B register
    zero = vf.full(0.0, dtype=pl.DT_BF16)
    out = vf.select(row, zero, preg_sel)
    vf.store_align(kk_tile + off, out, preg_all)  # subview.md:44-47 linear element offset


# =====================================================================================
# Module 3 VF helpers -- DESIGN section 1.3 V7..V15.
#
# EVERY call below was re-checked against the box-verified signature list after the first draft,
# and two of them were WRONG in that draft.  Recording both, because one would have compiled:
#   * ``vf.mul_add_dst`` is ``(src0, src1, preg, mode=None) -> dst`` -- THREE positional args with
#     an IMPLICIT destination, not ``(acc, a, b, preg)``.  The accumulator is the Python name the
#     result is assigned back to, and it must already hold the addend
#     (``mul_add_dst.md``: "dst寄存器既被读取（作为加数）又被写入（存储结果），调用前必须预初始化为加数值").
#     The official sample ``pro_ops/lightning_indexer/test_quant_lightning_indexer_vf.py:130-141``
#     is the shape copied here: ``sum0_r0 = vf.mul(...)`` outside the loop, then
#     ``sum0_r0 = vf.mul_add_dst(qk, w, preg)`` inside it.
#   * ``vf.mem_bar`` takes an optional ``mode: MemBarMode``.  Bare ``mem_bar()`` is VST_VLD
#     (store->load) ONLY; the WAR pairs below are ``VLD_VST`` and would not have been ordered by
#     the bare form.  Every barrier here names its mode and the pair it orders.
# A third draft error was found the same way: ``vf.relu`` is ``(src, preg, mode=None)``, not
# ``vf.relu(src)``.  The spelling used here is ``vf.max(s, zero_f, preg)``, which is what the
# official lightning-indexer sample uses for the same relu (:114-115) -- one fewer unverified op.
#
# Loop-carried scalars, masks and ``pl.min`` inside a ``@pl.vector_function`` body are all in the
# official softmax sample ``pro_ops/vf_api/test_softmax_tile_group_vf.py:67-110``; ``klg_softmax_vf``
# below is that sample's three-pass structure with the row loop removed.
# =====================================================================================
@pl.vector_function
def klg_rowinit_vf(z_row, y_row, pdi_row, dw_row, red_row, kmax):
    """V7: zero the per-row state this stage OWNS, on its own token-first task.

    DESIGN section 4 "状态归属": every per-row resident state is initialised, updated and written
    out by exactly ONE stage, because adjacent tokens share one pipeline -- while s1 is on token B,
    s3 is still finishing token A, so a prologue owned by the leading stage would clear state the
    lagging stage has not read yet.  s3_row owns all six.

    DEVIATION D-M3-3: ``pdi_row`` is zeroed here, which DESIGN section 1.3 V7 does not list.  It is
    required by the golden: ``di_row`` must be EXACTLY 0 at ``s >= real_k`` (golden_stage123
    ``dI = torch.where(pref, y*P - pv, 0)``), and the p load covers only ``[0, real_k)``, so
    without this the tail of the staged ``di_row`` would carry raw p.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    zero = vf.full(0.0, preg, dtype=pl.DT_FP32)
    n_reg = (kmax + LANES - 1) // LANES  # kmax is a trace-time Python int
    for r in pl.range(0, n_reg):
        vf.store_align(z_row + r * LANES, zero, preg)
        vf.store_align(y_row + r * LANES, zero, preg)
        vf.store_align(pdi_row + r * LANES, zero, preg)
    vf.store_align(dw_row, zero, preg)  # [1, N1] == exactly one register
    vf.store_align(red_row, zero, preg)  # P lives at red_row lane 0


@pl.vector_function
def klg_z_vf(s_vec, w_row, z_row, off, vts):
    """V8: z[s] = sum_j w[t,j] * relu(S[j,s]) for one chunk.  TS == one fp32 register.

    The golden is ``z = (relu(S) * w[:, :, None]).sum(dim=1)`` -- summation over heads in head
    order, which is the order of the j loop.  ``mul_add_dst`` keeps each product exact
    (no intermediate rounding), so only the 64 additions round.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pcol = vf.update_mask(vts, dtype=pl.DT_FP32)  # live columns of this chunk
    zero_f = vf.full(0.0, preg, dtype=pl.DT_FP32)
    acc = vf.full(0.0, preg, dtype=pl.DT_FP32)  # mul_add_dst addend, pre-set to 0
    for j in pl.range(0, N1):
        sj = vf.load_align(s_vec, j * TS)
        rj = vf.max(sj, zero_f, preg)  # relu, official-sample spelling
        wj = vf.load_align(w_row, j, dist=pl.LoadDist.BRC_B32)
        acc = vf.mul_add_dst(rj, wj, preg)  # acc += relu(S_j) * w_j, implicit dst
    vf.store_align(z_row + off, acc, pcol)


@pl.vector_function
def klg_pacc_vf(pdi_row, si_chunk, red_row, off, vts, lim):
    """V10: gate p in place and accumulate P = sum_{s < real_k} p_s * v_s at red_row lane 0.

    DEVIATION D-M3-2, and the one that resolves DESIGN section 1.3's own inconsistency.  Section 1.3
    gives V13 the signature ``klg_di_vf(y_row, pdi_row, red_row, real_k)`` -- no index argument --
    while V13's ``vector_selection`` api_sequence still lists ``vf.select(p, 0.0, v_mask)``.  The
    two cannot both hold: at row finalize the gate would need v_s for the WHOLE row, and s3 holds
    one chunk of indices at a time.  The golden settles it -- ``pv = pc * vf`` is formed once and
    ``dI = y*P - pv`` reads the ALREADY GATED value -- so the gate belongs here, at the one place
    where this chunk's indices are in hand, and V13 keeps the four-argument signature section 1.3
    gives it.  ``pdi_row`` therefore holds p in ``[0, real_k)`` only between V9 and this call.

    WHY THE WHOLE REGISTER IS STORED (2026-09-25).  Storing only the live prefix left
    ``pdi_row[real_k : roundup(real_k, 8))`` holding whatever the narrowed ``pl.load`` of p had put
    there, which is UNDEFINED.  That was the single failing gate on the first clean device run:
    di_row's tail, 10 of 11 cases, deficit exactly ``roundup(real_k, 8) - real_k`` per row.  The
    instruction was pinned by an asymmetry, not by the count -- ``klg_softmax_vf`` stores ``y_row``
    with the IDENTICAL masked ``vf.store_align(tile + r*LANES, reg, mreg)`` construct and its tail
    was exact 0 on all 11 cases, so a masked vector store does NOT write past its mask; the p load
    is the only narrowed DMA into this tile.

    WHICH TEST ACTUALLY EXERCISES THIS.  A host mutation study (the in-place gate removed, so V13
    subtracts raw p) is caught on ``neg1_oob_indices`` and MISSED on ``tail_mask3`` and
    ``tail_mask0`` -- because the gate is a no-op wherever every column of the prefix is already
    valid.  Counted over the eleven cases, invalid-columns-inside-the-prefix exist in exactly TWO:
    ``neg1_oob_indices`` (226 of 1504) and ``boundary_key`` (40 of 920); the other nine have zero.
    So a green run with either of those cases skipped does NOT test D-M3-2.  The per-case count is
    printed by the Module-1 gate as "N invalid columns are exact zero rows".

    SETTLED: both of those cases ran and passed on A5 with every invalid column an exact zero row --
    ``neg1_oob_indices`` 226 of 1504 and ``boundary_key`` 40 of 920, di_row max_abs_error=0 and its
    tail exact.  D-M3-2 is tested, not deferred.

    DEVIATION D-M3-4: the ``v_s`` mask is built with TWO ``vf.select`` calls instead of
    ``vf.ge``/``vf.lt``/``vf.and_``.  ``vf.and_`` on two MASK registers (as opposed to two integer
    reg_tensors) is the one construct in this stage whose semantics I could not pin down from
    either the signature or a sample, and two selects are exactly equivalent:
    ``select(select(p, 0, si >= 0), 0, si < lim)``.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    # A separate INT32 mask for the two integer compares.  FP32 and INT32 are both 32-bit, so the
    # mask layout is the same and the fp32 one would very likely do -- but "very likely" about a
    # mask width is not worth carrying into a box round, and this costs one instruction.
    preg_i = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
    pcol = vf.update_mask(vts, dtype=pl.DT_FP32)
    zero_f = vf.full(0.0, preg, dtype=pl.DT_FP32)
    zero_i = vf.full(0, preg_i, dtype=pl.DT_INT32)
    lim_i = vf.full(lim, preg_i, dtype=pl.DT_INT32)
    # RAW against the previous chunk's store of red_row, which is a store->load across two calls
    # of this body (KB vec-row-reduce-broadcast).  VST_VLD is mem_bar's default; it is named here
    # because the second barrier below is NOT the default and the two must not be confused.
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    p = vf.load_align(pdi_row, off)
    si = vf.load_align(si_chunk, 0)
    ge0 = vf.ge(si, zero_i, preg_i)
    ltl = vf.lt(si, lim_i, preg_i)
    pv0 = vf.select(p, zero_f, ge0)  # si < 0     -> 0
    pv = vf.select(pv0, zero_f, ltl)  # si >= lim  -> 0
    # Columns at s >= real_k inside this chunk are forced to an EXACT 0 and the WHOLE register is
    # stored, rather than storing only the live prefix.  This is what makes di_row's tail exact --
    # see the DMA note at the p load in the kernel.  For every chunk but the last, vts == TS, so
    # pcol is full and this is a no-op.
    pv_live = vf.select(pv, zero_f, pcol)  # s >= real_k -> exact 0
    vf.mem_bar(mode=pl.MemBarMode.VLD_VST)  # WAR: the load of pdi_row above
    vf.store_align(pdi_row + off, pv_live, preg)  # FULL register, aligned at off
    part = vf.reduce_sum(pv_live, pcol)  # lane 0, empty mask -> 0
    prev = vf.load_align(red_row, 0)
    tot = vf.add(prev, part, preg)
    vf.mem_bar(mode=pl.MemBarMode.VLD_VST)  # WAR: the load of red_row above
    vf.store_align(red_row, tot, preg)


@pl.vector_function
def klg_softmax_vf(z_row, y_row, real_k):
    """V11: masked softmax over [0, real_k): max -> broadcast -> exp_sub -> sum -> div.

    Structurally identical to the official sample ``pro_ops/vf_api/test_softmax_tile_group_vf.py``
    :76-110 (one row instead of a row loop), including the ``pl.min(LANES, ...)`` tail clamp that
    DESIGN section 6 (:1098) requires of every masked row op and the ``vf.full(reg, preg)``
    lane-0 broadcast that DESIGN :1152 names.

    ``real_k == 0`` makes ``n_regs`` zero, so nothing is written and ``y_row`` keeps the explicit
    zeros V7 put there -- which is the golden's ``torch.where((rk > 0)[:, None], ...)`` branch.

    DEVIATION D-M3-5: the ``red_row`` parameter of DESIGN section 1.3's signature is dropped.  It
    was there to land the reduction results; with the register broadcast it is never touched, and
    an unused tile argument is a cross-core-scanner hazard, not documentation.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (real_k + LANES - 1) // LANES
    row_max = vf.full(NEG_INF, preg, dtype=pl.DT_FP32)
    for r in pl.range(0, n_regs):
        valid = pl.min(LANES, real_k - r * LANES)
        mreg = vf.update_mask(valid, dtype=pl.DT_FP32)
        reg = vf.load_align(z_row, r * LANES)
        part = vf.reduce_max(reg, mreg)  # max -> lane0 (index -> lane1)
        row_max = vf.max(row_max, part, preg)
    row_max_b = vf.full(row_max, preg)  # Tensor mode: lane0 -> all lanes
    # (LOWEST default measured: align2)
    row_sum = vf.full(0.0, preg, dtype=pl.DT_FP32)
    for r in pl.range(0, n_regs):
        valid = pl.min(LANES, real_k - r * LANES)
        mreg = vf.update_mask(valid, dtype=pl.DT_FP32)
        reg = vf.load_align(z_row, r * LANES)
        e = vf.exp_sub(reg, row_max_b, mreg)  # dst_i = exp(src0_i - src1_i)
        part = vf.reduce_sum(e, mreg)
        row_sum = vf.add(row_sum, part, preg)
    row_sum_b = vf.full(row_sum, preg)
    for r in pl.range(0, n_regs):
        valid = pl.min(LANES, real_k - r * LANES)
        mreg = vf.update_mask(valid, dtype=pl.DT_FP32)
        reg = vf.load_align(z_row, r * LANES)
        e = vf.exp_sub(reg, row_max_b, mreg)
        out = vf.div(e, row_sum_b, mreg)
        vf.store_align(y_row + r * LANES, out, mreg)


@pl.vector_function
def klg_di_vf(y_row, pdi_row, red_row, real_k):
    """V13: dI_s = y_s * P - (p_s * v_s), overwriting pdi_row IN PLACE.

    ``pdi_row`` already holds the GATED p (see klg_pacc_vf / D-M3-2), so this is the golden's
    ``dI = y * P - pv`` term for term.  Lanes at ``s >= real_k`` are masked off and keep the zeros
    V7 wrote, which is the golden's ``torch.where(pref, ..., 0)``.

    DEVIATION D-M3-6: P is broadcast with ``vf.load_align(red_row, 0, dist=BRC_B32)`` and applied
    with ``vf.mul``, where DESIGN section 1.3 writes ``vf.muls(P)``.  ``vf.muls`` takes a SCALAR,
    and P is not a kernel scalar -- it lives at red_row lane 0 and there is no tile getval inside a
    VF body.  The broadcast load is the official lightning-indexer sample's idiom (:120, :138).
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    n_regs = (real_k + LANES - 1) // LANES
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)  # RAW: klg_pacc_vf's last red_row store
    p_b = vf.load_align(red_row, 0, dist=pl.LoadDist.BRC_B32)
    for r in pl.range(0, n_regs):
        valid = pl.min(LANES, real_k - r * LANES)
        mreg = vf.update_mask(valid, dtype=pl.DT_FP32)
        yr = vf.load_align(y_row, r * LANES)
        yp = vf.mul(yr, p_b, mreg)
        pv = vf.load_align(pdi_row, r * LANES)
        di = vf.sub(yp, pv, mreg)
        vf.mem_bar(mode=pl.MemBarMode.VLD_VST)  # WAR: the pdi_row load above
        vf.store_align(pdi_row + r * LANES, di, mreg)


@pl.vector_function
def klg_dw_vf(s_vec, pdi_row, dw_row, off, vts):
    """V14: dw[j] += sum_s relu(S[j,s]) * dI_s for one chunk.

    Summation order matches the golden's ``(relu_s * dI[:, None, :]).sum(dim=-1)``; no
    ``(P/Z)*A - B`` algebraic rearrangement.

    THIS HELPER HAS BEEN WRONG TWICE.  Both are recorded, because the second one is the reason
    Module 3 trapped on device and neither was visible to anything I could run on the host.

      1. The first draft called an invented ``vf.update_mask_at(j)`` and accumulated the whole
         register, which would have compiled and then produced dw[j] = sum over ALL heads.
      2. The correction stored the per-head scalar with ``vf.store_align(dw_row + j, acc, p1)``.
         That is a VECTOR STORE TO A 4*j BYTE OFFSET, and ``store_align`` requires the destination
         to be 32 B aligned under the default ``NORM`` dist (``store_align.md:82``).  56 of the 64
         stores it issued per chunk were illegal.  MEASURED, 2026-09-25: ladder rung 6 (this helper
         absent) RAN and rung 7 (this helper present, a two-line delta) FAULTED 507015, three times
         independently; and in ``probes/probe_klg_m3_align.py``, in a minimal vector-only kernel
         with no cross-core protocol and no pipeline, the old body FAULTED 507035 while the body
         below RAN and matched the CPU reference.  The align arms separate perfectly: offsets
         0/32/256 B RAN, offset 4 B FAULTED.

    WHY THE ADDRESS IS NOW ALWAYS ALIGNED.  The per-head scalar reaches element j through a ONE-HOT
    REGISTER instead of through the address.  ``update_mask(j+1)`` selects lanes 0..j and
    ``update_mask(j)`` selects lanes 0..j-1, so under ``MergeMode.ZEROING`` the difference of the
    two selected constants is 1.0 at lane j and 0.0 everywhere else.  The row is then read once and
    written once, both at offset 0 -- two memory accesses per chunk instead of 128, and no address
    in this helper that is not a register-width multiple.

    The arithmetic is UNCHANGED, and that is measured too, not argued: run through the full
    Module-3 tick plan on the host, this body reproduces the old one's dw BIT FOR BIT on all ten
    cases (max|delta| = 0 exactly, every case).  The fix is an addressing change and nothing else.

    HOW THIS BODY DIFFERS FROM THE ARM THAT WAS MEASURED, stated so the claim is not stronger than
    the evidence: exactly one added statement, the leading ``vf.mem_bar(VST_VLD)``.  The probe's
    ``dw_fixed`` kernel runs one chunk, so it has no previous chunk's store of ``dw_row`` to order
    against; here the accumulation spans chunks and that RAW barrier is required.  It is additive
    and it changes no address, so the alignment result carries over unchanged.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pcol = vf.update_mask(vts, dtype=pl.DT_FP32)
    zero_f = vf.full(0.0, preg, dtype=pl.DT_FP32)
    one_f = vf.full(1.0, preg, dtype=pl.DT_FP32)
    di = vf.load_align(pdi_row, off)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)  # RAW: the previous chunk's dw_row store
    acc = vf.load_align(dw_row, 0)  # aligned: the whole [1, N1] row
    for j in pl.range(0, N1):
        sj = vf.load_align(s_vec, j * TS)
        rj = vf.max(sj, zero_f, preg)
        pr = vf.mul(rj, di, pcol)
        part = vf.reduce_sum(pr, pcol)  # scalar in lane 0
        bc = vf.full(part, preg)  # Tensor mode: lane 0 -> all lanes
        # (LOWEST default measured: align2)
        mj1 = vf.update_mask(j + 1, dtype=pl.DT_FP32)  # lanes 0..j
        mj0 = vf.update_mask(j, dtype=pl.DT_FP32)  # lanes 0..j-1  (empty when j == 0)
        hi = vf.select(one_f, zero_f, mj1)
        lo = vf.select(one_f, zero_f, mj0)
        hot = vf.sub(hi, lo, preg)  # 1.0 at lane j ONLY
        acc = vf.mul_add_dst(bc, hot, preg)  # acc[j] += part_j, implicit dst
    vf.mem_bar(mode=pl.MemBarMode.VLD_VST)  # WAR: the dw_row read above
    vf.store_align(dw_row, acc, preg)  # ONE aligned store, offset 0


@pl.vector_function
def klg_ds_vf(s_vec, pdi_row, w_row, ds_ub, off, vts):
    """V15: dS[j, s] = round_bf16(dI_s * w_j * [S[j,s] > 0]) for one chunk, TWO heads per pass.

    READ THIS BEFORE CHANGING IT -- the two-head pairing is not a micro-optimisation, it is what
    makes the b32 -> b16 narrowing land CONTIGUOUSLY.

    One ds_ub row is TS = 64 bf16 = 128 B = HALF a vector register, while the fp32 source for one
    head is 64 lanes = a FULL register.  ``vf.astype(fp32 -> bf16)`` cannot produce 64 contiguous
    bf16: the instruction runs at the wider type's element count, so ``layout=CastLayout.ZERO``
    places the 64 results at the EVEN b16 indices of a 128-lane register (astype.md 功能说明,
    图2) -- storing that directly would write 256 B of alternating value/zero.  Two heads are
    therefore narrowed independently, both with ``ZERO``, and ``vf.de_interleave`` extracts the
    even elements of the CONCATENATION of its two sources (de_interleave.md:
    ``dstReg0_i = srcReg_{2i}``), giving

        d0 = [head j's 64 bf16][head (j+1)'s 64 bf16]   = exactly ds_ub rows j and j+1

    in one 256 B store.  ``d1`` is all zeros and is discarded, which is the same shape as DESIGN
    section 1.3's own ``m16, _ = vf.de_interleave(...)``.

    DIVERGENCE FROM THE DESIGN SKETCH, stated rather than smoothed over: V15's ``vector_selection``
    block merges an ``ONE``-layout half into a ``ZERO``-layout half and then stores at
    ``ds_ub + j*TS``.  That merge produces the INTERLEAVE of the two halves, not their
    concatenation, so taken literally it writes head j and head j+1 into alternate columns of row j.
    The sketch also routes a de-interleaved b16 mask (``m16``) into ``astype``; here the gate is
    applied to the VALUES by ``vf.select`` before the cast, so the cast mask is just this chunk's
    live-column prefix and no mask-width conversion is needed.

    THE LAYOUT READING IS NOW MEASURED, not inferred: ``ds_rounded`` is max_abs_error=0 against the
    golden on all eleven cases.  That pass is discriminating rather than lucky -- a host mutation
    study had already shown the gate catches BOTH plausible wrong layouts (de_interleave taking d1,
    and the design's literal ZERO/ONE merge, which interleaves the two heads instead of
    concatenating them) on 3/3 cases.  The fallback that is no longer needed, recorded because it
    was the alternative: a per-head ``vf.store_align(ds_ub + j*TS, c_j, pcol,
    dist=pl.StoreDist.PACK)``, which would have leaned on ``pl.StoreDist`` -- unverified -- instead
    of ``vf.de_interleave``, which the box confirmed.

    Columns at ``s >= real_k`` inside the chunk are zeroed by the cast mask, matching the golden's
    zero fill; the gate is ``S > 0`` strictly, so an exactly-zero S contributes nothing.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pcol = vf.update_mask(vts, dtype=pl.DT_FP32)
    pb16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    zero_f = vf.full(0.0, preg, dtype=pl.DT_FP32)
    di = vf.load_align(pdi_row, off)
    for jp in pl.range(0, N1 // 2):
        j0 = jp * 2
        s0 = vf.load_align(s_vec, j0 * TS)
        g0 = vf.gt(s0, zero_f, pcol)
        w0 = vf.load_align(w_row, j0, dist=pl.LoadDist.BRC_B32)
        m0 = vf.mul(di, w0, pcol)
        v0 = vf.select(m0, zero_f, g0)
        c0 = vf.astype(v0, pcol, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO, round_mode=pl.VFRoundMode.CAST_RINT)
        j1 = j0 + 1
        s1 = vf.load_align(s_vec, j1 * TS)
        g1 = vf.gt(s1, zero_f, pcol)
        w1 = vf.load_align(w_row, j1, dist=pl.LoadDist.BRC_B32)
        m1 = vf.mul(di, w1, pcol)
        v1 = vf.select(m1, zero_f, g1)
        c1 = vf.astype(v1, pcol, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO, round_mode=pl.VFRoundMode.CAST_RINT)
        d0, d1 = vf.de_interleave(c0, c1, dtype=pl.DT_BF16)
        vf.store_align(ds_ub + j0 * TS, d0, pb16)  # 128 bf16 = rows j0 and j0+1


@pl.vector_function
def klg_dwds_vf(s_vec, pdi_row, dw_row, w_row, ds_ub, off, vts):
    """Fused DW and dS. Same ascending-head DW reduction/FMA and dS RNE; one shared S load per head."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pcol = vf.update_mask(vts, dtype=pl.DT_FP32)
    pb16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    zero_f = vf.full(0.0, preg, dtype=pl.DT_FP32)
    one_f = vf.full(1.0, preg, dtype=pl.DT_FP32)
    di = vf.load_align(pdi_row, off)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    acc = vf.load_align(dw_row, 0)
    for jp in pl.range(0, N1 // 2):
        j0 = jp * 2
        s0 = vf.load_align(s_vec, j0 * TS)
        rj = vf.max(s0, zero_f, preg)
        pr = vf.mul(rj, di, pcol)
        part = vf.reduce_sum(pr, pcol)
        bc = vf.full(part, preg)
        mj1 = vf.update_mask(j0 + 1, dtype=pl.DT_FP32)
        mj0 = vf.update_mask(j0, dtype=pl.DT_FP32)
        hi = vf.select(one_f, zero_f, mj1)
        lo = vf.select(one_f, zero_f, mj0)
        hot = vf.sub(hi, lo, preg)
        acc = vf.mul_add_dst(bc, hot, preg)
        g0 = vf.gt(s0, zero_f, pcol)
        w0 = vf.load_align(w_row, j0, dist=pl.LoadDist.BRC_B32)
        m0 = vf.mul(di, w0, pcol)
        v0 = vf.select(m0, zero_f, g0)
        c0 = vf.astype(v0, pcol, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO, round_mode=pl.VFRoundMode.CAST_RINT)
        j1 = jp * 2 + 1
        s1 = vf.load_align(s_vec, j1 * TS)
        rj = vf.max(s1, zero_f, preg)
        pr = vf.mul(rj, di, pcol)
        part = vf.reduce_sum(pr, pcol)
        bc = vf.full(part, preg)
        mj1 = vf.update_mask(j1 + 1, dtype=pl.DT_FP32)
        mj0 = vf.update_mask(j1, dtype=pl.DT_FP32)
        hi = vf.select(one_f, zero_f, mj1)
        lo = vf.select(one_f, zero_f, mj0)
        hot = vf.sub(hi, lo, preg)
        acc = vf.mul_add_dst(bc, hot, preg)
        g1 = vf.gt(s1, zero_f, pcol)
        w1 = vf.load_align(w_row, j1, dist=pl.LoadDist.BRC_B32)
        m1 = vf.mul(di, w1, pcol)
        v1 = vf.select(m1, zero_f, g1)
        c1 = vf.astype(v1, pcol, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO, round_mode=pl.VFRoundMode.CAST_RINT)
        d0, d1 = vf.de_interleave(c0, c1, dtype=pl.DT_BF16)
        vf.store_align(ds_ub + j0 * TS, d0, pb16)
    vf.mem_bar(mode=pl.MemBarMode.VLD_VST)
    vf.store_align(dw_row, acc, preg)


# =====================================================================================
# Module 4 VF helpers.  All three walk a [rows, D] fp32 tile a 64-lane register at a time, so every
# address is a multiple of 64 fp32 = 256 B -- well clear of the 32 B store_align rule that trapped
# Module 3 (measured: offsets 0/32/256 B run, 4/8/16 B fault 507035).
# They exist because Module 5 is not in this file: the B4 boundary needs a consumer, both to observe
# dqp_vec / dkg_vec and to return the credit the Cube waits on.
# =====================================================================================
@pl.vector_function
def klg_dqcast_vf(src, dst, row0, n_rows):
    """V27: [n_rows, D] fp32 rows of ``src`` starting at ``row0`` -> [n_rows, D] bf16 in ``dst``.

    THE ONLY NEW HELPER MODULE 5 NEEDS, and DESIGN section 1.5's V27 line gets its SHAPE wrong: it
    writes ``klg_dqcast_vf(dq_row, dq_half_bf16, 0)`` as if the cast were shape-preserving.  It is
    not.  ``vf.astype`` with ``layout=CastLayout.ZERO`` runs at the WIDER type's element count, so 64
    fp32 lanes land at the EVEN b16 indices of a 128-lane register -- the same fact klg_ds_vf's
    docstring records, and why klg_ds_vf casts TWO fp32 registers and then de-interleaves.  One row
    of D = 128 fp32 therefore needs two casts plus ``vf.de_interleave`` to become 128 CONTIGUOUS
    bf16.  probe_klg_m5_constructs.py's ``cast`` arm measured this exact composition BIT-EXACT
    against torch's ``.to(torch.bfloat16)`` on 4096 elements including exact ties and values
    straddling a power of two, so CAST_RINT's ties-to-even is measured here rather than assumed
    (SPEC section 12 A8 requires ties-to-even for the dq narrowing).

    ADDRESSES: the register pair is ``r0 * LANES`` / ``r1 * LANES`` with ``r0 = (row0 + r) * 2``,
    NOT ``r * D`` / ``r * D + LANES``.  Both name the same bytes, but the second form makes
    probe_klg_vf_alignment.py report a 4 B step -- it cannot take the gcd of a ``Mult + Name``
    expression and falls back to the element size.  The fix is to write the offset in the form the
    checker already resolves, not to widen the checker: a checker widened to accept new code stops
    being a gate.  Steps are 512 B and 256 B on the loads and 256 B on the store.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pb16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    for r in pl.range(0, n_rows):
        r0 = (row0 + r) * 2
        r1 = r0 + 1
        v0 = vf.load_align(src, r0 * LANES)
        v1 = vf.load_align(src, r1 * LANES)
        c0 = vf.astype(v0, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO, round_mode=pl.VFRoundMode.CAST_RINT)
        c1 = vf.astype(v1, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO, round_mode=pl.VFRoundMode.CAST_RINT)
        d0, _d1 = vf.de_interleave(c0, c1, dtype=pl.DT_BF16)
        vf.store_align(dst + r * D, d0, pb16)


@pl.vector_function
def klg_wide_zero_vf(tile, n_reg):
    """Zero a [rows, D] fp32 tile.  STAGED: stands in for V23's dq_row prologue."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    zero = vf.full(0.0, preg, dtype=pl.DT_FP32)
    for r in pl.range(0, n_reg):
        vf.store_align(tile + r * LANES, zero, preg)


@pl.vector_function
def klg_wide_acc_vf(acc_tile, src_tile, n_reg):
    """``acc += src`` over a [rows, D] fp32 tile.  STAGED: stands in for V24.

    WHICH TEST ACTUALLY EXERCISES THE CROSS-CHUNK SUM.  Only a token with nce > 1 does, and a host
    mutation study (this accumulation replaced by a plain assignment, so only the last chunk
    survives) is caught on ``div_mask0`` and MISSED on ``tail_mask3`` and ``neg1_oob_indices``.
    Counted over the eleven cases, exactly TWO have multi-chunk tokens: ``div_mask0``
    (real_k = 512, nce = 8) and ``tail_mask0`` (real_k = 300, nce = 5).  The other nine -- including
    ``multi_tail``, whose 301 rows all have real_k <= 64 -- are single-chunk, so a green run with
    those two skipped does NOT test dq's accumulation.  The per-case chunk count is printed by
    ``_assert_store_window``.

    This is also the section-17 FIRST TOUCH of the cross-core ``dqp_vec``: a VECTOR instruction, not
    a DMA (KB framework-findings section 17, adopted at DESIGN :727 / :909).  The chunk sum lands in
    a LOCAL tile, and only that local tile is ever handed to ``pl.store``.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, n_reg):
        a = vf.load_align(acc_tile, r * LANES)
        b = vf.load_align(src_tile, r * LANES)
        c = vf.add(a, b, preg)
        vf.mem_bar(mode=pl.MemBarMode.VLD_VST)  # WAR: the acc_tile load above
        vf.store_align(acc_tile + r * LANES, c, preg)


@pl.vector_function
def klg_wide_copy_vf(dst_tile, src_tile, src_off, n_reg):
    """Copy ``n_reg`` registers from ``src_tile + src_off`` into ``dst_tile``.

    Section 17 again: ``dkg_vec`` is cross-core, so it is read by a VF instruction and forwarded to
    GM out of a local tile.  Half a chunk at a time -- ``src_off`` is 0 or ``DKG_HALF_ROWS * D``,
    both multiples of 64 elements, so both are 256 B aligned.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for r in pl.range(0, n_reg):
        reg = vf.load_align(src_tile, src_off + r * LANES)
        vf.store_align(dst_tile + r * LANES, reg, preg)


# =====================================================================================
# Kernel -- the ONE @pl.jit function of this file (PL02).  Modules 2-5 extend THIS function.
# =====================================================================================
@pl.jit(auto_mutex=True, tiling_key=KlgTilingKey)
def sparse_li_klgrad_pro_kernel_opt20261013finaldwds_public(
    query: pl.Tensor[[pl.DYNAMIC, N1, D], pl.DT_BF16],
    key: pl.Tensor[[pl.DYNAMIC, 1, D], pl.DT_BF16],
    weights: pl.Tensor[[pl.DYNAMIC, N1], pl.DT_FP32],
    sparse_indices: pl.Tensor[[pl.DYNAMIC, 1, pl.STATIC], pl.DT_INT32],
    attn_softmax_l1_norm: pl.Tensor[[pl.DYNAMIC, 1, pl.STATIC], pl.DT_FP32],
    cu_seqlens_q: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    cu_seqlens_k: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    cmp_residual_k: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
    dq_out: pl.Tensor[[pl.DYNAMIC, N1, D], pl.DT_BF16],
    dk_out: pl.Tensor[[pl.STATIC, 1, D], pl.DT_FP32],
    dw_out: pl.Tensor[[pl.DYNAMIC, N1], pl.DT_FP32],
    softmax_out: pl.Tensor[[pl.DYNAMIC, 1, pl.STATIC], pl.DT_FP32],
    cmp_ratio: pl.DT_INT64,
):
    # FOUR THINGS I COULD NOT VERIFY WITHOUT THE BOX, AND HOW EACH WAS SETTLED.  Kept as a record
    # rather than deleted: what is worth having is which of them the eventual failures were, and
    # none of them was.  All four are now measured (module123 11/11 on A5, one process per case).
    #  1. ``c3.pass_`` is READ here; Modules 1-2 only ever WROTE that field, and its name ends in an
    #     underscore.  SETTLED: every ladder rung compiled and ran -- a struct-field read is fine.
    #  2. ``vf.full(reg, preg)`` Tensor-mode broadcast and ``pl.min`` inside a VF body, both from the
    #     official sample ``pro_ops/vf_api/test_softmax_tile_group_vf.py:83-90``, neither on the
    #     box-verified signature list.  SETTLED TWICE: ladder rung 5 (klg_softmax_vf) RAN, and
    #     ``probes/probe_klg_m3_align2.py`` measured the PRIMITIVE -- broadcasting a register holding
    #     0.0..63.0 yields lane 0 in every lane, so ``DuplicatePos`` defaults to ``LOWEST`` and both
    #     call sites here are right.  ``pos=pl.DuplicatePos.LOWEST`` also compiles and behaves
    #     identically, so it is available but not required.
    #  3. ``vf.de_interleave``'s concatenation semantics (klg_ds_vf) -- the LAYOUT reading was mine,
    #     from de_interleave.md's dstReg0_i = srcReg_2i.  SETTLED: ``ds_rounded`` is max_abs_error=0
    #     against the golden on all eleven cases, and a host mutation study had already shown the
    #     gate catches two different wrong layouts on 3/3 cases, so the pass is discriminating.
    #  4. ``pl.load`` from a 2-D tensor into a [1, N1] tile with no ``order`` (w_row, dw_out).
    #     SETTLED before the device: ``pl.load``'s own docstring gives the default as "last N axes
    #     ascending", the identity for a 2-D tensor into a 2-D tile; then ladder rung 1 RAN and
    #     ``dw`` is max_abs_error=0, which exercises both the load of w_row and the store of dw.
    # WHAT ACTUALLY FAILED was on neither list, twice, and both were 32 B granularity rules:
    # ``vf.store_align`` needs a 32 B destination (the trap), and a ``pl.load`` narrowed by
    # ``set_validshape`` leaves the rest of its last 32 B block undefined (the di_row tail).
    # ``probes/probe_klg_vf_alignment.py`` now checks both classes statically.
    # ---- top level: shape / core arithmetic only (DESIGN section 4.2; LI sample :521-526) ----
    t1 = sparse_indices.shape[0]
    t2 = key.shape[0]
    kmax = sparse_indices.shape[2]  # pl.STATIC -> compile-time in this variant
    n_batch = cmp_residual_k.shape[0]
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()

    # ---------------------------------------------------------------------------------
    # (1) Cross-core shared groups -- declared OUTSIDE both sections (DESIGN section 4.2 step 1;
    # reference ..._dynrank.py:630-646).  Producer and consumer each call .next() once per VALID
    # task under the SAME gate, so their independent cursors select the same physical slot
    # (DESIGN section 2.3 rule 1).
    # ---------------------------------------------------------------------------------
    kk_l1_nz_g = pl.make_tile_group(  # B1 producer view: pl.insert target
        type=pl.TileType(shape=[TS, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=MA_KKCACHE_A,
        mutex_ids=MUTEX_KKCACHE,
    )
    kk_l1_zn_g = pl.make_tile_group(  # B1 consumer view: SAME L1 bytes, ZN
        type=pl.TileType(shape=[D, TS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=MA_KKCACHE_A,
        mutex_ids=MUTEX_KKCACHE,
    )
    s_vec_g = pl.make_tile_group(  # B2, single slot
        type=pl.TileType(shape=[N1, TS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=VA_S_VEC,
        mutex_ids=MUTEX_S_VEC,
    )
    # ---- MILESTONE 3 kernel-body groups.  tr_vec: cube drains kkt^T here (fp32), vector casts it;
    # ONE slot, two drain MODES (Vec0/Vec1) exactly like s_vec.  i_l1: built once in the prologue,
    # read every tick by the cube.  Both cross the section boundary so both are kernel-body level.
    # ---- B3: three L1 groups, depth 2, one event PAIR covering all three (DESIGN :1093).
    # kkt_l1 needs the same two-view spelling that B1 needed and the probe measured (BP-2b "b2"):
    # the producer inserts into the NZ view and the consumer moves from a ZN view of the SAME bytes,
    # because pl.insert's source must be NZ while a Right operand must be ZN.  ds_l1 and dst_l1 need
    # no alias -- their Left operands are NZ with the same logical shape. ----
    kkt_l1_nz_g = pl.make_tile_group(  # producer view: pl.insert target
        type=pl.TileType(shape=[TS, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=MA_KKCACHE_A,
        mutex_ids=MUTEX_KKCACHE,
    )
    kkt_l1_zn_g = pl.make_tile_group(  # consumer view: SAME L1 bytes, ZN
        type=pl.TileType(shape=[TS, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=MA_KKCACHE_A,
        mutex_ids=MUTEX_KKCACHE,
    )
    # ---- qt_l1, MOVED HERE from the cube section by the qt workaround.  It used to be
    # cube-private (the Cube loaded it itself with the transposing pl.load that this file removes);
    # it is now produced by the Vector and consumed by the Cube, so it is a cross-core group and
    # belongs in this block.  Same two-view spelling as kkt_l1 for the same reason -- pl.insert's
    # source must be NZ, a Right operand must be ZN, and both views are the SAME L1 bytes.  DEPTH 2:
    # see consequence (2) in the module docstring for the WAR argument that forces it. ----
    qt_l1_nz_g = pl.make_tile_group(  # producer view: pl.insert target
        type=pl.TileType(shape=[N1, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=MA_QT_L1,
        mutex_ids=MUTEX_QT_L1,
    )
    qt_l1_zn_g = pl.make_tile_group(  # consumer view: SAME L1 bytes, ZN
        type=pl.TileType(shape=[N1, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=MA_QT_L1,
        mutex_ids=MUTEX_QT_L1,
    )
    ds_l1_g = pl.make_tile_group(
        type=pl.TileType(shape=[N1, TS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=MA_DS_L1,
        mutex_ids=MUTEX_DS_L1,
    )
    # CONSUMER-SIDE VIEWS of the two B3 groups that do not need a layout change.  Identical type,
    # address and mutex ids; the ONLY thing they add is a SECOND CURSOR.
    #
    # FOUND BY A STATIC CHECK, 2026-09-25, while bisecting an unrelated score_s corruption, and
    # fixed on its own merits rather than on that bisect's outcome -- ladder rung8, which is exactly
    # this change, came back CORRUPT, so this is NOT the cause of that defect.  What it IS: of the
    # four cross-core rotating groups, kk_l1 and kkt_l1 are declared as two objects each (an NZ
    # producer view and a ZN consumer view) because pl.insert needs NZ while a Right operand needs
    # ZN, so each section advances its own cursor.  ds_l1 and dst_l1 needed no layout change, so
    # they were declared once and had .next() called on them from BOTH sections.  DESIGN section 2.3
    # rule 1 says that is legal ("两个 Section ... 各自持有该句柄的游标") and the official sample does
    # it -- but module123's 11/11 exercised only the TWO-object form, so the one-object form has
    # never run on this box, and two cursors on one cross-core buffer drifting apart by one is a
    # failure mode the sibling LI operator is chasing right now.  This costs two declarations and
    # removes the question.
    ds_l1_c_g = pl.make_tile_group(
        type=pl.TileType(shape=[N1, TS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=MA_DS_L1,
        mutex_ids=MUTEX_DS_L1,
    )
    dst_l1_c_g = pl.make_tile_group(
        type=pl.TileType(shape=[TS, N1], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=MA_DS_L1,
        mutex_ids=MUTEX_DS_L1,
    )

    # ---- TOKEN SPLIT: stream B's mirror of every cross-core L1 group.  Same shapes, same
    # depths, same two-view spellings, addresses +0x24000, and the SAME xcore mutex ids as
    # stream A (DESIGN section 2.3: sharing an id is conservative -- it can only serialise,
    # never miscompute -- and it keeps the id budget at the 32 it already was).  s_vec /
    # dqp_vec / dkg_vec are deliberately NOT mirrored: one handle, two drain MODES.
    kk_l1_nz_g2 = pl.make_tile_group(  # B1 producer view: pl.insert target
        type=pl.TileType(shape=[TS, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=MA_KKCACHE_B,
        mutex_ids=MUTEX_KKCACHE,
    )
    kk_l1_zn_g2 = pl.make_tile_group(  # B1 consumer view: SAME L1 bytes, ZN
        type=pl.TileType(shape=[D, TS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=MA_KKCACHE_B,
        mutex_ids=MUTEX_KKCACHE,
    )
    kkt_l1_nz_g2 = pl.make_tile_group(  # producer view: pl.insert target
        type=pl.TileType(shape=[TS, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=MA_KKCACHE_B,
        mutex_ids=MUTEX_KKCACHE,
    )
    kkt_l1_zn_g2 = pl.make_tile_group(  # consumer view: SAME L1 bytes, ZN
        type=pl.TileType(shape=[TS, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=MA_KKCACHE_B,
        mutex_ids=MUTEX_KKCACHE,
    )
    qt_l1_nz_g2 = pl.make_tile_group(  # producer view: pl.insert target
        type=pl.TileType(shape=[N1, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=MA_QT_L1_B,
        mutex_ids=MUTEX_QT_L1_B,
    )
    qt_l1_zn_g2 = pl.make_tile_group(  # consumer view: SAME L1 bytes, ZN
        type=pl.TileType(shape=[N1, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=MA_QT_L1_B,
        mutex_ids=MUTEX_QT_L1_B,
    )
    ds_l1_g2 = pl.make_tile_group(
        type=pl.TileType(shape=[N1, TS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=MA_DS_L1_B,
        mutex_ids=MUTEX_DS_L1_B,
    )
    ds_l1_c_g2 = pl.make_tile_group(
        type=pl.TileType(shape=[N1, TS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=MA_DS_L1_B,
        mutex_ids=MUTEX_DS_L1_B,
    )
    dst_l1_c_g2 = pl.make_tile_group(
        type=pl.TileType(shape=[TS, N1], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=MA_DS_L1_B,
        mutex_ids=MUTEX_DS_L1_B,
    )
    # ---- B4: two single-slot Vec groups, one event pair covering both (DESIGN :1095) ----
    dqp_vec_g = pl.make_tile_group(
        type=pl.TileType(shape=[N1, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=VA_DQP_VEC,
        mutex_ids=MUTEX_DQP_VEC,
    )
    dkg_vec_g = pl.make_tile_group(
        type=pl.TileType(shape=[TS, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=VA_DKG_VEC,
        mutex_ids=MUTEX_DKG_VEC,
    )

    # =================================================================================
    # (1b) dk ZERO PROLOGUE + INTER_BLOCK barrier -- DESIGN section 1.1, verbatim.
    #
    # pl.store(atomic=AtomicAdd) requires the destination pre-set to 0 (store.md:47) and the wrapper
    # may only torch.empty (wrapper-boundary.md), so the kernel's FIRST vector block writes the whole
    # of dk[T2, 1, D] to zero.  The INTER_BLOCK fence then guarantees every Block's zeroing has
    # landed before ANY Block issues its first AtomicAdd (set_cross_core_wait_cross_core.md:40,
    # mode 0).  The rendezvous count depends only on shape and core info, so the AIC and both AIVs
    # agree; this is also the ONE place both AIV sub-cores do work, because there is no row state --
    # DESIGN :839 names it as the exception to the sub_id == 0 rule.
    #
    # BOTH CONSTRUCTS HERE WERE UNEXERCISED ANYWHERE IN THIS TREE AND ARE NOW MEASURED:
    # probe_klg_m5_constructs.py, arms ``inter_block`` (NaN left 0, unique [2.0]) and ``atomic``
    # (NaN left 0, unique [8.0], i.e. every one of the 8 workers contributed exactly once and no
    # update was lost).  The NaN pre-fill is what makes the first of those load-bearing: the zero
    # loop is strided by ``workers``, so a wrong ``workers`` would leave un-zeroed rows and the NaN
    # would survive.  ``atomic=`` is documented at store.md:27/:53 with a dk-shaped example at :106.
    with pl.section_vector():
        zero_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=_va_si3(kmax),
            mutex_ids=MUTEX_ZERO_GRP,
        )
        # ---- MILESTONE 3: prologue identity build, ZERO new UB (both aliases, prologue-only) ----
        worker0 = pl.get_block_idx()
        workers0 = pl.get_block_num() * pl.get_subblock_num()
        zrow = zero_g.current()
        klg_wide_zero_vf(zrow, DROW_REGS)  # [1, D] fp32 = 2 whole registers, offsets r*64
        for gz in pl.range(worker0, t2, workers0):
            pl.store(dk_out, zrow, [gz, 0, 0], order=[0, 2])
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=EV_DK_ZERO, sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK)
        pl.system.wait_cross_core(
            pipe=pl.PipeType.MTE3, event_id=EV_DK_ZERO, sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK
        )

    # =================================================================================
    # (2) CUBE section -- Module 2, stage s2_score (delay 1).  DESIGN section 4.2 skeleton: the
    # two sections are PARALLEL top-level blocks, each with its own complete loop nest and its
    # own context ring filled by the identical section 4.1 decoder.
    # =================================================================================
    with pl.section_cube():
        # BOX FIX: section_cube has its own scalar scope (F00004); t2 is kernel-body level.
        t2_last = pl.max(t2 - 1, 0)
        q_l1_g = pl.make_tile_group(
            type=pl.TileType(shape=[N1, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=MA_Q_L1,
            mutex_ids=MUTEX_Q_L1,
        )
        l_m1_g = pl.make_tile_group(
            type=pl.TileType(shape=[N1, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=LA_M1,
            mutex_ids=MUTEX_L_M1,
        )
        r_m1_g = pl.make_tile_group(
            type=pl.TileType(shape=[D, TS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addrs=RA_M1,
            mutex_ids=MUTEX_R_M1,
        )
        # ---- MILESTONE 3 cube transpose operands (shared by both streams, serialised) ----
        s_acc_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[N1, TS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
            ),
            addrs=CA_S_ACC,
            mutex_ids=MUTEX_S_ACC,
        )
        l_m2_g = pl.make_tile_group(
            type=pl.TileType(shape=[N1, TS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=LA_M2,
            mutex_ids=MUTEX_L_M2,
        )
        l_m3_g = pl.make_tile_group(
            type=pl.TileType(shape=[TS, N1], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=LA_M3,
            mutex_ids=MUTEX_L_M3,
        )
        r_m2_g = pl.make_tile_group(
            type=pl.TileType(shape=[TS, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addrs=RA_M2,
            mutex_ids=MUTEX_R_M2,
        )
        r_m3_g = pl.make_tile_group(
            type=pl.TileType(shape=[N1, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addrs=RA_M3,
            mutex_ids=MUTEX_R_M3,
        )
        dq_acc_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[N1, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
            ),
            addrs=CA_DQ_ACC,
            mutex_ids=MUTEX_DQ_ACC,
        )
        dkg_acc_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[TS, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
            ),
            addrs=CA_DKG_ACC,
            mutex_ids=MUTEX_DKG_ACC,
        )

        # ---- TOKEN SPLIT: stream B's q_l1 and its three accumulators.  L0C is 262144 B
        # (measured), so duplicating the accumulators is exactly the form the BP arm ran.
        # L0A / L0B are NOT duplicated: the two streams reuse them serially.
        q_l1_g2 = pl.make_tile_group(
            type=pl.TileType(shape=[N1, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=MA_Q_L1_B,
            mutex_ids=MUTEX_Q_L1_B,
        )
        s_acc_g2 = pl.make_tile_group(
            type=pl.TileType(
                shape=[N1, TS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
            ),
            addrs=CA_S_ACC_B,
            mutex_ids=MUTEX_S_ACC_B,
        )
        dq_acc_g2 = pl.make_tile_group(
            type=pl.TileType(
                shape=[N1, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
            ),
            addrs=CA_DQ_ACC_B,
            mutex_ids=MUTEX_DQ_ACC_B,
        )
        dkg_acc_g2 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TS, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
            ),
            addrs=CA_DKG_ACC_B,
            mutex_ids=MUTEX_DKG_ACC_B,
        )
        # Consumer pre-fires one RELEASE credit per B1 slot (DESIGN section 6.1 "credit != slot";
        # reference ..._dynrank.py:802-805 pre-fires every backward event before the loop).
        pl.system.set_cross_core(
            pipe=pl.PipeType.MTE1, event_id=EV_B1_RELEASE[0], sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK
        )
        pl.system.set_cross_core(
            pipe=pl.PipeType.MTE1, event_id=EV_B1_RELEASE[1], sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK
        )
        # The Cube is also the B3 CONSUMER, so it pre-fires one RELEASE per B3 slot
        # (DESIGN :969 "每个边界的消费者在自己的主循环之前，为每个空槽各发一次 RELEASE").
        pl.system.set_cross_core(
            pipe=pl.PipeType.MTE1, event_id=EV_B3_RELEASE[0], sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK
        )
        pl.system.set_cross_core(
            pipe=pl.PipeType.MTE1, event_id=EV_B3_RELEASE[1], sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK
        )

        ctick = 0
        cslot = 0  # B1 slot parity, advanced per VALID task
        c4slot = 0  # B3 slot parity, per VALID pass-1 task
        # pl.struct_array's first argument must be an integer LITERAL (_struct_parser.py:81 tests
        # isinstance(call.args[0], ast.Constant)); the assert beside CTX_DEPTH is what keeps the
        # literal and the constant from drifting apart.
        cctx = pl.struct_array(
            4,
            "CubeCtx",
            t=0,
            pass_=0,
            chunk=0,
            real_k=0,
            vts=0,
            ks=0,
            avail=0,
            lim=0,
            g_safe=0,
            is_valid=0,
            first=0,
            last=0,
            cache=0,
            gen=0,
            b1=0,
            b3=0,
        )

        cctx2 = pl.struct_array(
            4,
            "CubeCtxB",
            t=0,
            pass_=0,
            chunk=0,
            real_k=0,
            vts=0,
            ks=0,
            avail=0,
            lim=0,
            g_safe=0,
            is_valid=0,
            first=0,
            last=0,
            cache=0,
            gen=0,
            b1=0,
            b3=0,
        )

        # TOKEN SPLIT: the outer loop walks token PAIRS, not single tokens.  A pair is two
        # ADJACENT tokens (base, base+1) because the Vector's worker id is 2*core_id+sub_id with
        # stride 2*num_cores, so AIV0 takes `base` and AIV1 takes `base+1`.  Iterating over pairs
        # rather than writing `pl.range(worker, t1, workers)` is REQUIRED, not stylistic: the
        # strided form gives the two sub-cores DIFFERENT iteration counts when t1 is odd, which is
        # the zero-iteration rendezvous defect this file warns about in four places.
        n_pairs = pl.max((t1 - 2 * core_id + 2 * num_cores - 1) // (2 * num_cores), 1)
        for pk in pl.range(0, n_pairs):
            base = 2 * core_id + 2 * num_cores * pk
            # ---- stream A: DESIGN 4.1 decoder, COPIED verbatim, locals suffixed _A ----
            # ===== DESIGN section 4.1 stateless row decoder (identical code in every section) ====
            t_A = base
            b_A = -1
            for bb_A in pl.range(0, n_batch):  # no early break: q_len_A == 0 batches exist
                lo_A = pl.getval(cu_seqlens_q, bb_A)
                hi_A = pl.getval(cu_seqlens_q, bb_A + 1)
                if lo_A <= t_A:
                    if t_A < hi_A:
                        b_A = bb_A
            qs_A = 0
            q_len_A = 0
            ks_A = 0
            kv_len_A = 0
            res_A = 0
            if b_A >= 0:
                qs_A = pl.getval(cu_seqlens_q, b_A)
                q_len_A = pl.getval(cu_seqlens_q, b_A + 1) - qs_A
                ks_A = pl.getval(cu_seqlens_k, b_A)
                kv_len_A = pl.getval(cu_seqlens_k, b_A + 1) - ks_A
                res_A = pl.getval(cmp_residual_k, b_A)
            pos_A = t_A - qs_A
            avail_A = kv_len_A
            real_k_A = pl.min(kv_len_A, kmax)
            if MaskMode == 3:  # pyrefly: ignore [unknown-name]
                avail_A = pl.min(kv_len_A + 1, t2 - ks_A)
                num_A = kv_len_A * cmp_ratio + res_A - q_len_A + pos_A + 1
                vis_A = 0
                if num_A > 0:  # NEVER divide a non-positive numerator
                    vis_A = num_A // cmp_ratio
                real_k_A = pl.min(pl.max(pl.min(vis_A, avail_A), 0), kmax)
            lim_A = avail_A  # validity_mode "avail_A"
            if b_A < 0:
                real_k_A = 0  # row outside every batch -> all zeros
            nce_A = pl.max((real_k_A + TS - 1) // TS, 1)  # >= 1: zero iterations break rendezvous
            g_safe_A = pl.min(pl.max(ks_A, 0), t2_last)  # always a legal DMA address
            # ---- stream B: DESIGN 4.1 decoder, COPIED verbatim, locals suffixed _B ----
            # ===== DESIGN section 4.1 stateless row decoder (identical code in every section) ====
            t_B = base + 1
            b_B = -1
            for bb_B in pl.range(0, n_batch):  # no early break: q_len_B == 0 batches exist
                lo_B = pl.getval(cu_seqlens_q, bb_B)
                hi_B = pl.getval(cu_seqlens_q, bb_B + 1)
                if lo_B <= t_B:
                    if t_B < hi_B:
                        b_B = bb_B
            qs_B = 0
            q_len_B = 0
            ks_B = 0
            kv_len_B = 0
            res_B = 0
            if b_B >= 0:
                qs_B = pl.getval(cu_seqlens_q, b_B)
                q_len_B = pl.getval(cu_seqlens_q, b_B + 1) - qs_B
                ks_B = pl.getval(cu_seqlens_k, b_B)
                kv_len_B = pl.getval(cu_seqlens_k, b_B + 1) - ks_B
                res_B = pl.getval(cmp_residual_k, b_B)
            pos_B = t_B - qs_B
            avail_B = kv_len_B
            real_k_B = pl.min(kv_len_B, kmax)
            if MaskMode == 3:  # pyrefly: ignore [unknown-name]
                avail_B = pl.min(kv_len_B + 1, t2 - ks_B)
                num_B = kv_len_B * cmp_ratio + res_B - q_len_B + pos_B + 1
                vis_B = 0
                if num_B > 0:  # NEVER divide a non-positive numerator
                    vis_B = num_B // cmp_ratio
                real_k_B = pl.min(pl.max(pl.min(vis_B, avail_B), 0), kmax)
            lim_B = avail_B  # validity_mode "avail_B"
            if b_B < 0:
                real_k_B = 0  # row outside every batch -> all zeros
            nce_B = pl.max((real_k_B + TS - 1) // TS, 1)  # >= 1: zero iterations break rendezvous
            g_safe_B = pl.min(pl.max(ks_B, 0), t2_last)  # always a legal DMA address
            # ---- COMMON tick count.  nce is max over the pair, so BOTH sub-cores and the AIC
            # compute the same value and every event, slot parity and .next() keeps firing under
            # the SAME gate `tk < 2 * nce` it fires under today.  That is what stops this walking
            # into rung 6 ("with the EVENTS guarded AIV1 never arrives and the AIC waits forever"):
            # per-stream validity would make the gate differ between sub-cores.
            # The short stream simply runs its extra chunks with vts == 0, which is an ALREADY
            # EXERCISED, PASSING path -- it is exactly what every real_k == 0 row does today
            # (case realk_zero_rows), so those chunks contribute exactly zero and no stage body
            # needs a new gate.  Cost measured at 0.35% aggregate and 0.00% on both timing cases
            # (scratchpad/rungs/pairing_efficiency.py: div_mask0 2.000x, multi_tail 1.993x).
            nce = pl.max(pl.max(nce_A, nce_B), 1)
            drain = 0
            if base + 2 * num_cores >= t1:
                drain = MAX_DELAY
            # =====================================================================================
            for tk in pl.range(0, 2 * nce + drain):
                # ---- ctx fill, one ring per stream.  `nce` is the pair's max, so is_valid /
                # pass_ / chunk / first / last come out IDENTICAL on both streams; the short
                # stream's extra chunks simply have vts == 0, which every real_k == 0 row in
                # case realk_zero_rows already exercises and passes.
                ccur = cctx[ctick % CTX_DEPTH]
                ccur.is_valid = 0
                if tk < 2 * nce:
                    cchunk = tk % nce
                    ccur.t = t_A
                    ccur.pass_ = tk // nce
                    ccur.chunk = cchunk
                    ccur.gen = pk % 2
                    ccur.cache = 0
                    ccur.b1 = tk % 2
                    ccur.b3 = 2 + cchunk % 2
                    if nce <= 5:
                        ccur.cache = 1
                        ccur.b1 = cchunk
                        ccur.b3 = cchunk
                    ccur.real_k = real_k_A
                    ccur.vts = pl.min(TS, pl.max(real_k_A - cchunk * TS, 0))
                    ccur.ks = ks_A
                    ccur.avail = avail_A
                    ccur.lim = lim_A
                    ccur.g_safe = g_safe_A
                    ccur.first = 0
                    if tk == 0:
                        ccur.first = 1
                    ccur.last = 0
                    if tk == 2 * nce - 1:
                        ccur.last = 1
                    ccur.is_valid = 1

                ccur2 = cctx2[ctick % CTX_DEPTH]
                ccur2.is_valid = 0
                if tk < 2 * nce:
                    cchunk2 = tk % nce
                    ccur2.t = t_B
                    ccur2.pass_ = tk // nce
                    ccur2.chunk = cchunk2
                    ccur2.gen = pk % 2
                    ccur2.cache = 0
                    ccur2.b1 = tk % 2
                    ccur2.b3 = 2 + cchunk2 % 2
                    if nce <= 5:
                        ccur2.cache = 1
                        ccur2.b1 = cchunk2
                        ccur2.b3 = cchunk2
                    ccur2.real_k = real_k_B
                    ccur2.vts = pl.min(TS, pl.max(real_k_B - cchunk2 * TS, 0))
                    ccur2.ks = ks_B
                    ccur2.avail = avail_B
                    ccur2.lim = lim_B
                    ccur2.g_safe = g_safe_B
                    ccur2.first = 0
                    if tk == 0:
                        ccur2.first = 1
                    ccur2.last = 0
                    if tk == 2 * nce - 1:
                        ccur2.last = 1
                    ccur2.is_valid = 1

                # ---------- Module 2, stage s2_score (delay 1, DESIGN section 1.4) ----------
                c2 = cctx[(ctick + CTX_DEPTH - 1) % CTX_DEPTH]
                # ---------- Module 2, stage s2_score (delay 1, DESIGN section 1.4) ----------
                c22 = cctx2[(ctick + CTX_DEPTH - 1) % CTX_DEPTH]
                # ONE gate for both streams -- see the note above on shared ctx flags.
                if c2.is_valid == 1:
                    # ---- per-stream operand handles and the B1 cursor, both streams ----
                    q_l1 = q_l1_g.current()
                    l_m1 = l_m1_g.current()
                    r_m1 = r_m1_g.current()
                    s_acc = s_acc_g.current()
                    s_vec = s_vec_g.current()

                    # C1: s2 OWNS q_l1 and loads it on the token's own first task.
                    if c2.first == 1:
                        pl.load(q_l1, query, [pl.min(c2.t, t1 - 1), 0, 0], order=[1, 2])
                    # C2: B1 hand-off.  The ZN alias is BP-2b spelling b2 -- DESIGN section 1.4
                    # writes spelling b1, which the probe proved does NOT compile.
                    kk_zn = kk_l1_zn_g[c2.gen * 5 + c2.b1]

                    q_l12 = q_l1_g2.current()
                    l_m12 = l_m1_g.current()
                    r_m12 = r_m1_g.current()
                    s_acc2 = s_acc_g2.current()
                    s_vec2 = s_vec_g.current()

                    # C1: s2 OWNS q_l12 and loads it on the token's own first task.
                    if c22.first == 1:
                        pl.load(q_l12, query, [pl.min(c22.t, t1 - 1), 0, 0], order=[1, 2])
                    # C2: B1 hand-off.  The ZN alias is BP-2b spelling b2 -- DESIGN section 1.4
                    # writes spelling b1, which the probe proved does NOT compile.
                    kk_zn2 = kk_l1_zn_g2[c22.gen * 5 + c22.b1]

                    # ---- B1 window: ONE event pair covering BOTH streams' kk_l1 ----
                    if cslot == 0:
                        pl.system.wait_cross_core(
                            pipe=pl.PipeType.MTE1, event_id=EV_B1_READY[0], sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK
                        )
                    else:
                        pl.system.wait_cross_core(
                            pipe=pl.PipeType.MTE1, event_id=EV_B1_READY[1], sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK
                        )
                    pl.move(r_m1, kk_zn)  # L1 -> L0B
                    pl.move(l_m1, q_l1)  # C3: L1 -> L0A
                    pl.set_validshape(s_acc, [N1, TS])  # C4: one block,
                    pl.matmul(s_acc, l_m1, r_m1)  #     no phase

                    pl.move(r_m12, kk_zn2)  # L1 -> L0B
                    pl.move(l_m12, q_l12)  # C3: L1 -> L0A
                    pl.set_validshape(s_acc2, [N1, TS])  # C4: one block,
                    pl.matmul(s_acc2, l_m12, r_m12)  #     no phase

                    if cslot == 0:
                        pl.system.set_cross_core(
                            pipe=pl.PipeType.MTE1, event_id=EV_B1_RELEASE[0], sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK
                        )
                    else:
                        pl.system.set_cross_core(
                            pipe=pl.PipeType.MTE1, event_id=EV_B1_RELEASE[1], sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK
                        )
                    cslot = 1 - cslot

                    # ---- B2 window: ONE event pair, both drains inside, A to AIV0's UB
                    # with SingleModeVec0 and B to AIV1's with SingleModeVec1 ----
                    # C5: B2 hand-off, Acc -> Vec over FIX.
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.FIX, event_id=EV_B2_RELEASE, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK
                    )
                    pl.move(s_vec, s_acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                    pl.move(s_vec2, s_acc2, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec1)
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.FIX, event_id=EV_B2_READY, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK
                    )

                # ---------- Module 4, stage s4_grad (delay 2, DESIGN section 1.4) ----------
                # Gate is `is_valid && pass == 1`, which is EXACTLY the gate the Vector's B3
                # producer uses -- DESIGN section 2.3 rule 1: both sides call .next() once per valid
                # task under the same condition, so their independent cursors select the same slot.
                c4 = cctx[(ctick + CTX_DEPTH - 2) % CTX_DEPTH]
                # ---------- Module 4, stage s4_grad (delay 2, DESIGN section 1.4) ----------
                # Gate is `is_valid && pass == 1`, which is EXACTLY the gate the Vector's B3
                # producer uses -- DESIGN section 2.3 rule 1: both sides call .next() once per valid
                # task under the same condition, so their independent cursors select the same slot.
                c42 = cctx2[(ctick + CTX_DEPTH - 2) % CTX_DEPTH]
                if c4.is_valid == 1:
                    if c4.pass_ == 1:
                        # qt_l1's cursor advances once per TOKEN, under the same gate the
                        # Vector's producer uses (is_valid && pass == 1 && chunk == 0), so
                        # DESIGN section 2.3 rule 1 holds.  The handle is taken from an
                        # UNCONDITIONAL .current() because DESIGN :775 records that a variable
                        # assigned inside a branch goes out of scope in DSL codegen.
                        if c4.chunk == 0:
                            qt_l1_zn_g.next()
                        qt_zn = qt_l1_zn_g.current()
                        l_m2 = l_m2_g.current()
                        l_m3 = l_m3_g.current()
                        r_m2 = r_m2_g.current()
                        r_m3 = r_m3_g.current()
                        dq_acc = dq_acc_g.current()
                        dkg_acc = dkg_acc_g.current()
                        dqp_vec = dqp_vec_g.current()
                        dkg_vec = dkg_vec_g.current()
                        # C6 IS GONE.  ``pl.load(qt_nz, query, [c4.t, 0, 0], order=[2, 1])``
                        # stood here and is the statement the 25-rung bisect isolated; the
                        # Vector now produces qt_l1 by transpose instead (V16a..V16d below).
                        # This removes the only descending order= in the whole operator.

                        # C7: the B3 hand-off.  One event pair covers all three buffers: all three
                        # inserts complete before the Vector's set, all four moves complete before
                        # the Cube's release (DESIGN :1067).
                        ds_l1 = ds_l1_c_g.next()
                        dst_l1 = dst_l1_c_g.next()
                        kkt_zn = kkt_l1_zn_g[c4.gen * 5 + c4.b3]

                        # qt_l1's cursor advances once per TOKEN, under the same gate the
                        # Vector's producer uses (is_valid && pass == 1 && chunk == 0), so
                        # DESIGN section 2.3 rule 1 holds.  The handle is taken from an
                        # UNCONDITIONAL .current() because DESIGN :775 records that a variable
                        # assigned inside a branch goes out of scope in DSL codegen.
                        if c42.chunk == 0:
                            qt_l1_zn_g2.next()
                        qt_zn2 = qt_l1_zn_g2.current()
                        l_m22 = l_m2_g.current()
                        l_m32 = l_m3_g.current()
                        r_m22 = r_m2_g.current()
                        r_m32 = r_m3_g.current()
                        dq_acc2 = dq_acc_g2.current()
                        dkg_acc2 = dkg_acc_g2.current()
                        dqp_vec2 = dqp_vec_g.current()
                        dkg_vec2 = dkg_vec_g.current()
                        # C6 IS GONE.  ``pl.load(qt_nz, query, [c42.t, 0, 0], order=[2, 1])``
                        # stood here and is the statement the 25-rung bisect isolated; the
                        # Vector now produces qt_l1 by transpose instead (V16a..V16d below).
                        # This removes the only descending order= in the whole operator.

                        # C7: the B3 hand-off.  One event pair covers all three buffers: all three
                        # inserts complete before the Vector's set, all four moves complete before
                        # the Cube's release (DESIGN :1067).
                        ds_l12 = ds_l1_c_g2.next()
                        dst_l12 = dst_l1_c_g2.next()
                        kkt_zn2 = kkt_l1_zn_g2[c42.gen * 5 + c42.b3]

                        # ---- B3 window: ONE event pair covering BOTH streams ----
                        if c4slot == 0:
                            pl.system.wait_cross_core(
                                pipe=pl.PipeType.MTE1,
                                event_id=EV_B3_READY[0],
                                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
                            )
                        else:
                            pl.system.wait_cross_core(
                                pipe=pl.PipeType.MTE1,
                                event_id=EV_B3_READY[1],
                                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
                            )
                        pl.move(l_m2, ds_l1)  # L1 -> L0A
                        pl.move(r_m2, kkt_zn)  # L1 -> L0B (ZN view)
                        pl.move(l_m3, dst_l1)  # L1 -> L0A
                        pl.move(r_m3, qt_zn)  # L1 -> L0B (ZN view)
                        # C8 / C9: Kc = 64 for both, so no phase, no matmul_acc, one pl.matmul each
                        # (DESIGN section 1.4: phase can only be answered by pl.store(phase=), and
                        # every accumulator here is drained by pl.move).
                        pl.set_validshape(dq_acc, [N1, D])
                        pl.matmul(dq_acc, l_m2, r_m2)  # dq_part[64,128] = dS[64,TS]@KK[TS,128]
                        pl.set_validshape(dkg_acc, [TS, D])
                        pl.matmul(dkg_acc, l_m3, r_m3)  # dkg[TS,128] = dS^T[TS,64]@Q[64,128]

                        pl.move(l_m22, ds_l12)  # L1 -> L0A
                        pl.move(r_m22, kkt_zn2)  # L1 -> L0B (ZN view)
                        pl.move(l_m32, dst_l12)  # L1 -> L0A
                        pl.move(r_m32, qt_zn2)  # L1 -> L0B (ZN view)
                        # C8 / C9: Kc = 64 for both, so no phase, no matmul_acc, one pl.matmul each
                        # (DESIGN section 1.4: phase can only be answered by pl.store(phase=), and
                        # every accumulator here is drained by pl.move).
                        pl.set_validshape(dq_acc2, [N1, D])
                        pl.matmul(dq_acc2, l_m22, r_m22)  # dq_part[64,128] = dS[64,TS]@KK[TS,128]
                        pl.set_validshape(dkg_acc2, [TS, D])
                        pl.matmul(dkg_acc2, l_m32, r_m32)  # dkg[TS,128] = dS^T[TS,64]@Q[64,128]

                        if c4slot == 0:
                            pl.system.set_cross_core(
                                pipe=pl.PipeType.MTE1,
                                event_id=EV_B3_RELEASE[0],
                                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
                            )
                        else:
                            pl.system.set_cross_core(
                                pipe=pl.PipeType.MTE1,
                                event_id=EV_B3_RELEASE[1],
                                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
                            )
                        c4slot = 1 - c4slot

                        # ---- B4 window: ONE event pair, four drains, Vec0 then Vec1 ----
                        # C10: B4 hand-off, both accumulators drained over FIX under one event pair.
                        pl.system.wait_cross_core(
                            pipe=pl.PipeType.FIX, event_id=EV_B4_RELEASE, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK
                        )
                        pl.move(dqp_vec, dq_acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                        pl.move(dkg_vec, dkg_acc, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)
                        pl.move(dqp_vec2, dq_acc2, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec1)
                        pl.move(dkg_vec2, dkg_acc2, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec1)
                        pl.system.set_cross_core(
                            pipe=pl.PipeType.FIX, event_id=EV_B4_READY, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK
                        )
                ctick = ctick + 1

        # Cube epilogue: one wait per B2 slot, so the Cube does not leave while the Vector still
        # owns s_vec (DESIGN section 4.2 "Cube 侧收尾").
        pl.system.wait_cross_core(
            pipe=pl.PipeType.FIX, event_id=EV_B2_RELEASE, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK
        )
        pl.system.wait_cross_core(
            pipe=pl.PipeType.FIX, event_id=EV_B4_RELEASE, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK
        )
        # TR epilogue: drain the final outstanding TR credit (prefire + last release),
        # so the Cube does not leave while the Vector still owns tr_vec.

    # =================================================================================
    # (3) VECTOR section -- Module 1 (s1_gather) unchanged, plus the B1 producer half and
    # the staged Module-2 observation store at the delay where s3 will read s_vec.
    # =================================================================================
    with pl.section_vector():
        # ---- resources: DESIGN section 3.1 addresses + section 2.3 mutex ids, verbatim ----
        krow_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
            addrs=VA_KROW,
            mutex_ids=MUTEX_KROW,
        )
        si_chunk_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, TS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
            addrs=VA_SI_CHUNK,
            mutex_ids=MUTEX_SI_CHUNK,
        )
        kk_ub_g = pl.make_tile_group(
            type=pl.TileType(shape=[TS, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
            addrs=VA_KK_UB,
            mutex_ids=MUTEX_KK_UB,
        )
        nz_stage_g = pl.make_tile_group(
            type=pl.TileType(shape=[TS, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addrs=VA_NZ_STAGE,
            mutex_ids=MUTEX_NZ_STAGE,
        )
        # scopy_g and gid_g are GONE with the staged observation stores they served.
        # ---- Module 3 row state.  Addresses are the frozen DESIGN section 3.1 map: every tile
        # BEFORE z_row is K-independent, so the three [1, K] rows stride from one base by kmax*4
        # and w/dw/red follow them.  s3_row OWNS all six (DESIGN section 4 state table). ----
        z_row_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, kmax], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=VA_ROWS_BASE,
            mutex_ids=MUTEX_Z_ROW,
        )
        y_row_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, kmax], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=VA_ROWS_BASE + kmax * 4,
            mutex_ids=MUTEX_Y_ROW,
        )
        pdi_row_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, kmax], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=VA_ROWS_BASE + 2 * kmax * 4,
            mutex_ids=MUTEX_PDI_ROW,
        )
        w_row_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, N1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=VA_ROWS_BASE + 3 * kmax * 4,
            mutex_ids=MUTEX_W_ROW,
        )
        dw_row_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, N1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=VA_ROWS_BASE + 3 * kmax * 4 + N1 * 4,
            mutex_ids=MUTEX_DW_ROW,
        )
        red_row_g = pl.make_tile_group(
            type=pl.TileType(shape=[1, N1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=VA_ROWS_BASE + 3 * kmax * 4 + 2 * N1 * 4,
            mutex_ids=MUTEX_RED_ROW,
        )
        ds_ub_g = pl.make_tile_group(
            type=pl.TileType(shape=[N1, TS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
            addrs=VA_DS_UB,
            mutex_ids=MUTEX_DS_UB,
        )
        si3_g = pl.make_tile_group(  # DELIVERED; aliases zero_grp, see _va_si3
            type=pl.TileType(shape=[1, TS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
            addrs=_va_si3(kmax),
            mutex_ids=MUTEX_SI3,
        )
        # ---- Module 4 producer-side tiles.  nz_stage_t / nz_stage_s are ALIASES of nz_stage --
        # same bytes, same mutex 7, different shape and orientation -- which is the alias_group
        # pattern klg_plan.tile_table already records, and the same device the B1 hand-off uses for
        # kk_l1.  Mutex 7 is what orders them against the B1 producer's insert in the same tick. ----
        nz_stage_t_g = pl.make_tile_group(  # alias of nz_stage, kk^T orientation
            type=pl.TileType(shape=[TS, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addrs=VA_NZ_STAGE,
            mutex_ids=MUTEX_NZ_STAGE,
        )
        nz_stage_s_g = pl.make_tile_group(  # alias of nz_stage, dS / dS^T orientation
            type=pl.TileType(shape=[N1, TS], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec, layout=pl.NZ),
            addrs=VA_NZ_STAGE,
            mutex_ids=MUTEX_NZ_STAGE,
        )
        # THIRD alias of nz_stage, added by the qt workaround: the ND landing tile for the
        # NON-transposing query load.  [N1, D] bf16 is 16384 B, exactly nz_stage's slot.  It costs
        # no UB and no vector mutex id because a transpose's SOURCE dies at the transpose, so the
        # landing tile and the NZ staging tile can be the same bytes -- only the transpose
        # DESTINATION must differ, and that is kkt_ub.  Mutex 7 orders every one of those reuses.
        q_ub_g = pl.make_tile_group(  # alias of nz_stage, query ND orientation
            type=pl.TileType(shape=[N1, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
            addrs=VA_KKT_UB,
            mutex_ids=MUTEX_KKT_UB,
        )
        # ---- Module 5 (s5_out) tiles.  s5 OWNS all four; every address and mutex id is the
        # frozen DESIGN section 3.1 / klg_plan value.  dq_row and dq_half_bf16 were reserved from
        # the start and were borrowed by the staged B4 consumer; this is their real owner. ----
        dq_row_g = pl.make_tile_group(
            type=pl.TileType(shape=[N1, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=VA_DQ_ROW,
            mutex_ids=MUTEX_DQ_ROW,
        )
        dq_half_g = pl.make_tile_group(
            type=pl.TileType(shape=[DQ_HALF_ROWS, D], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec),
            addrs=VA_DQ_HALF,
            mutex_ids=MUTEX_DQ_HALF,
        )
        si_out_g = pl.make_tile_group(  # s5's OWN index chunk, never s1's
            type=pl.TileType(shape=[1, TS], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
            addrs=_va_si_chunk_out(kmax),
            mutex_ids=MUTEX_SI_CHUNK_OUT,
        )
        dkrow_g = pl.make_tile_group(  # the LOCAL row the atomic store reads
            type=pl.TileType(shape=[1, D], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=_va_dkrow(kmax),
            mutex_ids=MUTEX_DKROW,
        )

        sub_id = pl.get_subblock_idx()  # DESIGN section 4: only AIV0 computes
        vslot = 0  # B1 slot parity, advanced per VALID task
        # Cross-core EVENTS run on BOTH AIVs; only arithmetic and stores are AIV0-only.
        # DESIGN.md:765 / :939 verbatim: "只有 sub_id == 0 的 AIV 执行算术与 store，sub_id == 1 的 AIV 执行完全相同的循环与完全相同的跨核事件序列（INTRA_BLOCK 要求 AIC 等待两个 AIV 子核都完成）"
        # My first version guarded the events too, on credit-arithmetic reasoning.  That was
        # wrong and the ladder measured it: rung 6 hangs because the AIC waits for both subblocks
        # and AIV1 never arrives.  The pre-fire is therefore UNGUARDED.
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=EV_B2_RELEASE, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        # The Vector is also the B4 CONSUMER, so it pre-fires that boundary's single credit.
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=EV_B4_RELEASE, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        # The Vector is the TR CONSUMER (it casts kkt), so it pre-fires TR's single credit too.
        v3slot = 0  # B3 slot parity, per VALID pass-1 task
        t2_last = pl.max(t2 - 1, 0)  # highest legal key row
        tick = 0
        # `mid` marks pass 0's LAST chunk, which is where DESIGN section 1.3 puts the row
        # finalize (V11..V13).  `last` already marks pass 1's last chunk (V21).  nce is not in the
        # struct, so the flag is set where nce is in scope rather than derived at the read site.
        ctx = pl.struct_array(
            4,
            "VecCtx",
            t=0,
            pass_=0,
            chunk=0,
            real_k=0,
            vts=0,
            ks=0,
            avail=0,
            lim=0,
            g_safe=0,
            is_valid=0,
            first=0,
            last=0,
            mid=0,
            cache=0,
            gen=0,
            b1=0,
            b3=0,
        )

        # TOKEN SPLIT: same pair walk as the Cube, and each sub-core takes its OWN member.
        # `worker`/`workers` is the spelling the dk zero prologue (section 1.1) already uses; this
        # promotes it from the prologue to the whole Vector Section.  No stage is gated by sub_id
        # any more -- `sub_id` only picks this sub-core's token and its stream's L1 handles.
        worker = pl.get_block_idx()
        workers = pl.get_block_num() * pl.get_subblock_num()
        n_pairs = pl.max((t1 - 2 * core_id + 2 * num_cores - 1) // (2 * num_cores), 1)
        for pk in pl.range(0, n_pairs):
            base = 2 * core_id + 2 * num_cores * pk
            my_t = base + sub_id
            other_t = base + 1 - sub_id
            # ---- own token (this sub-core's): DESIGN 4.1 decoder, COPIED verbatim ----
            # ===== DESIGN section 4.1 stateless row decoder (identical code in every section) ====
            t = my_t
            b = -1
            for bb in pl.range(0, n_batch):  # no early break: q_len == 0 batches exist
                lo = pl.getval(cu_seqlens_q, bb)
                hi = pl.getval(cu_seqlens_q, bb + 1)
                if lo <= t:
                    if t < hi:
                        b = bb
            qs = 0
            q_len = 0
            ks = 0
            kv_len = 0
            res = 0
            if b >= 0:
                qs = pl.getval(cu_seqlens_q, b)
                q_len = pl.getval(cu_seqlens_q, b + 1) - qs
                ks = pl.getval(cu_seqlens_k, b)
                kv_len = pl.getval(cu_seqlens_k, b + 1) - ks
                res = pl.getval(cmp_residual_k, b)
            pos = t - qs
            avail = kv_len
            real_k = pl.min(kv_len, kmax)
            if MaskMode == 3:  # pyrefly: ignore [unknown-name]
                avail = pl.min(kv_len + 1, t2 - ks)
                num = kv_len * cmp_ratio + res - q_len + pos + 1
                vis = 0
                if num > 0:  # NEVER divide a non-positive numerator
                    vis = num // cmp_ratio
                real_k = pl.min(pl.max(pl.min(vis, avail), 0), kmax)
            lim = avail  # validity_mode "avail"
            if b < 0:
                real_k = 0  # row outside every batch -> all zeros
            nce = pl.max((real_k + TS - 1) // TS, 1)  # >= 1: zero iterations break rendezvous
            g_safe = pl.min(pl.max(ks, 0), t2_last)  # always a legal DMA address
            # ---- the OTHER stream, for nce_p only: DESIGN 4.1 decoder, COPIED verbatim ----
            # ===== DESIGN section 4.1 stateless row decoder (identical code in every section) ====
            t_O = other_t
            b_O = -1
            for bb_O in pl.range(0, n_batch):  # no early break: q_len_O == 0 batches exist
                lo_O = pl.getval(cu_seqlens_q, bb_O)
                hi_O = pl.getval(cu_seqlens_q, bb_O + 1)
                if lo_O <= t_O:
                    if t_O < hi_O:
                        b_O = bb_O
            qs_O = 0
            q_len_O = 0
            ks_O = 0
            kv_len_O = 0
            res_O = 0
            if b_O >= 0:
                qs_O = pl.getval(cu_seqlens_q, b_O)
                q_len_O = pl.getval(cu_seqlens_q, b_O + 1) - qs_O
                ks_O = pl.getval(cu_seqlens_k, b_O)
                kv_len_O = pl.getval(cu_seqlens_k, b_O + 1) - ks_O
                res_O = pl.getval(cmp_residual_k, b_O)
            pos_O = t_O - qs_O
            avail_O = kv_len_O
            real_k_O = pl.min(kv_len_O, kmax)
            if MaskMode == 3:  # pyrefly: ignore [unknown-name]
                avail_O = pl.min(kv_len_O + 1, t2 - ks_O)
                num_O = kv_len_O * cmp_ratio + res_O - q_len_O + pos_O + 1
                vis_O = 0
                if num_O > 0:  # NEVER divide a non-positive numerator
                    vis_O = num_O // cmp_ratio
                real_k_O = pl.min(pl.max(pl.min(vis_O, avail_O), 0), kmax)
            lim_O = avail_O  # validity_mode "avail_O"
            if b_O < 0:
                real_k_O = 0  # row outside every batch -> all zeros
            nce_O = pl.max((real_k_O + TS - 1) // TS, 1)  # >= 1: zero iterations break rendezvous
            g_safe_O = pl.min(pl.max(ks_O, 0), t2_last)  # always a legal DMA address
            # ---- COMMON tick count, byte-for-byte the same arithmetic as the Cube's.  See the
            # Cube's note: this is what keeps every event and cursor under one gate that both
            # sub-cores evaluate identically.
            nce = pl.max(pl.max(nce, nce_O), 1)
            drain = 0
            if base + 2 * num_cores >= t1:
                drain = MAX_DELAY
            # =====================================================================================
            for tk in pl.range(0, 2 * nce + drain):
                # ---- TOKEN SPLIT: s1 + B1 + s3 + B3 are duplicated under sub_id because an L1
                # group handle cannot be selected by a runtime value (DESIGN 2.3).  The bodies
                # are COPIES with the control's five `if sub_id == 0:` gates REMOVED -- both
                # sub-cores compute now -- and only the five per-stream L1 handles differ.
                # Every event and every .next() is inside both copies, so each fires once per
                # sub-core exactly as before and INTRA_BLOCK still sees both arrive. ----
                if sub_id == 0:
                    # section 2.3 rule 3: the kk_ub cursor advances UNCONDITIONALLY every tick, drain
                    # ticks included (s3 reads it with previous() from Module 3 on).
                    kk_ub_cur = kk_ub_g.next()
                    cur = ctx[tick % CTX_DEPTH]
                    cur.is_valid = 0
                    if tk < 2 * nce:
                        chunk = tk % nce
                        cur.t = t
                        cur.pass_ = tk // nce
                        cur.chunk = chunk
                        cur.gen = pk % 2
                        cur.cache = 0
                        cur.b1 = tk % 2
                        cur.b3 = 2 + chunk % 2
                        if nce <= 5:
                            cur.cache = 1
                            cur.b1 = chunk
                            cur.b3 = chunk
                        cur.real_k = real_k
                        cur.vts = pl.min(TS, pl.max(real_k - chunk * TS, 0))
                        cur.ks = ks
                        cur.avail = avail
                        cur.lim = lim
                        cur.g_safe = g_safe
                        cur.first = 0
                        if tk == 0:
                            cur.first = 1
                        cur.last = 0
                        if tk == 2 * nce - 1:
                            cur.last = 1
                        cur.mid = 0
                        if tk == nce - 1:
                            cur.mid = 1
                        cur.is_valid = 1

                        # ---------- Module 1, stage s1_gather (delay 0, DESIGN section 1.2) ----------
                        if cur.cache == 0 or cur.pass_ == 0:
                            c_t = cur.t
                            c_real_k = cur.real_k
                            c_ks = cur.ks
                            c_avail = cur.avail
                            c_lim = cur.lim
                            c_gsafe = cur.g_safe
                            c_base = cur.chunk * TS
                            c_span = pl.max(cur.vts, 1)  # deviation D3: set_validshape >= 1

                            # V1: the chunk's index row, GM -> UB
                            si_c = si_chunk_g.next()
                            # NARROWED DMA, adjudicated harmless: a pl.load narrowed by
                            # set_validshape leaves the remainder of the last 32 B destination block
                            # UNDEFINED (measured -- see the p load in the s3 block).  Here the tail is
                            # read by si_c[0, s] for every s < TS, but every use is gated by
                            # `if col < c_real_k`, so an undefined value can only reach `vsel = 0` and
                            # `g_use = g_safe`.  Module 1 is bit-exact on device, which confirms it.
                            pl.set_validshape(si_c, [1, c_span])
                            pl.load(si_c, sparse_indices, [pl.min(c_t, t1 - 1), 0, c_base], order=[0, 2])

                            # V2 + V3 + V4: scalar index resolution, one key row per column
                            if cur.vts < TS:
                                klg_zero_key_tail_vf(kk_ub_cur, cur.vts)
                            for s in pl.range(0, cur.vts):
                                si = si_c[0, s]  # getval, S pipe
                                col = c_base + s
                                vsel = 0
                                if si >= 0:
                                    if si < c_lim:
                                        vsel = 1  # v_s (SPEC step 4)
                                g_use = c_gsafe
                                if col < c_real_k:
                                    g_use = pl.min(
                                        pl.max(c_ks + pl.min(pl.max(si, 0), pl.max(c_avail - 1, 0)), 0), t2_last
                                    )
                                else:
                                    vsel = 0  # s >= real_k: zero row, safe address
                                v_lanes = vsel * D  # 128 b16 lanes, or none
                                if vsel == 1:
                                    kk_row = kk_ub_cur[s : s + 1, 0:D]
                                    pl.load(kk_row, key, [g_use, 0, 0], order=[0, 2])
                                else:
                                    klg_zero_key_row_vf(kk_ub_cur, s * D)

                            # V5: ND -> NZ staging of the finished block (Module 2 adds the pl.insert
                            # into kk_l1_g plus the B1 event pair -- deviation D4).
                            nz = nz_stage_g.current()
                            pl.set_validshape(kk_ub_cur, [TS, D])
                            pl.set_validshape(nz, [TS, D])
                            pl.move(nz, kk_ub_cur)

                            # The two staged observation stores (kk_gathered, g_ids) are GONE: the
                            # delivered kernel has four outputs.  Module 1 is therefore no longer
                            # directly observable -- see the module docstring's localisation procedure.

                        # ---- Module 2: B1 producer half.  EVENTS run on BOTH AIVs, pl.insert only
                        # on AIV0.  DESIGN.md:765 / :939 verbatim: "只有 sub_id == 0 的 AIV 执行算术与 store，sub_id == 1 的 AIV 执行完全相同的循环与完全相同的跨核事件序列（INTRA_BLOCK 要求 AIC 等待两个 AIV 子核都完成）"
                        # The ladder measured both wrong spellings: with NOTHING guarded AIV1 also
                        # stores and corrupts the result (rung 5: 17/32768, 11730x the fp32 bound);
                        # with the EVENTS guarded AIV1 never arrives and the AIC waits forever
                        # (rung 6: aicore timeout).  Only the split below is correct. ----
                        nz_b1 = nz_stage_g.current()  # current() does not advance a cursor
                        kk_nz = kk_l1_nz_g[cur.gen * 5 + cur.b1]
                        if vslot == 0:
                            pl.system.wait_cross_core(
                                pipe=pl.PipeType.MTE3,
                                event_id=EV_B1_RELEASE[0],
                                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
                            )
                        else:
                            pl.system.wait_cross_core(
                                pipe=pl.PipeType.MTE3,
                                event_id=EV_B1_RELEASE[1],
                                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
                            )
                        if cur.cache == 0 or cur.pass_ == 0:
                            pl.insert(kk_nz, nz_b1, [0, 0])  # normal NZ cache write
                        if vslot == 0:
                            pl.system.set_cross_core(
                                pipe=pl.PipeType.MTE3,
                                event_id=EV_B1_READY[0],
                                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
                            )
                        else:
                            pl.system.set_cross_core(
                                pipe=pl.PipeType.MTE3,
                                event_id=EV_B1_READY[1],
                                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
                            )
                        vslot = 1 - vslot

                    # ---- Module 3, stage s3_row (delay 1, DESIGN section 1.3), plus the staged
                    # Module-2 observation store that used to be the whole of this block.
                    #
                    # s3 is the B2 CONSUMER: every read of s_vec (V8, V14, V15) sits between the
                    # wait on B2.READY and the set of B2.RELEASE, and the very first touch of s_vec
                    # after the wait is a VECTOR instruction (KB section 17, DESIGN :727 / :909).
                    # Work that does not touch s_vec -- the row prologue and this chunk's index DMA --
                    # is issued BEFORE the wait so it overlaps the Cube's matmul.
                    #
                    # The score_s store stays FULL WIDTH [N1, TS]: columns s >= real_k inside the chunk
                    # carry key rows Module 1 zeroed exactly, so S is exactly 0 there and the golden
                    # holds 0 too.  That avoids narrowing a CROSS-CORE tile with set_validshape (which
                    # would collide with the Cube's next Acc->Vec move) and avoids a runtime subview.
                    # nce*TS <= K for both K choices, so a full-width store never runs past score_s --
                    # and that same inequality is what makes the untrimmed si / ds windows below legal.
                    c3 = ctx[(tick + CTX_DEPTH - 1) % CTX_DEPTH]
                    if c3.is_valid == 1:
                        # ---- STRUCTURAL, not semantic.  `c3.chunk >= 0` is ALWAYS true (chunk is
                        # tk % nce).  It is here because the probe measured that an always-true
                        # conditional enclosing a cross-core group decides whether the kernel
                        # deadlocks: probe_klg_delay1_vcguard.py sha f2f483dd3fae000e, two arms whose
                        # ONLY difference is this nesting -- identical event counts, ids, order, loop
                        # bounds and get_buf/rls_buf placement in the generated C++ -- where the bare
                        # arm hit aicore timeout 507014 and the nested arm passed 5/5 at 32768/32768.
                        # Applied ONLY here, matching the construct that was measured: in that probe's
                        # passing arm the B1 producer half and the whole cube side stayed BARE, so
                        # nesting them too would be speculation, not evidence.  The whole s3 body is
                        # inside it because s3 IS that cross-core consumer now -- the guard must enclose
                        # the wait/set pair, and putting MORE straight-line code inside an always-true
                        # branch cannot change which branch runs.
                        # REMOVE THIS the moment the framework behaviour is explained or fixed.
                        if c3.chunk >= 0:
                            s_vec_v = s_vec_g.current()
                            z_row = z_row_g.current()
                            y_row = y_row_g.current()
                            pdi_row = pdi_row_g.current()
                            w_row = w_row_g.current()
                            dw_row = dw_row_g.current()
                            red_row = red_row_g.current()
                            ds_ub = ds_ub_g.current()
                            si3 = si3_g.current()
                            v_t = c3.t
                            v_rk = c3.real_k
                            v_vts = c3.vts
                            v_lim = c3.lim
                            v_off = c3.chunk * TS

                            # ---- V7 row prologue + V9 p load + the staged index DMA.  None of these
                            # touch s_vec, so they are issued before the B2 wait.  AIV0 only: the
                            # Acc->Vec drain is SingleModeVec0, so AIV1's copy of every Vec tile is
                            # uninitialised and must not be written (rung 5: 17/32768). ----
                            if c3.first == 1:
                                pl.load(w_row, weights, [pl.min(v_t, t1 - 1), 0])
                                klg_rowinit_vf(z_row, y_row, pdi_row, dw_row, red_row, kmax)
                                # V9, hoisted to the token prologue -- DEVIATION D-M3-1.  DESIGN
                                # section 1.3 loads p one chunk at a time into a segment of pdi_row;
                                # a pl.load cannot target a tile OFFSET, so the per-chunk form needs
                                # either a second [1, TS] tile plus a VF copy or a DMA subview, and
                                # neither is in the frozen plan.  One [1, real_k] DMA per token is
                                # strictly fewer transfers and reads exactly the region the golden
                                # defines; the tail [real_k, K) keeps V7's zeros, which is what makes
                                # the staged di_row exact there.
                                if v_rk > 0:
                                    # MEASURED, 2026-09-25: a pl.load narrowed by set_validshape writes
                                    # whole 32 B DESTINATION BLOCKS, and the remainder of the last block
                                    # holds UNDEFINED data -- not the tensor's values.  With a width of
                                    # real_k that put garbage into pdi_row[real_k : roundup(real_k, 8)),
                                    # which is the whole of the di_row/tail failure: the deficit was
                                    # roundup(real_k, 8) - real_k summed over rows, EXACT on 11 of 11
                                    # cases, while p itself is zero in that range on all 11 -- so the
                                    # spilled content was never p's.  Loading a whole number of
                                    # 64-element registers leaves no partial block to be undefined, and
                                    # klg_pacc_vf then forces [real_k, v_pspan) to an exact 0.
                                    v_pspan = pl.max(((v_rk + TS - 1) // TS) * TS, TS)
                                    pl.set_validshape(pdi_row, [1, v_pspan])
                                    pl.load(pdi_row, attn_softmax_l1_norm, [pl.min(v_t, t1 - 1), 0, 0], order=[0, 2])
                                    pl.set_validshape(pdi_row, [1, kmax])
                            if c3.pass_ == 0:
                                # Full TS width, no set_validshape: v_off + TS <= nce*TS <= K because
                                # both K choices are multiples of TS, so the read is in bounds, and
                                # lanes at s >= real_k are masked out of every reduction downstream.
                                pl.load(si3, sparse_indices, [pl.min(v_t, t1 - 1), 0, v_off], order=[0, 2])

                            # ---- B2 window.  Events on BOTH AIVs (DESIGN :765 / :939 verbatim: "只有
                            # sub_id == 0 的 AIV 执行算术与 store，sub_id == 1 的 AIV 执行完全相同的循环与完全相同的跨核事件序列
                            # （INTRA_BLOCK 要求 AIC 等待两个 AIV 子核都完成）"), arithmetic and stores on AIV0 only.
                            # Guarding the events instead was measured as rung 6's aicore timeout. ----
                            pl.system.wait_cross_core(
                                pipe=pl.PipeType.V, event_id=EV_B2_READY, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK
                            )
                            # Staged Module-2 observation.  KB section 17: the consumer's FIRST touch
                            # of a cross-core buffer must be a VECTOR instruction, and a GM forward
                            # must store a LOCAL tile, never the cross-core one.  Storing straight off
                            # s_vec is what the ladder measured as 5/32768 at rung 8.
                            # score_s's staged store and its section-17 landing copy are GONE.
                            # SECTION 17 STILL HOLDS: the first touch of the cross-core s_vec in each
                            # branch below is a VF (klg_z_vf on pass 0, klg_dw_vf on pass 1), not a DMA,
                            # and no pl.store in this file reads a cross-core tile -- selfcheck check 4
                            # is the standing gate on that.
                            if c3.pass_ == 0:
                                klg_z_vf(s_vec_v, w_row, z_row, v_off, v_vts)  # V8
                                klg_pacc_vf(pdi_row, si3, red_row, v_off, v_vts, v_lim)  # V10
                                if c3.mid == 1:
                                    # ---- row finalize, immediately after pass 0's last chunk ----
                                    klg_softmax_vf(z_row, y_row, v_rk)  # V11
                                    pl.set_validshape(y_row, [1, kmax])  # V12
                                    if v_t < t1:  # TOKEN SPLIT: odd-t1 bounds guard (see section 13)
                                        pl.store(softmax_out, y_row, [v_t, 0, 0], order=[0, 2])
                                    klg_di_vf(y_row, pdi_row, red_row, v_rk)  # V13
                            else:
                                klg_dwds_vf(s_vec_v, pdi_row, dw_row, w_row, ds_ub, v_off, v_vts)
                                if c3.last == 1:
                                    if v_t < t1:  # TOKEN SPLIT: odd-t1 bounds guard (see section 13)
                                        pl.store(dw_out, dw_row, [v_t, 0])  # V21
                            pl.system.set_cross_core(
                                pipe=pl.PipeType.V, event_id=EV_B2_RELEASE, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK
                            )

                            # ---- Module 4's PRODUCER half, V16..V20 (DESIGN section 1.3), pass 1 only.
                            # Gate is `is_valid && pass == 1`, matching s4's gate exactly (section 2.3
                            # rule 1).  The three .next() calls sit OUTSIDE `if sub_id == 0:` so both AIVs
                            # advance their cursors identically -- the same split the B1 producer uses and
                            # the ladder measured -- while every transpose / move / insert is AIV0-only and
                            # the event pair runs on both AIVs. ----
                            if c3.pass_ == 1:
                                ds_l1_cur = ds_l1_g.next()  # V17
                                kkt_l1_cur = kkt_l1_nz_g[c3.gen * 5 + c3.b3]  # V20 (NZ producer view)
                                # qt_l1's cursor advances once per TOKEN, not once per tick -- the same
                                # gate the Cube's consumer uses.  It sits OUTSIDE `if sub_id == 0:` with
                                # the other three so both AIVs advance identically, and the handle comes
                                # from an UNCONDITIONAL .current() (DESIGN :775 scope rule).
                                if c3.chunk == 0:
                                    qt_l1_nz_g.next()
                                qt_l1_cur = qt_l1_nz_g.current()
                                if v3slot == 0:
                                    pl.system.wait_cross_core(
                                        pipe=pl.PipeType.MTE3,
                                        event_id=EV_B3_RELEASE[0],
                                        sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
                                    )
                                else:
                                    pl.system.wait_cross_core(
                                        pipe=pl.PipeType.MTE3,
                                        event_id=EV_B3_RELEASE[1],
                                        sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
                                    )
                                nz_s = nz_stage_s_g.current()
                                nz_t = nz_stage_t_g.current()
                                # ---- V16a..V16d: the qt WORKAROUND, once per token.  This is the
                                # kkt pattern (V19/V20) with `query` in place of `kk_ub`, and it
                                # reuses V19/V20's own scratch: q_nd aliases nz_stage, the
                                # transpose lands in kkt_ub, the NZ staging is nz_t.  All four
                                # reuses are ordered by mutex 7 / mutex 6 against V19/V20 below,
                                # which is the same device the B1 producer already relies on.
                                # The load is ASCENDING: GM -> UB, no transpose in the DMA.
                                # The transpose is [64,128] -> [128,64] bf16, which is EXACTLY the
                                # shape BP-2a measured EXACT on this box.
                                if c3.chunk == 0:
                                    q_nd = q_ub_g.current()
                                    # `c3.t` directly, NOT `v_t`: v_t is assigned inside the
                                    # B2-consumer branch, a SIBLING of this one, and DESIGN :775
                                    # records that such a variable leaves scope in DSL codegen.
                                    pl.load(q_nd, query, [pl.min(c3.t, t1 - 1), 0, 0], order=[1, 2])  # V16a
                                    pl.move(nz_t, q_nd)  # normal Q ND->NZ
                                    pl.insert(qt_l1_cur, nz_t, [0, 0])  # V16d
                                pl.move(nz_s, ds_ub)  # V17 ND -> NZ
                                pl.insert(ds_l1_cur, nz_s, [0, 0])
                                # ---- V19.  `.previous()` is a one-TICK lookback, not a one-WRITE
                                # lookback: it lowers to (cursor + n_slots - 1) with NO write-back, so at
                                # tick T it names the slot `next()` returned at tick T-1 (BP-9 ruling,
                                # architect 2026-09-25, measured on device -- three arms, all PASS).
                                # Correctness needs exactly two things, neither of them a slot count:
                                #   (i) the reader's delay is exactly 1 -- s3 is;
                                #  (ii) the reader's gate is the producer's gate shifted one tick --
                                #       `kk_ub_g.next()` runs UNCONDITIONALLY every tick (it is at the
                                #       top of this tick body, outside both `if tk < 2*nce` and
                                #       `if sub_id == 0`), and this read is under `if c3.is_valid`, which
                                #       is task T-1's validity.  So a read at tick T implies a write at
                                #       tick T-1, into the very slot previous() selects.
                                # TWO HARD REQUIREMENTS from section 2.3 rule 3, each with a negative
                                # control in event_pairing_sim.py:
                                #   (1) this call must stay INSIDE `if c3.is_valid` -- it must NOT be
                                #       hoisted to the top of the tick body "for symmetry with next()";
                                #   (2) the step that reads it must not leave the delay-1 stage.
                                # Both hold here.  Do not "tidy" either one.
                                # ---- V19 via Cube transpose (milestone 3).  The Cube drained KK^T to
                                # tr_vec over the TR window; cast fp32 -> bf16 here, replacing the per-tick
                                # pl.transpose.  tr_vec is ONE slot, but SingleModeVec0/1 put stream A in
                                # AIV0's private UB and stream B in AIV1's, so each sub_id reads its own
                                # token's KK^T from the same handle.  The event pair sits in BOTH the
                                # `if sub_id==0` (stream A) and `else` (stream B) branches, so BOTH AIVs
                                # fire it -- the INTRA_BLOCK rendezvous the Cube's FIX set/wait needs.
                                # S1 advances the two KK slots every tick. S3 consumes
                                # only the previous valid task, exactly one tick later.
                                if c3.cache == 0:
                                    kk_previous = kk_ub_g.previous()
                                    pl.move(nz_t, kk_previous)  # fallback normal KK
                                    pl.insert(kkt_l1_cur, nz_t, [0, 0])
                                if v3slot == 0:
                                    pl.system.set_cross_core(
                                        pipe=pl.PipeType.MTE3,
                                        event_id=EV_B3_READY[0],
                                        sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
                                    )
                                else:
                                    pl.system.set_cross_core(
                                        pipe=pl.PipeType.MTE3,
                                        event_id=EV_B3_READY[1],
                                        sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
                                    )
                                v3slot = 1 - v3slot

                else:
                    # section 2.3 rule 3: the kk_ub cursor advances UNCONDITIONALLY every tick, drain
                    # ticks included (s3 reads it with previous() from Module 3 on).
                    kk_ub_cur = kk_ub_g.next()
                    cur = ctx[tick % CTX_DEPTH]
                    cur.is_valid = 0
                    if tk < 2 * nce:
                        chunk = tk % nce
                        cur.t = t
                        cur.pass_ = tk // nce
                        cur.chunk = chunk
                        cur.gen = pk % 2
                        cur.cache = 0
                        cur.b1 = tk % 2
                        cur.b3 = 2 + chunk % 2
                        if nce <= 5:
                            cur.cache = 1
                            cur.b1 = chunk
                            cur.b3 = chunk
                        cur.real_k = real_k
                        cur.vts = pl.min(TS, pl.max(real_k - chunk * TS, 0))
                        cur.ks = ks
                        cur.avail = avail
                        cur.lim = lim
                        cur.g_safe = g_safe
                        cur.first = 0
                        if tk == 0:
                            cur.first = 1
                        cur.last = 0
                        if tk == 2 * nce - 1:
                            cur.last = 1
                        cur.mid = 0
                        if tk == nce - 1:
                            cur.mid = 1
                        cur.is_valid = 1

                        # ---------- Module 1, stage s1_gather (delay 0, DESIGN section 1.2) ----------
                        if cur.cache == 0 or cur.pass_ == 0:
                            c_t = cur.t
                            c_real_k = cur.real_k
                            c_ks = cur.ks
                            c_avail = cur.avail
                            c_lim = cur.lim
                            c_gsafe = cur.g_safe
                            c_base = cur.chunk * TS
                            c_span = pl.max(cur.vts, 1)  # deviation D3: set_validshape >= 1

                            # V1: the chunk's index row, GM -> UB
                            si_c = si_chunk_g.next()
                            # NARROWED DMA, adjudicated harmless: a pl.load narrowed by
                            # set_validshape leaves the remainder of the last 32 B destination block
                            # UNDEFINED (measured -- see the p load in the s3 block).  Here the tail is
                            # read by si_c[0, s] for every s < TS, but every use is gated by
                            # `if col < c_real_k`, so an undefined value can only reach `vsel = 0` and
                            # `g_use = g_safe`.  Module 1 is bit-exact on device, which confirms it.
                            pl.set_validshape(si_c, [1, c_span])
                            pl.load(si_c, sparse_indices, [pl.min(c_t, t1 - 1), 0, c_base], order=[0, 2])

                            # V2 + V3 + V4: scalar index resolution, one key row per column
                            if cur.vts < TS:
                                klg_zero_key_tail_vf(kk_ub_cur, cur.vts)
                            for s in pl.range(0, cur.vts):
                                si = si_c[0, s]  # getval, S pipe
                                col = c_base + s
                                vsel = 0
                                if si >= 0:
                                    if si < c_lim:
                                        vsel = 1  # v_s (SPEC step 4)
                                g_use = c_gsafe
                                if col < c_real_k:
                                    g_use = pl.min(
                                        pl.max(c_ks + pl.min(pl.max(si, 0), pl.max(c_avail - 1, 0)), 0), t2_last
                                    )
                                else:
                                    vsel = 0  # s >= real_k: zero row, safe address
                                v_lanes = vsel * D  # 128 b16 lanes, or none
                                if vsel == 1:
                                    kk_row = kk_ub_cur[s : s + 1, 0:D]
                                    pl.load(kk_row, key, [g_use, 0, 0], order=[0, 2])
                                else:
                                    klg_zero_key_row_vf(kk_ub_cur, s * D)

                            # V5: ND -> NZ staging of the finished block (Module 2 adds the pl.insert
                            # into kk_l1_g plus the B1 event pair -- deviation D4).
                            nz = nz_stage_g.current()
                            pl.set_validshape(kk_ub_cur, [TS, D])
                            pl.set_validshape(nz, [TS, D])
                            pl.move(nz, kk_ub_cur)

                            # The two staged observation stores (kk_gathered, g_ids) are GONE: the
                            # delivered kernel has four outputs.  Module 1 is therefore no longer
                            # directly observable -- see the module docstring's localisation procedure.

                        # ---- Module 2: B1 producer half.  EVENTS run on BOTH AIVs, pl.insert only
                        # on AIV0.  DESIGN.md:765 / :939 verbatim: "只有 sub_id == 0 的 AIV 执行算术与 store，sub_id == 1 的 AIV 执行完全相同的循环与完全相同的跨核事件序列（INTRA_BLOCK 要求 AIC 等待两个 AIV 子核都完成）"
                        # The ladder measured both wrong spellings: with NOTHING guarded AIV1 also
                        # stores and corrupts the result (rung 5: 17/32768, 11730x the fp32 bound);
                        # with the EVENTS guarded AIV1 never arrives and the AIC waits forever
                        # (rung 6: aicore timeout).  Only the split below is correct. ----
                        nz_b1 = nz_stage_g.current()  # current() does not advance a cursor
                        kk_nz = kk_l1_nz_g2[cur.gen * 5 + cur.b1]
                        if vslot == 0:
                            pl.system.wait_cross_core(
                                pipe=pl.PipeType.MTE3,
                                event_id=EV_B1_RELEASE[0],
                                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
                            )
                        else:
                            pl.system.wait_cross_core(
                                pipe=pl.PipeType.MTE3,
                                event_id=EV_B1_RELEASE[1],
                                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
                            )
                        if cur.cache == 0 or cur.pass_ == 0:
                            pl.insert(kk_nz, nz_b1, [0, 0])  # normal NZ cache write
                        if vslot == 0:
                            pl.system.set_cross_core(
                                pipe=pl.PipeType.MTE3,
                                event_id=EV_B1_READY[0],
                                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
                            )
                        else:
                            pl.system.set_cross_core(
                                pipe=pl.PipeType.MTE3,
                                event_id=EV_B1_READY[1],
                                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
                            )
                        vslot = 1 - vslot

                    # ---- Module 3, stage s3_row (delay 1, DESIGN section 1.3), plus the staged
                    # Module-2 observation store that used to be the whole of this block.
                    #
                    # s3 is the B2 CONSUMER: every read of s_vec (V8, V14, V15) sits between the
                    # wait on B2.READY and the set of B2.RELEASE, and the very first touch of s_vec
                    # after the wait is a VECTOR instruction (KB section 17, DESIGN :727 / :909).
                    # Work that does not touch s_vec -- the row prologue and this chunk's index DMA --
                    # is issued BEFORE the wait so it overlaps the Cube's matmul.
                    #
                    # The score_s store stays FULL WIDTH [N1, TS]: columns s >= real_k inside the chunk
                    # carry key rows Module 1 zeroed exactly, so S is exactly 0 there and the golden
                    # holds 0 too.  That avoids narrowing a CROSS-CORE tile with set_validshape (which
                    # would collide with the Cube's next Acc->Vec move) and avoids a runtime subview.
                    # nce*TS <= K for both K choices, so a full-width store never runs past score_s --
                    # and that same inequality is what makes the untrimmed si / ds windows below legal.
                    c3 = ctx[(tick + CTX_DEPTH - 1) % CTX_DEPTH]
                    if c3.is_valid == 1:
                        # ---- STRUCTURAL, not semantic.  `c3.chunk >= 0` is ALWAYS true (chunk is
                        # tk % nce).  It is here because the probe measured that an always-true
                        # conditional enclosing a cross-core group decides whether the kernel
                        # deadlocks: probe_klg_delay1_vcguard.py sha f2f483dd3fae000e, two arms whose
                        # ONLY difference is this nesting -- identical event counts, ids, order, loop
                        # bounds and get_buf/rls_buf placement in the generated C++ -- where the bare
                        # arm hit aicore timeout 507014 and the nested arm passed 5/5 at 32768/32768.
                        # Applied ONLY here, matching the construct that was measured: in that probe's
                        # passing arm the B1 producer half and the whole cube side stayed BARE, so
                        # nesting them too would be speculation, not evidence.  The whole s3 body is
                        # inside it because s3 IS that cross-core consumer now -- the guard must enclose
                        # the wait/set pair, and putting MORE straight-line code inside an always-true
                        # branch cannot change which branch runs.
                        # REMOVE THIS the moment the framework behaviour is explained or fixed.
                        if c3.chunk >= 0:
                            s_vec_v = s_vec_g.current()
                            z_row = z_row_g.current()
                            y_row = y_row_g.current()
                            pdi_row = pdi_row_g.current()
                            w_row = w_row_g.current()
                            dw_row = dw_row_g.current()
                            red_row = red_row_g.current()
                            ds_ub = ds_ub_g.current()
                            si3 = si3_g.current()
                            v_t = c3.t
                            v_rk = c3.real_k
                            v_vts = c3.vts
                            v_lim = c3.lim
                            v_off = c3.chunk * TS

                            # ---- V7 row prologue + V9 p load + the staged index DMA.  None of these
                            # touch s_vec, so they are issued before the B2 wait.  AIV0 only: the
                            # Acc->Vec drain is SingleModeVec0, so AIV1's copy of every Vec tile is
                            # uninitialised and must not be written (rung 5: 17/32768). ----
                            if c3.first == 1:
                                pl.load(w_row, weights, [pl.min(v_t, t1 - 1), 0])
                                klg_rowinit_vf(z_row, y_row, pdi_row, dw_row, red_row, kmax)
                                # V9, hoisted to the token prologue -- DEVIATION D-M3-1.  DESIGN
                                # section 1.3 loads p one chunk at a time into a segment of pdi_row;
                                # a pl.load cannot target a tile OFFSET, so the per-chunk form needs
                                # either a second [1, TS] tile plus a VF copy or a DMA subview, and
                                # neither is in the frozen plan.  One [1, real_k] DMA per token is
                                # strictly fewer transfers and reads exactly the region the golden
                                # defines; the tail [real_k, K) keeps V7's zeros, which is what makes
                                # the staged di_row exact there.
                                if v_rk > 0:
                                    # MEASURED, 2026-09-25: a pl.load narrowed by set_validshape writes
                                    # whole 32 B DESTINATION BLOCKS, and the remainder of the last block
                                    # holds UNDEFINED data -- not the tensor's values.  With a width of
                                    # real_k that put garbage into pdi_row[real_k : roundup(real_k, 8)),
                                    # which is the whole of the di_row/tail failure: the deficit was
                                    # roundup(real_k, 8) - real_k summed over rows, EXACT on 11 of 11
                                    # cases, while p itself is zero in that range on all 11 -- so the
                                    # spilled content was never p's.  Loading a whole number of
                                    # 64-element registers leaves no partial block to be undefined, and
                                    # klg_pacc_vf then forces [real_k, v_pspan) to an exact 0.
                                    v_pspan = pl.max(((v_rk + TS - 1) // TS) * TS, TS)
                                    pl.set_validshape(pdi_row, [1, v_pspan])
                                    pl.load(pdi_row, attn_softmax_l1_norm, [pl.min(v_t, t1 - 1), 0, 0], order=[0, 2])
                                    pl.set_validshape(pdi_row, [1, kmax])
                            if c3.pass_ == 0:
                                # Full TS width, no set_validshape: v_off + TS <= nce*TS <= K because
                                # both K choices are multiples of TS, so the read is in bounds, and
                                # lanes at s >= real_k are masked out of every reduction downstream.
                                pl.load(si3, sparse_indices, [pl.min(v_t, t1 - 1), 0, v_off], order=[0, 2])

                            # ---- B2 window.  Events on BOTH AIVs (DESIGN :765 / :939 verbatim: "只有
                            # sub_id == 0 的 AIV 执行算术与 store，sub_id == 1 的 AIV 执行完全相同的循环与完全相同的跨核事件序列
                            # （INTRA_BLOCK 要求 AIC 等待两个 AIV 子核都完成）"), arithmetic and stores on AIV0 only.
                            # Guarding the events instead was measured as rung 6's aicore timeout. ----
                            pl.system.wait_cross_core(
                                pipe=pl.PipeType.V, event_id=EV_B2_READY, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK
                            )
                            # Staged Module-2 observation.  KB section 17: the consumer's FIRST touch
                            # of a cross-core buffer must be a VECTOR instruction, and a GM forward
                            # must store a LOCAL tile, never the cross-core one.  Storing straight off
                            # s_vec is what the ladder measured as 5/32768 at rung 8.
                            # score_s's staged store and its section-17 landing copy are GONE.
                            # SECTION 17 STILL HOLDS: the first touch of the cross-core s_vec in each
                            # branch below is a VF (klg_z_vf on pass 0, klg_dw_vf on pass 1), not a DMA,
                            # and no pl.store in this file reads a cross-core tile -- selfcheck check 4
                            # is the standing gate on that.
                            if c3.pass_ == 0:
                                klg_z_vf(s_vec_v, w_row, z_row, v_off, v_vts)  # V8
                                klg_pacc_vf(pdi_row, si3, red_row, v_off, v_vts, v_lim)  # V10
                                if c3.mid == 1:
                                    # ---- row finalize, immediately after pass 0's last chunk ----
                                    klg_softmax_vf(z_row, y_row, v_rk)  # V11
                                    pl.set_validshape(y_row, [1, kmax])  # V12
                                    if v_t < t1:  # TOKEN SPLIT: odd-t1 bounds guard (see section 13)
                                        pl.store(softmax_out, y_row, [v_t, 0, 0], order=[0, 2])
                                    klg_di_vf(y_row, pdi_row, red_row, v_rk)  # V13
                            else:
                                klg_dwds_vf(s_vec_v, pdi_row, dw_row, w_row, ds_ub, v_off, v_vts)
                                if c3.last == 1:
                                    if v_t < t1:  # TOKEN SPLIT: odd-t1 bounds guard (see section 13)
                                        pl.store(dw_out, dw_row, [v_t, 0])  # V21
                            pl.system.set_cross_core(
                                pipe=pl.PipeType.V, event_id=EV_B2_RELEASE, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK
                            )

                            # ---- Module 4's PRODUCER half, V16..V20 (DESIGN section 1.3), pass 1 only.
                            # Gate is `is_valid && pass == 1`, matching s4's gate exactly (section 2.3
                            # rule 1).  The three .next() calls sit OUTSIDE `if sub_id == 0:` so both AIVs
                            # advance their cursors identically -- the same split the B1 producer uses and
                            # the ladder measured -- while every transpose / move / insert is AIV0-only and
                            # the event pair runs on both AIVs. ----
                            if c3.pass_ == 1:
                                ds_l1_cur = ds_l1_g2.next()  # V17
                                kkt_l1_cur = kkt_l1_nz_g2[c3.gen * 5 + c3.b3]  # V20 (NZ producer view)
                                # qt_l1's cursor advances once per TOKEN, not once per tick -- the same
                                # gate the Cube's consumer uses.  It sits OUTSIDE `if sub_id == 0:` with
                                # the other three so both AIVs advance identically, and the handle comes
                                # from an UNCONDITIONAL .current() (DESIGN :775 scope rule).
                                if c3.chunk == 0:
                                    qt_l1_nz_g2.next()
                                qt_l1_cur = qt_l1_nz_g2.current()
                                if v3slot == 0:
                                    pl.system.wait_cross_core(
                                        pipe=pl.PipeType.MTE3,
                                        event_id=EV_B3_RELEASE[0],
                                        sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
                                    )
                                else:
                                    pl.system.wait_cross_core(
                                        pipe=pl.PipeType.MTE3,
                                        event_id=EV_B3_RELEASE[1],
                                        sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
                                    )
                                nz_s = nz_stage_s_g.current()
                                nz_t = nz_stage_t_g.current()
                                # ---- V16a..V16d: the qt WORKAROUND, once per token.  This is the
                                # kkt pattern (V19/V20) with `query` in place of `kk_ub`, and it
                                # reuses V19/V20's own scratch: q_nd aliases nz_stage, the
                                # transpose lands in kkt_ub, the NZ staging is nz_t.  All four
                                # reuses are ordered by mutex 7 / mutex 6 against V19/V20 below,
                                # which is the same device the B1 producer already relies on.
                                # The load is ASCENDING: GM -> UB, no transpose in the DMA.
                                # The transpose is [64,128] -> [128,64] bf16, which is EXACTLY the
                                # shape BP-2a measured EXACT on this box.
                                if c3.chunk == 0:
                                    q_nd = q_ub_g.current()
                                    # `c3.t` directly, NOT `v_t`: v_t is assigned inside the
                                    # B2-consumer branch, a SIBLING of this one, and DESIGN :775
                                    # records that such a variable leaves scope in DSL codegen.
                                    pl.load(q_nd, query, [pl.min(c3.t, t1 - 1), 0, 0], order=[1, 2])  # V16a
                                    pl.move(nz_t, q_nd)  # normal Q ND->NZ
                                    pl.insert(qt_l1_cur, nz_t, [0, 0])  # V16d
                                pl.move(nz_s, ds_ub)  # V17 ND -> NZ
                                pl.insert(ds_l1_cur, nz_s, [0, 0])
                                # ---- V19.  `.previous()` is a one-TICK lookback, not a one-WRITE
                                # lookback: it lowers to (cursor + n_slots - 1) with NO write-back, so at
                                # tick T it names the slot `next()` returned at tick T-1 (BP-9 ruling,
                                # architect 2026-09-25, measured on device -- three arms, all PASS).
                                # Correctness needs exactly two things, neither of them a slot count:
                                #   (i) the reader's delay is exactly 1 -- s3 is;
                                #  (ii) the reader's gate is the producer's gate shifted one tick --
                                #       `kk_ub_g.next()` runs UNCONDITIONALLY every tick (it is at the
                                #       top of this tick body, outside both `if tk < 2*nce` and
                                #       `if sub_id == 0`), and this read is under `if c3.is_valid`, which
                                #       is task T-1's validity.  So a read at tick T implies a write at
                                #       tick T-1, into the very slot previous() selects.
                                # TWO HARD REQUIREMENTS from section 2.3 rule 3, each with a negative
                                # control in event_pairing_sim.py:
                                #   (1) this call must stay INSIDE `if c3.is_valid` -- it must NOT be
                                #       hoisted to the top of the tick body "for symmetry with next()";
                                #   (2) the step that reads it must not leave the delay-1 stage.
                                # Both hold here.  Do not "tidy" either one.
                                # ---- V19 via Cube transpose (milestone 3).  The Cube drained KK^T to
                                # tr_vec over the TR window; cast fp32 -> bf16 here, replacing the per-tick
                                # pl.transpose.  tr_vec is ONE slot, but SingleModeVec0/1 put stream A in
                                # AIV0's private UB and stream B in AIV1's, so each sub_id reads its own
                                # token's KK^T from the same handle.  The event pair sits in BOTH the
                                # `if sub_id==0` (stream A) and `else` (stream B) branches, so BOTH AIVs
                                # fire it -- the INTRA_BLOCK rendezvous the Cube's FIX set/wait needs.
                                # S1 advances the two KK slots every tick. S3 consumes
                                # only the previous valid task, exactly one tick later.
                                if c3.cache == 0:
                                    kk_previous = kk_ub_g.previous()
                                    pl.move(nz_t, kk_previous)  # fallback normal KK
                                    pl.insert(kkt_l1_cur, nz_t, [0, 0])
                                if v3slot == 0:
                                    pl.system.set_cross_core(
                                        pipe=pl.PipeType.MTE3,
                                        event_id=EV_B3_READY[0],
                                        sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
                                    )
                                else:
                                    pl.system.set_cross_core(
                                        pipe=pl.PipeType.MTE3,
                                        event_id=EV_B3_READY[1],
                                        sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
                                    )
                                v3slot = 1 - v3slot

                # ---- Module 5, stage s5_out (delay 2, DESIGN section 1.5).  V22..V27.
                # Gate is `is_valid && pass == 1`, the same gate s4 uses on the Cube side.
                c5 = ctx[(tick + CTX_DEPTH - 2) % CTX_DEPTH]
                if c5.is_valid == 1:
                    # Same always-true structural guard as the B2 consumer above, for the same
                    # measured reason: probe_klg_delay1_vcguard.py sha f2f483dd3fae000e.  Structural,
                    # not semantic.  It goes when the framework behaviour is explained or fixed.
                    if c5.chunk >= 0:
                        if c5.pass_ == 1:
                            dqp_v = dqp_vec_g.current()
                            dkg_v = dkg_vec_g.current()
                            dq_row = dq_row_g.current()
                            dq_half = dq_half_g.current()
                            si_out = si_out_g.current()
                            dkrow = dkrow_g.current()
                            v5_t = c5.t
                            v5_base = c5.chunk * TS
                            v5_rk = c5.real_k
                            v5_lim = c5.lim
                            v5_ks = c5.ks
                            v5_avail = c5.avail
                            v5_gsafe = c5.g_safe
                            v5_span = pl.max(c5.vts, 1)  # deviation D3: set_validshape >= 1

                            # V23: s5 zeroes dq_row on ITS OWN first pass-1 task (DESIGN section 4's
                            # state-ownership table).  klg_wide_zero_vf is the independent helper
                            # DESIGN section 1.5 requires -- 128 whole-register stores at offsets
                            # r*LANES, all 32 B aligned -- and NOT klg_rowinit_vf, whose arity, None
                            # arguments and [1, kmax] shape all fail (the three-point defect record at
                            # section 1.5).  It has run 11/11 twice as the staged consumer's zeroing.
                            # Touches no cross-core tile, so it is issued before the B4 wait.
                            if (
                                v5_t < t1
                            ):  # TOKEN SPLIT: was `sub_id == 0`.  s5 runs on BOTH sub-cores now, each for its own token; what remains is the odd-t1 bounds guard.
                                if c5.chunk == 0:
                                    klg_wide_zero_vf(dq_row, WIDE_REGS)

                            # V22
                            pl.system.wait_cross_core(
                                pipe=pl.PipeType.V, event_id=EV_B4_READY, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK
                            )

                            # EVERY store below is gated on sub_id == 0, DESIGN :765 / :939.  In the CV
                            # shape get_subblock_num() is 2 and BOTH AIV sub-cores execute this section,
                            # so an UNGATED atomic add would apply every dk contribution TWICE and dk
                            # would come out exactly 2x -- a single uniform wrong answer that passes
                            # every structural check and reads as a scale bug.  The vector-only 8.0 that
                            # probe_klg_m5_constructs.py measured does NOT bear on this: that kernel had
                            # no section_cube, so get_subblock_num() was 1 there (D3) and 8 was the
                            # complete count.  probe_klg_m5_cvatomic.py measures it in the CV shape.
                            if (
                                v5_t < t1
                            ):  # TOKEN SPLIT: was `sub_id == 0`.  s5 runs on BOTH sub-cores now, each for its own token; what remains is the odd-t1 bounds guard.
                                # V24: dq_row += dqp_vec.  This is also the section 17 FIRST TOUCH of
                                # the cross-core dqp_vec -- a VECTOR instruction, not a DMA.
                                klg_wide_acc_vf(dq_row, dqp_v, WIDE_REGS)

                                # V25: s5 loads its OWN index chunk (never s1's tile -- cross-stage
                                # state is what DESIGN section 4's ownership table forbids), then one
                                # atomic row per valid column.
                                # NARROWED DMA, adjudicated exactly as s1's si_c is: a pl.load narrowed
                                # by set_validshape writes whole 32 B destination blocks and the last
                                # block's remainder is UNDEFINED (measured).  si_out[0, s] is read for
                                # every s < TS, but every use is gated by `col < real_k`, so an
                                # undefined value can only reach `ok = 0` and `g_use = g_safe`.  Module 1
                                # is bit-exact on device with the identical construction.
                                pl.set_validshape(si_out, [1, v5_span])
                                pl.load(si_out, sparse_indices, [pl.min(v5_t, t1 - 1), 0, v5_base], order=[0, 2])
                                # Complete N1 reductions remain in the original FP32 Cube tile.
                                # SIMT runs on V after B4.READY and before the original B4.RELEASE.
                                pl.simt.launch(
                                    klg_simt_dk_atomic,
                                    threads=1024,
                                    args=(dkg_v, si_out, dk_out, v5_base, v5_rk, v5_lim, v5_ks),
                                )

                                # V27: the token's last pass-1 task casts dq_row to bf16 and writes it
                                # out in two [32, D] halves.  SPEC section 12 A8 requires ties-to-even,
                                # which is CAST_RINT; the two-cast + de_interleave composition is
                                # measured bit-exact against torch (probe_klg_m5_constructs, arm cast).
                                if c5.last == 1:
                                    klg_dqcast_vf(dq_row, dq_half, 0, DQ_HALF_ROWS)
                                    pl.store(dq_out, dq_half, [v5_t, 0, 0], order=[1, 2])
                                    klg_dqcast_vf(dq_row, dq_half, DQ_HALF_ROWS, DQ_HALF_ROWS)
                                    pl.store(dq_out, dq_half, [v5_t, DQ_HALF_ROWS, 0], order=[1, 2])

                            # V26
                            pl.system.set_cross_core(
                                pipe=pl.PipeType.V, event_id=EV_B4_RELEASE, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK
                            )
                tick = tick + 1

        # Vector epilogue: the Vector PRODUCES B1 and B3, so it waits its own boundaries' RELEASE
        # credits back before leaving, one per slot -- DESIGN :935 "Vector 侧收尾：对 B1 / B3 的每个
        # 槽位各等一次 RELEASE".  The arithmetic: the Cube pre-fires 2 RELEASE per boundary and then
        # fires 1 per valid task while the Vector waits 1 per valid task, so exactly the 2 pre-fired
        # credits per boundary are still pending here and these four waits retire immediately.
        # There is no cycle: the Cube's remaining work needs only READY credits the Vector has
        # already fired, plus B2/B4 RELEASE which s3 and the staged B4 consumer have already sent.
        # module123 OMITTED these for B1 and still passed 11/11, because leftover credits are inert
        # rather than fatal -- but design_checks/event_pairing_sim.py's negative control "producer
        # skips the tail RELEASE wait" fires on exactly that, reporting the unconsumed
        # [(2,1),(3,1),(5,1),(8,1),(9,1),(11,1)], so the omission was a real if harmless deviation
        # and this file does not repeat it.
        pl.system.wait_cross_core(
            pipe=pl.PipeType.MTE3, event_id=EV_B1_RELEASE[0], sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK
        )
        pl.system.wait_cross_core(
            pipe=pl.PipeType.MTE3, event_id=EV_B1_RELEASE[1], sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK
        )
        pl.system.wait_cross_core(
            pipe=pl.PipeType.MTE3, event_id=EV_B3_RELEASE[0], sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK
        )
        pl.system.wait_cross_core(
            pipe=pl.PipeType.MTE3, event_id=EV_B3_RELEASE[1], sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK
        )


# =====================================================================================
# Entry point (hard contract) -- constraints/wrapper-boundary.md allowlist only:
# metadata reads, pure Python int arithmetic, torch.empty of the declared outputs, one launch.
# =====================================================================================
def _require_dense(name, tensor, ndim, dtype):
    """Integer-only density check: ``stride()[d] == prod(shape[d+1:])`` for every dim with size > 1.

    ``is_contiguous`` is deliberately NOT used -- it is not in the wrapper-boundary allowlist.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    if tensor.dim() != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got {tensor.dim()}-D")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
    shape = tensor.shape
    stride = tensor.stride()
    expect = 1
    for d in range(ndim - 1, -1, -1):
        if int(shape[d]) > 1 and int(stride[d]) != expect:
            raise ValueError(f"{name} is not densely packed: stride()[{d}]={int(stride[d])} != {expect}")
        expect = expect * int(shape[d])
    return shape


def sparse_li_klgrad_pro(
    query=None,
    key=None,
    weights=None,
    sparse_indices=None,
    attn_softmax_l1_norm=None,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    cmp_residual_k=None,
    mask_mode=3,
    cmp_ratio=4,
    metadata=None,
    layout_q="TND",
    layout_k="TND",
    seqused_q=None,
    seqused_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    q=None,
    k=None,
    w=None,
):
    """Public LI gradient ABI; every calculation and output initialization is in one kernel."""
    if q is not None:
        if query is not None:
            raise TypeError("query and q cannot both be supplied")
        query = q
    if k is not None:
        if key is not None:
            raise TypeError("key and k cannot both be supplied")
        key = k
    if w is not None:
        if weights is not None:
            raise TypeError("weights and w cannot both be supplied")
        weights = w
    if layout_q != "TND" or layout_k != "TND":
        raise ValueError("only TND query and key layouts are supported")
    if int(mask_mode) not in MASK_MODES:
        raise ValueError(f"mask_mode must be one of {MASK_MODES}, got {mask_mode!r}")
    if int(cmp_ratio) != CMP_RATIO_DEFAULT:
        raise ValueError(f"cmp_ratio must be {CMP_RATIO_DEFAULT}, got {cmp_ratio!r}")

    q_shape = _require_dense("query", query, 3, torch.bfloat16)
    k_shape = _require_dense("key", key, 3, torch.bfloat16)
    w_shape = _require_dense("weights", weights, 2, torch.float32)
    si_shape = _require_dense("sparse_indices", sparse_indices, 3, torch.int32)
    p_shape = _require_dense("attn_softmax_l1_norm", attn_softmax_l1_norm, 3, torch.float32)
    cu_q_shape = _require_dense("cu_seqlens_q", cu_seqlens_q, 1, torch.int32)
    cu_k_shape = _require_dense("cu_seqlens_k", cu_seqlens_k, 1, torch.int32)
    res_shape = _require_dense("cmp_residual_k", cmp_residual_k, 1, torch.int32)

    t1 = int(q_shape[0])
    t2 = int(k_shape[0])
    kmax = int(si_shape[2])
    if int(q_shape[1]) != N1 or int(q_shape[2]) != D:
        raise ValueError(f"query must be [T1, {N1}, {D}], got {tuple(int(v) for v in q_shape)}")
    if int(k_shape[1]) != 1 or int(k_shape[2]) != D:
        raise ValueError(f"key must be [T2, 1, {D}], got {tuple(int(v) for v in k_shape)}")
    if int(w_shape[0]) != t1 or int(w_shape[1]) != N1:
        raise ValueError(f"weights must be [{t1}, {N1}], got {tuple(int(v) for v in w_shape)}")
    if kmax not in K_CHOICES:
        raise ValueError(f"K must be one of {K_CHOICES}, got {kmax}")
    if int(si_shape[0]) != t1 or int(si_shape[1]) != 1:
        raise ValueError(f"sparse_indices must be [{t1}, 1, {kmax}], got {tuple(int(v) for v in si_shape)}")
    if tuple(int(v) for v in p_shape) != tuple(int(v) for v in si_shape):
        raise ValueError("attn_softmax_l1_norm and sparse_indices must have the same shape")
    n_batch = int(cu_q_shape[0]) - 1
    if n_batch < 1:
        raise ValueError(f"cu_seqlens_q must hold B+1 >= 2 entries, got {int(cu_q_shape[0])}")
    if int(cu_k_shape[0]) != n_batch + 1:
        raise ValueError(f"cu_seqlens_k must hold {n_batch + 1} entries, got {int(cu_k_shape[0])}")
    if int(res_shape[0]) != n_batch:
        raise ValueError(f"cmp_residual_k must hold {n_batch} entries, got {int(res_shape[0])}")
    # _require_dense has already established these inputs are tensors.
    dev = cast("torch.Tensor", query).device
    if dev.type != "npu":
        raise ValueError("query must be on an NPU")
    if not 1 <= t1 <= 65536 or not 1 <= t2 <= 262144 or not 1 <= n_batch <= 65536:
        raise ValueError("T1, T2, or B is outside the public contract range")
    for name, tensor in (
        ("key", key),
        ("weights", weights),
        ("sparse_indices", sparse_indices),
        ("attn_softmax_l1_norm", attn_softmax_l1_norm),
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_k", cu_seqlens_k),
        ("cmp_residual_k", cmp_residual_k),
    ):
        checked_tensor = cast("torch.Tensor", tensor)
        if checked_tensor.device != dev:
            raise ValueError(f"{name} is on {checked_tensor.device}, query is on {dev}")

    from pypto_pro.runtime.platform import get_platform_info  # pyrefly: ignore [missing-import]

    # D5 REVISED for Module 2.  Module 1's kernel was section_vector only -> aiv-only -> the
    # launch width was vector_core_num.  Adding section_cube makes this a MIXED kernel, whose
    # launch width is core_num (one AIC + its AIVs per block).  Launching a mixed kernel at
    # vector_core_num would ask for more blocks than there are cube cores.
    max_blocks = int(get_platform_info().core_num)
    block_dim = max(1, min(max_blocks, t1))

    # THE FOUR CONTRACT OUTPUTS.  dq is bf16 -- the comparator rejects fp32 PRE-NUMERICALLY
    # ("dq: FAIL shape/dtype mismatch ... vs contract torch.bfloat16", measured on the box), which is
    # also where the staged gates' fp32-dq mismatch finally resolves.
    #   dq  written ONCE per token over its whole [N1, D] extent by V27 -> torch.empty, so a token
    #       the kernel never finalizes shows as garbage in the gate instead of matching a zero golden
    #   dk  torch.empty ON PURPOSE: the KERNEL zeroes it (section 1.1) because pl.store(AtomicAdd)
    #       requires a pre-zeroed destination (store.md:47) and the wrapper may only torch.empty
    #       (wrapper-boundary.md).  Pre-zeroing here would HIDE a failed prologue, which is exactly
    #       what probe_klg_m5_constructs.py's NaN pre-fill was built to catch.
    #   dw, softmax_out  written whole, once per token -> torch.empty
    dq_out = torch.empty((t1, N1, D), dtype=torch.bfloat16, device=dev)
    dk_out = torch.empty((t2, 1, D), dtype=torch.float32, device=dev)
    dw_out = torch.empty((t1, N1), dtype=torch.float32, device=dev)
    softmax_out = torch.empty((t1, 1, kmax), dtype=torch.float32, device=dev)
    sparse_li_klgrad_pro_kernel_opt20261013finaldwds_public[None, block_dim, {"MaskMode": int(mask_mode)}](
        query,
        key,
        weights,
        sparse_indices,
        attn_softmax_l1_norm,
        cu_seqlens_q,
        cu_seqlens_k,
        cmp_residual_k,
        dq_out,
        dk_out,
        dw_out,
        softmax_out,
        int(cmp_ratio),
    )
    return (dq_out, dk_out, dw_out, softmax_out)
