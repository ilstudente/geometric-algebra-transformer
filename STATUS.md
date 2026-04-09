# Session Status — Scalar-Path GM (Variant 3) Implementation

**Date:** 2026-04-09  
**Goal:** Implement and wire up Variant 3 (scalar-path-only GM replacement) for n-body experiments.

---

## What Has Been Built

### New source files
| File | Purpose |
|------|---------|
| `finite_subgroup_gatr/layers/scalar_gm_linear.py` | Core GM-structured linear layer: `[..., X, C_in] → [..., X, C_out]` via right-shift group convolution |
| `finite_subgroup_gatr/layers/scalar_ffn_bundle.py` | Configurable scalar FFN: modes `dense` / `gm_exact` / `gm_approx`; wraps ScalarGMLinear |
| `finite_subgroup_gatr/layers/scalar_attention_bundle.py` | GM-structured scalar Q/K projections for DiscreteMVAttention |
| `finite_subgroup_gatr/layers/scalar_mv_interface.py` | GM pre/post-projection around EquiLinear MV↔scalar interface |

### New test files (5 files, ~60 tests)
- `finite_subgroup_gatr/tests/test_scalar_gm_linear.py`
- `finite_subgroup_gatr/tests/test_scalar_ffn_bundle.py`
- `finite_subgroup_gatr/tests/test_scalar_attention_bundle.py`
- `finite_subgroup_gatr/tests/test_scalar_mv_interface.py`
- `finite_subgroup_gatr/tests/test_scalar_equivariance.py`

### Modified existing files
| File | Change |
|------|--------|
| `finite_subgroup_gatr/layers/geo_mlp.py` | Added optional `scalar_ffn: nn.Module | None = None` parameter — uses injected bundle instead of building dense `nn.Sequential` |
| `finite_subgroup_gatr/layers/discrete_attention.py` | Added optional `scalar_attn_bundle` parameter for GM-structured Q/K projections |
| `finite_subgroup_gatr/layers/common_blocks.py` (GMOnlyBlock) | Added 4 scalar-path params: `scalar_mlp_mode`, `scalar_mlp_radius`, `scalar_mlp_error_mode`, `scalar_mlp_error_rank`; conditionally builds and injects `ScalarFFNBundle` |
| `finite_subgroup_gatr/models/variant_a_gm_only.py` (VariantA) | Added the same 4 params, passes them to each GMOnlyBlock |
| `finite_subgroup_gatr/layers/__init__.py` | Added exports: ScalarGMLinear, ScalarFFNBundle, ScalarAttentionProjectionBundle, ScalarMVScalarInterface |
| `finite_subgroup_gatr/training/configs.py` | Added `ScalarPathGMConfig` dataclass |
| `finite_subgroup_gatr/training/equivariance_metrics.py` | Added `compute_scalar_equivariance_error` and `compute_scalar_path_metrics` |
| `scripts/slurm/02_group_comparison.sh` | Fixed `model.backend.n=8` → `++model.backend.n=8` (Hydra struct override for CyclicBackend `n` param) |

### New config
`config/model/finite_subgroup_scalar_gm_nbody.yaml` — wires VariantA with `c_s=32` (scalar path enabled), `scalar_mlp_mode: "dense"` (override at CLI to `gm_exact` or `gm_approx`).

---

## Key Architecture Notes

### Right-shift equivariance (important!)
`ScalarGMLinear` with right-shifts is **always exactly equivariant** to token permutations regardless of weight values. This is proven algebraically:
```
f(T_g s)(x) = sum_k W_k * s(g^{-1} * x * g_k) = T_g f(s)(x)
```
The `error_E` / `error_U/V` residuals in `error_mode != "none"` do NOT break token-permutation equivariance — they break it from the perspective of a different group action. The `error_norm()` is purely a regularization/diagnostic metric.

### GeoMLP scalar path
`GeoMLP.forward()` calls `self.scalar_mlp(x_s)` where `x_s: [B, X, C_s]` — the token axis is at position -2. This is correct for `ScalarFFNBundle` in GM mode (ScalarGMLinear expects `[..., X, C_in]`).

### ScalarFFNBundle injection path
```
GMOnlyBlock.__init__
  → if c_s is not None and scalar_mlp_mode != "dense":
      scalar_ffn = ScalarFFNBundle(backend, c_s, hidden_s=2*c_s, mode=scalar_mlp_mode, ...)
  → GeoMLP(..., scalar_ffn=scalar_ffn)
      → if scalar_ffn is not None: self.scalar_mlp = scalar_ffn
```

---

## OPEN ISSUE: Parameter Count Discrepancy

**Problem:** Dense (223,377) and gm_exact (223,377) show the same param count. Expected: gm_exact should have K × (c_in × c_out) params per ScalarGMLinear instead of 1 × (c_in × c_out), so should be significantly larger.

**What was confirmed working:**
- ScalarFFNBundle IS being instantiated for gm_exact (debug showed `block 0 scalar_mlp: ScalarFFNBundle`)
- `GeoMLP.scalar_mlp` IS set to ScalarFFNBundle
- Scalars DO flow through the network (32-dim scalar features visible from embed_in output)

**Likely cause being investigated at session end:**
The `OctahedralBackend.neighborhood(1)` uses a BFS word metric. Generators are 2 elements (90° rotations about z and x axes). With inverses added, K = neighborhood at radius=1 = {identity, R_z90, R_z90⁻¹, R_x90, R_x90⁻¹} = 5 elements.

For K=5, c_s=32, hidden_s=64:
- Dense total per FFN: 32×64 + 64 + 64×32 + 32 = 4192 params
- GM total per FFN: 5×(32×64) + 5×64(dead ApproxLinear biases) + 64 + 5×(64×32) + 5×32(dead biases) + 32 = 5×2048 + 320 + 64 + 5×2048 + 160 + 32 = 20768 params
- Delta per block: ~16576
- Delta for 4 blocks: ~66304

That still should differ. Need to actually debug with working Python env.

**Python environment issue:** The GMCNN venv at `GMCNN/.venv` does NOT have torch installed. The correct Python is at `/n/home10/dettel/conda_envs/gatr/bin/python` (used in all SLURM scripts).

**Debugging commands to run next session:**
```bash
PYTHON=/n/home10/dettel/conda_envs/gatr/bin/python
cd /n/home10/dettel/geometric-algebra-transformer
$PYTHON - <<'EOF'
from finite_subgroup_gatr.backends.rotation_backend import OctahedralBackend
b = OctahedralBackend()
nbr = b.neighborhood(1)
print('K:', len(nbr), 'indices:', nbr)
EOF
```

Then count params explicitly:
```bash
$PYTHON - <<'EOF'
import torch
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

# Quick manual instantiation
from finite_subgroup_gatr.backends.rotation_backend import OctahedralBackend
from finite_subgroup_gatr.models.variant_a_gm_only import VariantA

backend = OctahedralBackend()
for mode in ["dense", "gm_exact"]:
    m = VariantA(backend=backend, in_mv_channels=1, out_mv_channels=1,
                 c_mv=16, c_s=32, num_blocks=4, neighborhood_radius=1,
                 scalar_mlp_mode=mode)
    n = sum(p.numel() for p in m.parameters())
    # Count scalar_mlp params specifically
    scalar_params = sum(p.numel() for name, p in m.named_parameters() if 'scalar_mlp' in name)
    print(f'{mode}: total={n}, scalar_mlp={scalar_params}')
EOF
```

---

## PENDING TASKS

### 1. Fix/investigate param count discrepancy (before submission)
- Use `/n/home10/dettel/conda_envs/gatr/bin/python` (has torch)
- Debug commands above
- If K > 1 but counts match: check if ScalarFFNBundle params are somehow shared or if a bug in GMMixingWeights initialization means only 1 weight is actually created

### 2. Write SLURM ablation script `scripts/slurm/05_scalar_path_ablation.sh`
Pattern: copy from `01_variant_comparison.sh` (uses SLURM header, `conda_envs/gatr/bin/python`, module loads)

Experiments to run:
```
# Exp 1: Dense baseline (c_s=32, scalar_mlp_mode=dense)
model=finite_subgroup_scalar_gm_nbody  # defaults are dense

# Exp 2: Exact GM scalar MLP
model=finite_subgroup_scalar_gm_nbody model.net.scalar_mlp_mode=gm_exact

# Exp 3: Approx GM scalar MLP (low-rank residual)
model=finite_subgroup_scalar_gm_nbody model.net.scalar_mlp_mode=gm_approx \
  model.net.scalar_mlp_error_mode=low_rank model.net.scalar_mlp_error_rank=4
```

### 3. Submit and monitor jobs

---

## File Reference

```
geometric-algebra-transformer/
├── config/model/
│   └── finite_subgroup_scalar_gm_nbody.yaml    ← NEW: Variant 3 config
├── finite_subgroup_gatr/
│   ├── layers/
│   │   ├── scalar_gm_linear.py                  ← NEW
│   │   ├── scalar_ffn_bundle.py                 ← NEW
│   │   ├── scalar_attention_bundle.py           ← NEW
│   │   ├── scalar_mv_interface.py               ← NEW
│   │   ├── geo_mlp.py                           ← MODIFIED: scalar_ffn param
│   │   ├── discrete_attention.py                ← MODIFIED: scalar_attn_bundle param
│   │   ├── common_blocks.py                     ← MODIFIED: GMOnlyBlock scalar params
│   │   └── __init__.py                          ← MODIFIED: new exports
│   ├── models/
│   │   └── variant_a_gm_only.py                 ← MODIFIED: scalar params
│   ├── training/
│   │   ├── configs.py                           ← MODIFIED: ScalarPathGMConfig
│   │   └── equivariance_metrics.py              ← MODIFIED: scalar metrics
│   └── tests/
│       ├── test_scalar_gm_linear.py             ← NEW
│       ├── test_scalar_ffn_bundle.py            ← NEW
│       ├── test_scalar_attention_bundle.py      ← NEW
│       ├── test_scalar_mv_interface.py          ← NEW
│       └── test_scalar_equivariance.py          ← NEW
└── scripts/slurm/
    ├── 02_group_comparison.sh                   ← MODIFIED: ++model.backend.n fix
    └── 05_scalar_path_ablation.sh               ← TODO: create this
```

---

## SLURM Job Status (as of session start)

Previously submitted jobs (variants 4273767, groups 4273768, arch_sweep 4273770) had errors fixed:
- MLflow schema: ran `mlflow db upgrade` on the sqlite DB
- Hydra struct override: changed `model.backend.n=8` → `++model.backend.n=8`

Status of these jobs at session start was not checked. Use:
```bash
squeue -u dettel
```
