# SEGMENTATION — Semantic Segmentation (ADE20K)

## OVERVIEW

Mask2Former and linear head segmentation on ADE20K. Deepest module tree in the project — includes custom CUDA ops, backbone adapter, and multiple decoder heads.

## STRUCTURE

```
segmentation/
├── run.py               # Entry point
├── config.py            # OmegaConf config dataclass
├── configs/             # YAML configs (ADE20K linear training, M2F inference)
├── train.py             # Training loop
├── eval.py              # Evaluation loop
├── inference.py         # Slide-window inference for segmentation
├── loss.py              # Segmentation losses
├── metrics.py           # mIoU, pixel accuracy
├── transforms.py        # Segmentation-specific augmentations
├── schedulers.py        # LR schedulers
└── models/
    ├── __init__.py      # Model factory (build_model) — **"Important" comment at line 114**
    ├── backbone/
    │   └── dinov3_adapter.py  # Wraps DINOv3 backbone for seg — **"Important" at line 325**
    ├── heads/
    │   ├── pixel_decoder.py   # FPN-style pixel decoder (heavy TODO debt)
    │   └── mask2former_transformer_decoder.py  # Mask2Former decoder (pyre-fixme debt)
    └── utils/
        └── ops/         # Custom CUDA/C++ deformable attention ops
            ├── setup.py # CUDA extension build (requires GPU)
            ├── test.py  # Ad-hoc gradcheck (not pytest)
            ├── modules/ # Python wrappers
            ├── functions/# Autograd functions
            └── src/     # C++/CUDA source (cpu/ + cuda/)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Change seg model factory | `models/__init__.py` | `build_model()` — check "Important" constraint |
| Modify backbone adapter | `models/backbone/dinov3_adapter.py` | Line 325 "Important" constraint |
| Change decoder head | `models/heads/` | `pixel_decoder.py` or `mask2former_transformer_decoder.py` |
| Build CUDA ops | `models/utils/ops/setup.py` | Requires CUDA toolkit. `pip install -e .` from that dir |
| Run inference | Use `inference.py` | Slide-window with configurable stride/crop |

## ANTI-PATTERNS

- `models/heads/pixel_decoder.py` — **Highest TODO density in project** (8+ TODO/FIXME). Ported code with significant cleanup debt.
- `models/heads/mask2former_transformer_decoder.py` — pyre-fixme suppressions and workaround comments.
- `models/__init__.py:114` and `models/backbone/dinov3_adapter.py:325` — "Important" constraints. Read before modifying.
- CUDA ops `test.py` is a standalone script, not integrated into any test framework.
- `schedulers.py` shares the same deprecation debt pattern as `depth/schedulers.py`.
