# DEPTH — Monocular Depth Estimation

## OVERVIEW

Linear and DPT-head depth estimation on NYUv2 and SYNTHMIX datasets. Self-contained eval pipeline with own config, data, training, and inference.

## STRUCTURE

```
depth/
├── run.py               # Entry point (main + __main__)
├── config.py            # OmegaConf config dataclass
├── configs/             # YAML configs (NYU linear, NYU SYNTHMIX DPT inference)
├── train.py             # Training loop
├── eval.py              # Evaluation loop
├── data.py              # Depth-specific data loading + transforms
├── datasets/            # Dataset-specific loaders
├── loss.py              # Depth-specific losses (SILog, gradient matching)
├── metrics.py           # Depth metrics (RMSE, AbsRel, δ thresholds)
├── models/              # Head architectures
│   ├── encoder.py       # DINOv3 backbone wrapper for depth
│   ├── dpt_head.py      # DPT (Dense Prediction Transformer) head
│   └── linear_head.py   # Simple linear probe head
├── transforms.py        # Depth-specific augmentations
├── schedulers.py        # LR schedulers (XXX deprecation debt)
├── checkpoint_utils.py  # Depth checkpoint save/load
├── utils.py             # Misc utilities
└── visualization_utils.py # Depth map visualization
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add depth dataset | `datasets/` | Follow NYU pattern |
| Change depth head | `models/` | `dpt_head.py` (DPT) or `linear_head.py` |
| Modify depth metrics | `metrics.py` | Standard depth metrics |
| Run inference | `configs/config-nyu-synthmix-dpt-inference.yaml` | Pretrained DPT model |
| Train linear probe | `configs/config-nyu.yaml` | Linear head on NYUv2 |

## ANTI-PATTERNS

- `schedulers.py:36,103` — `XXX` deprecation markers. Legacy scheduler code with backward-compat debt.
- `models/encoder.py:46` — `XXX` backward-compatibility note.
- `models/dpt_head.py:475,483` — TODO items in DPT head.
