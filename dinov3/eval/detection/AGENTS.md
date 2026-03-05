# DETECTION — Object Detection (COCO)

## OVERVIEW

DETR-based object detection on COCO2017. Ported model code with significant hack/workaround debt.

## STRUCTURE

```
detection/
├── __init__.py
├── config.py            # Detection config dataclass
├── models/
│   ├── __init__.py
│   ├── backbone.py      # DINOv3 backbone wrapper for detection
│   ├── detr.py          # DETR main model — hack comments at lines 103, 111
│   ├── transformer.py   # Transformer encoder/decoder — hacks at lines 232, 393
│   ├── transformer_encoder.py  # Separate encoder module
│   ├── global_ape_decoder.py   # APE decoder — hacks at lines 211, 278
│   ├── global_rpe_decomp_decoder.py  # RPE decoder — hacks at lines 308, 390
│   ├── position_encoding.py    # Positional encoding (TODO at line 120)
│   ├── utils.py         # Detection model utilities
│   └── windows.py       # Window attention utilities
└── util/
    ├── box_ops.py       # Box coordinate transforms — hack at line 69
    └── misc.py          # NestedTensor, interpolation utils — TODOs at lines 81, 83
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Modify DETR architecture | `models/detr.py` | Main detection model |
| Change decoder | `models/global_ape_decoder.py` or `global_rpe_decomp_decoder.py` | Two decoder variants |
| Box operations | `util/box_ops.py` | COCO-format box transforms |

## ANTI-PATTERNS

- **Treat as read-only reference code** unless specifically tasked with detection modifications.
- 10+ hack/workaround comments across model files — ported from external repo without cleanup.
- `util/misc.py:81,83` — TODO items on NestedTensor handling.
- No standalone `run.py` in this directory — detection is invoked through the hub interface (`dinov3_vit7b16_de`).
