# EVAL — Downstream Task Evaluation

## OVERVIEW

Self-contained evaluation pipelines for DINOv3 backbone features. Each task subdirectory is independently runnable.

## STRUCTURE

```
eval/
├── knn.py           # k-NN classification (ImageNet)
├── linear.py        # Linear probing with augmentation (ImageNet)
├── log_regression.py# Logistic regression classification (ImageNet)
├── accumulators.py  # Feature accumulation utilities
├── data.py          # Eval-specific data loading
├── helpers.py       # Shared eval helpers (model building, feature extraction)
├── results.py       # Result serialization (CSV/JSON)
├── utils.py         # Eval utilities
├── metrics/         # Classification + ImageNet-C metrics
├── setup.py         # Separate packaging for eval dependencies
├── depth/           # Monocular depth estimation (see depth/AGENTS.md)
├── segmentation/    # Semantic segmentation (see segmentation/AGENTS.md)
├── detection/       # Object detection (see detection/AGENTS.md)
└── text/            # Text alignment / dino.txt (see text/AGENTS.md)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add new eval task | Create new subdirectory | Follow depth/segmentation pattern: `run.py` + `config.py` + `configs/` |
| Modify feature extraction | `helpers.py` | `ModelWithIntermediateLayers` wrapper |
| Add classification metric | `metrics/classification.py` | Used by linear/knn/logreg |
| Change result format | `results.py` | Heavy `type: ignore` — treat carefully |
| Shared data loading | `data.py` | Dataset/sampler setup for all eval tasks |

## CONVENTIONS

- **Entry pattern**: Each task has `run.py` with `main()` + `if __name__ == "__main__"`, invoked via `dinov3.run.submit`.
- **Config**: OmegaConf YAML in each task's `configs/` directory. Task-specific config dataclass in `config.py`.
- **Runtime bootstrap**: All tasks use `dinov3.run.init.job_context` for distributed/logging setup.
- **Model loading**: Backbone loaded via `dinov3.hub` functions or from checkpoint path. Hub function name passed as `model.dino_hub` config key.
- **Outputs**: Each task saves `model_final.pth` + `results-{task}.csv` + task-specific config YAML.

## ANTI-PATTERNS

- `results.py` has concentrated `type: ignore` suppressions — avoid extending without fixing.
- `setup.py` here is for eval-specific pip dependencies, NOT the main package.
- Detection/segmentation model code has significant hack/TODO debt (ported from external repos). Treat as reference unless specifically modifying.
