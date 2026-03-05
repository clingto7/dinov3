# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-05
**Commit:** 91ffb68
**Branch:** main

## GUIDELINES 工作总则

- 优先遵循本文件
- 使用UV管理python环境
  - 如,添加python依赖使用`uv add xxx`
- 本机器配置为RTX4060显卡 13th Gen Intel(R) Core(TM) i9-13900H
- 对跨模块变更（环境注册、算法接入、配置项）要保持“代码 + 配置 + 文档”同步。
- 每次工作可以修改的文件**仅限项目目录内**
- 每次工作**基本流程**
  - 阅读该文档
  - 根据AGENTS.md文档确定自己任务下一步需要看哪个部分并仔细阅读
  - 进行修改
  - 执行必要的测试
  - 直到达到目标
  - 将以上过程记录在一个md文件内，命名规则<日期>_<session-id>_<概括任务信息>，放在**report/目录**下
    - 并且生成一份给其他agent读，方便你的其他“同事”了解工作情况的文档，放在**report/coop/目录**下，命名规则同上
- 同一个会话中一般有多次工作，每次工作**如有必要的话**要**更新**对应的report和coop下的文件，如：1. 做了新的修改，在report中的内容补充对当次工作的描述；2. coop下的内容经过当次工作修改后已不适用，需要修改；......

## OVERVIEW

DINOv3 — Meta AI's self-supervised vision foundation model (ViT + ConvNeXt backbones). PyTorch research codebase for pretraining, distillation, and downstream evaluation (classification, segmentation, depth, detection, text alignment). Python 3.11+, OmegaConf config, SLURM/submitit job orchestration.

## STRUCTURE

```
dinov3/
├── models/          # Backbone architectures (ViT, ConvNeXt)
├── layers/          # NN building blocks (attention, RoPE, patch embed, FP8, etc.)
├── loss/            # SSL losses (DINO cls, iBOT patch, KoLeo, Gram)
├── train/           # Training loop, meta-arch (SSLMetaArch), LR scheduler
├── eval/            # Downstream task evaluation (see eval/AGENTS.md)
│   ├── depth/       # Monocular depth estimation (NYUv2, SYNTHMIX)
│   ├── segmentation/# Semantic segmentation (ADE20K, Mask2Former)
│   ├── detection/   # Object detection (COCO, DETR-based)
│   └── text/        # Text alignment / dino.txt (CLIP-style)
├── data/            # Data pipeline (datasets, augmentations, samplers, masking)
├── configs/         # OmegaConf YAML configs (ssl_default_config.yaml + train/)
├── hub/             # torch.hub entry points (backbones, classifiers, depthers, etc.)
├── distributed/     # Distributed training primitives (FSDP wrapper)
├── fsdp/            # Activation checkpointing + compile + parallelize
├── run/             # Job submission (submitit/SLURM) + runtime bootstrap
├── checkpointer/    # Checkpoint save/load with FSDP support
├── logging/         # Logging setup helpers
├── utils/           # Misc utilities (cluster defaults, dtype, seeds)
├── thirdparty/      # Vendored CLIP (text encoder for dino.txt)
└── env/             # Environment detection (empty init)
hubconf.py           # torch.hub.load() surface
notebooks/           # Demo notebooks (PCA, matching, segmentation, tracking)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add/modify backbone | `dinov3/models/` | `vision_transformer.py`, `convnext.py` |
| Change attention/layers | `dinov3/layers/` | See `layers/AGENTS.md` |
| Add new loss | `dinov3/loss/` | Follow existing pattern (`DINOLoss`, `iBOTPatchLoss`) |
| Modify training loop | `dinov3/train/ssl_meta_arch.py` | `SSLMetaArch` is the core meta-architecture |
| Multi-distillation | `dinov3/train/multidist_meta_arch.py` | Separate meta-arch for multi-student distillation |
| Add eval task | `dinov3/eval/` | Each task is self-contained subdirectory |
| Add dataset | `dinov3/data/datasets/` | Subclass pattern, dump_extra() for metadata |
| Change augmentations | `dinov3/data/augmentations.py` | `DataAugmentationDINO` class |
| Expose via torch.hub | `dinov3/hub/` + `hubconf.py` | Add function in hub/, re-export in hubconf.py |
| Change config defaults | `dinov3/configs/ssl_default_config.yaml` | OmegaConf schema |
| Add train config | `dinov3/configs/train/` | YAML inheriting from default |
| Modify job submission | `dinov3/run/submit.py` | Submitit/SLURM launcher |
| Runtime bootstrap | `dinov3/run/init.py` | `job_context` context manager |

## CONVENTIONS

- **Config**: OmegaConf throughout. CLI overrides via `key=value` args, merged with YAML configs. Default config in `ssl_default_config.yaml`.
- **Entry pattern**: All runnable scripts expose `main()` + `if __name__ == "__main__"`. Invoked via `python -m dinov3.run.submit <script.py> [args]` for SLURM or directly with `python`/`torchrun`.
- **Imports**: `PYTHONPATH=.` required at invocation (not installed as editable by default).
- **Logging**: Single logger `logging.getLogger("dinov3")` used everywhere.
- **Line length**: 120 chars (ruff). `E501` ignored in lint but `ruff format` enforces.
- **Type checking**: mypy with `ignore_missing_imports = true`, scoped to `dinov3/`.
- **`__init__.py` exports**: Inconsistent — some barrel-export (layers, loss, data), some empty. Check per-module.
- **Naming**: `run/init.py` is a module (not `__init__.py`) — imports as `dinov3.run.init`.
- **Third-party**: CLIP vendored in `thirdparty/CLIP/clip/` (uppercase dir, non-PEP8).

## ANTI-PATTERNS (THIS PROJECT)

- `dinov3/layers/block.py:81` — "do not" comment: respect the constraint on block modification.
- `dinov3/models/vision_transformer.py:175` — "never" comment: follow the stated invariant.
- `dinov3/checkpointer/checkpointer.py:50` — "do not" constraint on checkpoint handling.
- `dinov3/data/samplers.py:71,215` — "Always" constraints on sampler behavior.
- Detection/segmentation eval code has significant hack/TODO debt — treat as read-only reference code unless specifically modifying those tasks.
- No `type: ignore` or `noqa` suppressions in new code. Existing ones are concentrated in `eval/results.py` and `distributed/torch_distributed_wrapper.py`.

## COMMANDS

```bash
# Environment setup
micromamba env create -f conda.yaml && micromamba activate dinov3

# Training (SLURM)
PYTHONPATH=. python -m dinov3.run.submit dinov3/train/train.py \
  --nodes 4 --config-file dinov3/configs/train/vitl_im1k_lin834.yaml \
  --output-dir <OUT> train.dataset_path=ImageNet22k:root=<DATA>:extra=<DATA>

# Evaluation (examples)
PYTHONPATH=. python -m dinov3.run.submit dinov3/eval/knn.py \
  model.config_file=<OUT>/config.yaml model.pretrained_weights=<OUT>/teacher_checkpoint.pth \
  output_dir=<OUT> train.dataset=ImageNet:split=TRAIN:root=<DATA>:extra=<DATA> \
  eval.test_dataset=ImageNet:split=VAL:root=<DATA>:extra=<DATA>

# Lint (CI mirrors this)
ruff check dinov3
ruff format --diff dinov3
mypy --txt-report .
pylint --exit-zero dinov3
docstr-coverage dinov3
```

## NOTES

- **No test suite**: No pytest/unittest framework. Only ad-hoc CUDA op test at `eval/segmentation/models/utils/ops/test.py`. mypy config references `dinov3/tests/` pattern but directory doesn't exist yet.
- **Multiple setup.py**: Root + `dinov3/eval/setup.py` + `dinov3/eval/segmentation/models/utils/ops/setup.py` (CUDA extension build).
- **SLURM-oriented**: Hardcoded cluster defaults in `dinov3/utils/cluster.py` (`fair_amaia_cw_explore` partition). Override via config for other environments.
- **CI is lint-only**: No build/test CI. `pylint --exit-zero` never fails. `docstr-coverage` `fail_under: 0`.
- **3-stage ViT-7B training**: Pretrain → Gram anchoring → High-res adaptation. Each stage has separate config YAML.
- **SAT-493M models**: Satellite imagery models use different normalization constants than LVD-1689M web models.
