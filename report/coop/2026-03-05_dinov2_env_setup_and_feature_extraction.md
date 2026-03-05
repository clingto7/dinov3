# 2026-03-05 DINOv2 环境与特征提取 — 协作文档

## 当前状态

DINOv2环境已搭建完毕，两个特征提取脚本已编写并通过端到端测试。

## 环境信息

- **Python虚拟环境**: `/home/m1zu/ws/dinov3/.venv/` (Python 3.12.12)
- **PyTorch**: 2.10.0+cu128
- **DINOv2仓库**: `/home/m1zu/ws/dinov2/`（需要设置 `PYTHONPATH=/home/m1zu/ws/dinov2`）
- **GPU**: RTX 4060 Laptop GPU, 8GB VRAM
- **已下载权重**: `dinov2_vits14` (ViT-S/14, 22M params, embed_dim=384)

## 可用脚本

### `scripts/extract_image_features.py`
从图片提取DINOv2特征（CLS token, patch tokens, 中间层），支持批量处理、特征保存、数据集多样性统计。

```bash
source .venv/bin/activate
PYTHONPATH=/home/m1zu/ws/dinov2 python scripts/extract_image_features.py --demo
PYTHONPATH=/home/m1zu/ws/dinov2 python scripts/extract_image_features.py --image-dir <DIR> --output features.pt
```

### `scripts/extract_video_features.py`
从视频提取逐帧DINOv2特征，支持FPS控制、时序分析（帧间相似度、场景变化检测）、空间分析。

```bash
PYTHONPATH=/home/m1zu/ws/dinov2 python scripts/extract_video_features.py --demo --max-frames 20
PYTHONPATH=/home/m1zu/ws/dinov2 python scripts/extract_video_features.py --video <FILE> --fps 2 --output vid_feats.pt
```

## 关键API模式

```python
import torch
model = torch.hub.load('/home/m1zu/ws/dinov2', 'dinov2_vits14', source='local')
model.eval().cuda()

# CLS token: model(x) → [B, 384]
# Full: model.forward_features(x) → dict with x_norm_clstoken, x_norm_patchtokens
# Intermediate: model.get_intermediate_layers(x, n=4, return_class_token=True)
```

## 注意事项

- 使用HuggingFace库时必须 `ALL_PROXY= all_proxy=` 取消SOCKS代理
- DINOv3权重需要认证（目前不可用），已改用DINOv2
- xFormers警告可忽略（推理不需要）
- 可用模型: `dinov2_vits14/vitb14/vitl14/vitg14` + `_reg` 变体

## 待完成

- 数据选取pipeline（从N条数据中选subset，计算特征方差作为多样性指标）
- Pluggable API设计（视觉特征只是一种标准，需支持其他标准）
- 集成到RobotWin/OpenPI环境
