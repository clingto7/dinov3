# 2026-03-05 DINOv2 环境搭建与特征提取脚本

## 任务概述

用户希望使用DINOv2模型熟悉视觉特征提取，最终目标是为机器人学习(RobotWin/OpenPI)搭建数据选取pipeline。本次工作完成了DINOv2环境搭建和两个特征提取脚本的编写与验证。

## 工作内容

### 1. DINOv3 AGENTS.md 生成
- 为DINOv3仓库生成了7个层级化AGENTS.md文件
- 根目录 + eval子目录 + layers子目录

### 2. DINOv2 仓库克隆与AGENTS.md
- 从 `git@github.com:clingto7/dinov2.git` 克隆到 `/home/m1zu/ws/dinov2/`
- 生成了 `/home/m1zu/ws/dinov2/AGENTS.md`（148行），格式与dinov3一致，包含GUIDELINES部分

### 3. Python环境搭建
- 使用 `uv venv --python 3.12` 在 `/home/m1zu/ws/dinov3/.venv/` 创建虚拟环境
- 安装依赖：PyTorch 2.10.0+cu128, torchvision, transformers, omegaconf, scikit-learn, opencv-python-headless, requests
- GPU验证：RTX 4060 Laptop GPU，CUDA可用

### 4. DINOv2 权重与API验证
- 确认DINOv2权重公开可访问（无需auth），HTTP 200
- 下载 ViT-S/14 权重（84MB）到 `~/.cache/torch/hub/checkpoints/`
- 验证三种API方法：
  - `model(x)` → CLS输出 `[1, 384]`
  - `model.forward_features(x)` → CLS + patch tokens
  - `model.get_intermediate_layers(x, n=4)` → 中间层特征

### 5. 图像特征提取脚本
- **文件**: `scripts/extract_image_features.py`
- **功能**:
  - 加载DINOv2模型（支持所有变体 vits14/vitb14/vitl14/vitg14 + _reg）
  - 从目录/文件/demo URL加载图片
  - 提取CLS token、patch tokens、中间层特征
  - 计算pairwise余弦相似度矩阵
  - 计算数据集级别特征方差（数据多样性指标）
  - 支持保存features到.pt文件
- **测试结果**: 成功下载5张COCO demo图片，提取所有特征类型，输出正确

### 6. 视频特征提取脚本
- **文件**: `scripts/extract_video_features.py`
- **功能**:
  - 从视频文件采样帧（可控FPS和最大帧数）
  - 批量提取每帧CLS和patch特征
  - 计算时序统计（特征方差、帧间余弦相似度、场景变化检测）
  - 计算空间统计（每帧patch方差 = 场景复杂度）
  - 支持中间层特征提取
  - 支持保存features到.pt文件
- **测试结果**: 成功下载BigBuckBunny demo视频（64.7MB），采样20帧，提取所有特征，输出正确

## 使用方法

```bash
# 激活环境
source /home/m1zu/ws/dinov3/.venv/bin/activate

# 图像特征提取 - demo模式
PYTHONPATH=/home/m1zu/ws/dinov2 python scripts/extract_image_features.py --demo

# 图像特征提取 - 指定目录
PYTHONPATH=/home/m1zu/ws/dinov2 python scripts/extract_image_features.py --image-dir /path/to/images

# 图像特征提取 - 保存结果 + 中间层
PYTHONPATH=/home/m1zu/ws/dinov2 python scripts/extract_image_features.py --demo --output features.pt --intermediate-layers 4

# 视频特征提取 - demo模式
PYTHONPATH=/home/m1zu/ws/dinov2 python scripts/extract_video_features.py --demo --max-frames 20

# 视频特征提取 - 指定视频
PYTHONPATH=/home/m1zu/ws/dinov2 python scripts/extract_video_features.py --video /path/to/video.mp4 --fps 2

# 使用更大模型
PYTHONPATH=/home/m1zu/ws/dinov2 python scripts/extract_image_features.py --demo --model dinov2_vitb14
```

## 关键发现

1. **DINOv3权重需要认证** — Meta下载链接403, HuggingFace 401 (Gated)
2. **DINOv2权重公开可用** — `dl.fbaipublicfiles.com/dinov2/` 无需认证
3. **代理问题** — `ALL_PROXY=socks://` 需要在使用HuggingFace时取消设置
4. **xFormers非必需** — 推理/特征提取不需要xFormers，只有训练需要
5. **DINOv2 patch_size=14**（非16） — 224×224输入产生16×16=256个patch

## 修改的文件

### 新增
- `/home/m1zu/ws/dinov3/AGENTS.md` 及6个子目录AGENTS.md
- `/home/m1zu/ws/dinov2/AGENTS.md`
- `/home/m1zu/ws/dinov3/scripts/extract_image_features.py`
- `/home/m1zu/ws/dinov3/scripts/extract_video_features.py`
- `/home/m1zu/ws/dinov3/.venv/` (Python 3.12虚拟环境)

## 后续工作方向

- 搭建数据选取pipeline：从200条数据中选100条subset，计算每个subset的特征方差
- 建立pluggable API，支持不同数据选取标准
- 将特征提取集成到RobotWin/OpenPI环境中
