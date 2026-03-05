# LAYERS — Neural Network Building Blocks

## OVERVIEW

Core NN components shared by ViT and ConvNeXt backbones. Attention, positional encoding, normalization, and projection layers.

## STRUCTURE

```
layers/
├── __init__.py              # Barrel exports all public classes
├── attention.py             # Multi-head self-attention (MemEffAttention, xformers/SDPA)
├── block.py                 # Transformer block (Block, NestedTensorBlock)
├── dino_head.py             # DINO/iBOT projection heads (DINOHead)
├── ffn_layers.py            # Feed-forward network variants (Mlp, SwiGLUFFN, SwiGLUFFNFused)
├── fp8_linear.py            # FP8 quantized linear layer
├── layer_scale.py           # Layer scale parameter (LayerScale)
├── patch_embed.py           # Patch embedding (PatchEmbed)
├── rms_norm.py              # RMSNorm (RMSNorm, FP32RMSNorm)
├── rope_position_encoding.py# Rotary Position Encoding (RoPE2D, PositionGetter)
└── sparse_linear.py         # Sparse linear layer for MoE-style routing
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Change attention implementation | `attention.py` | Supports xformers memory-efficient + PyTorch SDPA backends |
| Modify transformer block | `block.py` | **Caution: line 81 "do not" constraint** |
| Add positional encoding | `rope_position_encoding.py` | RoPE2D for 2D image patches |
| Change projection head | `dino_head.py` | Shared by DINO cls + iBOT patch heads |
| Add FFN variant | `ffn_layers.py` | SwiGLU is default for ViT-7B |
| FP8 training support | `fp8_linear.py` | Used in large model training |

## CONVENTIONS

- All layers are `nn.Module` subclasses with standard `forward()` signatures.
- `__init__.py` barrel-exports all public classes — add new layers there.
- Attention supports both `xformers.ops.memory_efficient_attention` and `torch.nn.functional.scaled_dot_product_attention`.
- NestedTensor support in `block.py` for variable-resolution inputs.

## ANTI-PATTERNS

- `block.py:81` — **"do not" constraint**: respect the stated limitation on block modification.
- Do not add `type: ignore` or `noqa` to this module.
