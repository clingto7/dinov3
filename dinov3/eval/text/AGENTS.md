# TEXT — Text Alignment (dino.txt)

## OVERVIEW

CLIP-style text alignment for DINOv3 features using the dino.txt method. Trains a text encoder to align with frozen DINOv3 vision features for zero-shot tasks.

## STRUCTURE

```
text/
├── train_dinotxt.py     # Entry point (main + __main__)
├── build_dinotxt.py     # Model factory for dino.txt
├── dinotxt_model.py     # Full dino.txt model (vision + text towers)
├── vision_tower.py      # DINOv3 vision encoder wrapper — FIXME at line 75
├── text_tower.py        # Text encoder tower
├── text_transformer.py  # Text transformer architecture
├── tokenizer.py         # BPE tokenizer (uses vendored CLIP vocab)
├── clip_loss.py         # CLIP contrastive loss
├── gram_loss.py         # Gram matrix loss for text alignment
├── ac_comp_parallelize.py # Activation checkpointing for text training
└── configs/
    └── dinov3_vitl_text.yaml  # Example config (CocoCaptions dataset)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Modify text model | `dinotxt_model.py` | Full model combining vision + text |
| Change text encoder | `text_tower.py` + `text_transformer.py` | Text-side architecture |
| Change vision wrapper | `vision_tower.py` | Wraps frozen DINOv3 backbone |
| Add text loss | `clip_loss.py` or `gram_loss.py` | CLIP contrastive + Gram |
| Tokenizer | `tokenizer.py` | Uses `thirdparty/CLIP/clip/` vocab file |
| Train text alignment | `train_dinotxt.py` | 4-node default (32 GPUs) |

## CONVENTIONS

- Vision backbone is **frozen** during text alignment training.
- Tokenizer depends on vendored CLIP vocabulary at `dinov3/thirdparty/CLIP/clip/bpe_simple_vocab_16e6.txt.gz` (or downloaded separately).
- Example config uses CocoCaptions dataset — paper used private dataset.

## ANTI-PATTERNS

- `vision_tower.py:75` — FIXME: known issue to address.
- `ac_comp_parallelize.py` duplicates pattern from `dinov3/fsdp/ac_compile_parallelize.py` — task-specific variant.
