# SAM from Scratch

Building toward [Segment Anything (SAM)](https://arxiv.org/abs/2304.02643) by implementing its components from scratch in PyTorch.

## Progress

| Model | Status | Blog |
|-------|--------|------|
| [ViT](./vit-model/) | ✅ Complete |[A comprehensive guide to building Vision Transformers from Scratch](https://levelup.gitconnected.com/a-comprehensive-guide-to-building-vision-transformers-from-scratch-b65546f6183d)|
| MAE | ✅ Complete | Coming soon |
| CLIP | 🔄 In progress | — |
| SAM | ⬜ Planned | — |

## Structure

```
├── vit-model/     # Vision Transformer
├── mae/           # Masked Autoencoder
├── clip/          # CLIP
└── sam/           # Segment Anything
```

## Requirements

```
torch
einops
```
