# Vision Transformer (ViT) from Scratch

PyTorch implementation of [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929).

📝 **Blog:** [Vision Transformer from Scratch: A Complete Guide]

## Components

All built from scratch — no `timm`, no `nn.TransformerEncoder`:

- Patch Embedding
- Positional Embedding (1D learnable)
- Multi-Head Self-Attention
- LayerNorm
- Transformer Encoder
- CLS Token Classification

## Usage

```python
from vit import VisionTransformer

model = VisionTransformer(
    patch_size=16,
    embed_size=768,
    image_size=224,
    num_classes=1000,
    num_heads=12,
    num_layers=12
)

x = torch.randn(4, 3, 224, 224)
output = model(x)  # (4, 1000)
```

## References

- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
