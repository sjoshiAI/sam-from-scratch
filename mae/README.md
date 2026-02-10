# Masked Autoencoder (MAE) from Scratch

PyTorch implementation of [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377).

## Components

All built from scratch

- Patch Embedding
- Positional Embedding (Sin-Cos)
- Multi-Head Self-Attention
- LayerNorm
- Transformer Encoder (Asymmetric)
- Transformer Decoder
- CLS Token
- Random Masking Strategy
- Reconstruction Head

## Architecture

MAE uses an asymmetric encoder-decoder architecture:
- **Encoder**: Processes only visible (unmasked) patches
- **Decoder**: Reconstructs original image from encoded visible patches + mask tokens
- High masking ratio (75%) enables efficient self-supervised learning

## Usage

```python
from mae import MAE

model = MAE(
    patch_size=16,
    encoder_embed_size=768,
    decoder_embed_size=512,
    image_size=224,
    in_channels=3,
    masking_ratio=0.75,
    num_encoder_layers=12,
    num_decoder_layers=8,
    encoder_num_heads=12,
    encoder_head_dim=64,
    decoder_num_heads=8,
    decoder_head_dim=64
)

x = torch.randn(4, 3, 224, 224)
reconstructions, mask_indices = model(x)  # (4, num_patches, patch_pixels), (4, num_masked)
```

## Training

```python
from mae import mae_loss, patchify

# Forward pass
reconstructions, mask_indices = model(images)

# Compute loss on masked patches only
target_patches = patchify(images, patch_size=16)
loss = mae_loss(reconstructions, target_patches, mask_indices)
```

## References

- [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
- [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929)
