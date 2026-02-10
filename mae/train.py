from mae import MAE, patchify, mae_loss
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from einops import rearrange
import os

def unpatchify(patches, patch_size, image_size):
    """
    Convert patches back to images

    Args:
        patches: (B, num_patches, patch_size*patch_size*C)
        patch_size: int
        image_size: int (assumes square images)

    Returns:
        images: (B, C, H, W)
    """
    patches_per_side = image_size // patch_size
    images = rearrange(
        patches,
        'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
        h=patches_per_side,
        w=patches_per_side,
        p1=patch_size,
        p2=patch_size,
        c=3
    )
    return images

def visualize_reconstruction(model, images, patch_size, image_size, epoch, device, save_dir='reconstructions'):
    """
    Visualize original images, masked images, and reconstructions
    """
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        # Get model predictions (model returns predictions for ALL patches and mask indices)
        pred, mask_indices = model(images)

        # Convert to patches
        target_patches = patchify(images, patch_size)

        # Create reconstruction that combines:
        # - Original visible patches (from keep_indices)
        # - Reconstructed masked patches (from pred at mask_indices)
        combined_patches = target_patches.clone()
        mask_indices_expanded = mask_indices.unsqueeze(-1).expand(-1, -1, patch_size*patch_size*3)
        pred_masked = torch.gather(pred, dim=1, index=mask_indices_expanded)
        combined_patches.scatter_(dim=1, index=mask_indices_expanded, src=pred_masked)

        # Unpatchify to images
        combined_images = unpatchify(combined_patches, patch_size, image_size)

        # Create visible-only version (masked patches are gray)
        visible_patches = target_patches.clone()
        visible_patches.scatter_(dim=1, index=mask_indices_expanded, value=0.0)
        visible_images = unpatchify(visible_patches, patch_size, image_size)

        # Denormalize for visualization (ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)

        images_vis = images * std + mean
        visible_images_vis = visible_images * std + mean
        combined_images_vis = combined_images * std + mean

        # Plot 8 examples
        num_examples = min(8, images.shape[0])
        fig, axes = plt.subplots(num_examples, 3, figsize=(9, 3*num_examples))

        for i in range(num_examples):
            # Original
            axes[i, 0].imshow(images_vis[i].cpu().permute(1, 2, 0).clamp(0, 1))
            axes[i, 0].set_title('Original' if i == 0 else '')
            axes[i, 0].axis('off')

            # Visible only (input to encoder)
            axes[i, 1].imshow(visible_images_vis[i].cpu().permute(1, 2, 0).clamp(0, 1))
            axes[i, 1].set_title('Visible (25%)' if i == 0 else '')
            axes[i, 1].axis('off')

            # Combined: visible + reconstructed
            axes[i, 2].imshow(combined_images_vis[i].cpu().permute(1, 2, 0).clamp(0, 1))
            axes[i, 2].set_title('Visible + Reconstructed' if i == 0 else '')
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/reconstruction_epoch_{epoch+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved reconstruction visualization to {save_dir}/reconstruction_epoch_{epoch+1}.png")

    model.train()

if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")

    # Hyperparameters for Food101 (128x128 images)
    patch_size = 16          # 128/16 = 8 patches per side → 64 total patches
    image_size = 128         # Resize Food101 to 128x128
    in_channels = 3          # RGB

    # Model architecture (medium size for Food101)
    encoder_embed_size = 512
    decoder_embed_size = 256
    num_encoder_layers = 8
    num_decoder_layers = 6
    encoder_num_heads = 8
    encoder_head_dim = 64    # 8 * 64 = 512
    decoder_num_heads = 8
    decoder_head_dim = 32    # 8 * 32 = 256
    masking_ratio = 0.75     # Mask 75% of patches (MAE default)

    # Training hyperparameters
    batch_size = 64          # Smaller batch for larger images
    learning_rate = 1e-3
    num_epochs = 50
    weight_decay = 0.05

    # Food101 transform (resize + normalize)
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Food101 dataset (auto-downloads!)
    print("Downloading Food101 dataset (this may take a while on first run - ~5GB)...")
    train_dataset = datasets.Food101(
        root='./data',
        split='train',  # 75,750 training images
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    # Initialize model
    model = MAE(
        patch_size=patch_size,
        encoder_embed_size=encoder_embed_size,
        decoder_embed_size=decoder_embed_size,
        image_size=image_size,
        in_channels=in_channels,
        masking_ratio=masking_ratio,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        encoder_num_heads=encoder_num_heads,
        encoder_head_dim=encoder_head_dim,
        decoder_num_heads=decoder_num_heads,
        decoder_head_dim=decoder_head_dim
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Optimizer (AdamW with weight decay like in MAE paper)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Get a fixed batch for visualization
    vis_images, _ = next(iter(train_loader))
    vis_images = vis_images[:8].to(device)  # Use first 8 images

    # Training loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, _) in enumerate(train_loader):
            # Move to device
            images = images.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            pred, mask_indices = model(images)

            # Compute loss
            target = patchify(images, patch_size)
            loss = mae_loss(pred, target, mask_indices)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

            # Print progress
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Print epoch summary
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_loss:.4f}\n")

        # Visualize reconstruction every 5 epochs or on the last epoch
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            visualize_reconstruction(model, vis_images, patch_size, image_size, epoch, device)

    print("Training completed!")

    # Save model
    torch.save(model.state_dict(), 'mae_food101.pth')
    print("Model saved to mae_food101.pth")     