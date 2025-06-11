import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from datetime import datetime
import time

try:
    from .model_v2 import DiffusionModelV2
    from .dataset_v2 import ImageDatasetV2
    from .utils import get_device, linear_beta_schedule, forward_noise
except ImportError:
    # For direct execution
    from model_v2 import DiffusionModelV2
    from dataset_v2 import ImageDatasetV2
    from utils import get_device, linear_beta_schedule, forward_noise

def train_diffusion_model_v2(
    dataset_root="H:/Repositories/ImgDatasetGen_256",
    epochs=50,
    batch_size=4,  # Per GPU
    accumulation_steps=4,  # Gradient accumulation
    learning_rate=1e-4,
    img_size=64,
    context_size=8,
    model_checkpoint_path=None,
    save_checkpoint_prefix="diffusion_v2_ckpt",
    save_interval=5,
    progress_callback=None,
    loss_callback=None,
    checkpoint_saved_callback=None,
    use_multi_gpu=True
):
    """
    Train diffusion model V2 with visual conditioning
    
    Args:
        dataset_root: Path to image dataset
        epochs: Number of training epochs
        batch_size: Batch size per GPU
        accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        img_size: Target image size (64)
        context_size: Visual context size (8)
        model_checkpoint_path: Path to resume from checkpoint
        save_checkpoint_prefix: Prefix for saved checkpoints
        save_interval: Save checkpoint every N epochs
        progress_callback: Callback for progress updates
        loss_callback: Callback for loss updates
        checkpoint_saved_callback: Callback when checkpoint is saved
        use_multi_gpu: Use DataParallel if multiple GPUs available
    """
    
    device = get_device()
    print(f"Training on device: {device}")
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    # Load dataset
    print("Loading dataset...")
    if progress_callback:
        progress_callback.emit("Loading dataset...", 0)
    
    dataset = ImageDatasetV2(dataset_root, img_size, context_size)
    
    # Adjust batch size for multi-GPU
    effective_batch_size = batch_size
    if use_multi_gpu and num_gpus > 1:
        effective_batch_size = batch_size * num_gpus
        print(f"Using DataParallel with {num_gpus} GPUs")
        print(f"Effective batch size: {effective_batch_size}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Initialize model
    print("Initializing model...")
    model = DiffusionModelV2(
        img_size=img_size,
        img_channels=3,
        latent_dim=256,
        time_dim=64,
        visual_context_dim=192,
        num_transformer_blocks=3,
        num_heads=4,
        initial_conv_filters=32,
        conv_dim_mults=(1, 2, 4, 8)
    )
    
    # Multi-GPU setup
    if use_multi_gpu and num_gpus > 1:
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # Load checkpoint if provided
    start_epoch = 0
    if model_checkpoint_path and os.path.exists(model_checkpoint_path):
        print(f"Loading checkpoint: {model_checkpoint_path}")
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Loss function
    loss_fn = nn.MSELoss()
    
    # Diffusion parameters (use same as utils.py)
    timesteps = 200  # Match TIMESTEPS in utils.py
    beta_schedule = linear_beta_schedule(timesteps)
    alphas = 1.0 - beta_schedule
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
    
    # Training loop
    print("Starting training...")
    model.train()
    
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        if progress_callback:
            progress_callback.emit(f"Epoch {epoch+1}/{epochs}", int((epoch/epochs)*100))
        
        # Reset gradients
        optimizer.zero_grad()
        
        for batch_idx, (main_images, context_images, _) in enumerate(dataloader):
            main_images = main_images.to(device)  # (B, 3, 64, 64)
            context_images = context_images.to(device)  # (B, 3, 8, 8)
            
            # Sample timesteps
            t = torch.randint(0, timesteps, (main_images.shape[0],), device=device)
            
            # Add noise to images
            x_t, noise = forward_noise(main_images, t, device)
            
            # Predict noise
            predicted_noise = model(x_t, t, context_images)
            
            # Calculate loss
            loss = loss_fn(predicted_noise, noise)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * accumulation_steps
            batch_count += 1
            
            # Progress within epoch
            if batch_idx % 100 == 0:
                progress_pct = int((epoch + (batch_idx/len(dataloader))) / epochs * 100)
                if progress_callback:
                    progress_callback.emit(
                        f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(dataloader)}", 
                        progress_pct
                    )
        
        # Final gradient step if needed
        if batch_count % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
        
        if loss_callback:
            loss_callback.emit(epoch + 1, avg_loss)
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = f"checkpoints/{save_checkpoint_prefix}_epoch{epoch+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            os.makedirs("checkpoints", exist_ok=True)
            
            # Save model state
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'model_config': {
                    'img_size': img_size,
                    'img_channels': 3,
                    'latent_dim': 256,
                    'time_dim': 64,
                    'visual_context_dim': 192,
                    'num_transformer_blocks': 3,
                    'num_heads': 4,
                    'initial_conv_filters': 32,
                    'conv_dim_mults': (1, 2, 4, 8)
                }
            }, checkpoint_path)
            
            print(f"Saved checkpoint: {checkpoint_path}")
            
            if checkpoint_saved_callback:
                checkpoint_saved_callback.emit(checkpoint_path)
    
    print("Training completed!")
    if progress_callback:
        progress_callback.emit("Training completed!", 100)
    
    return model

# Test training function
if __name__ == '__main__':
    print("Testing training setup...")
    
    # Mock callbacks for testing
    class MockCallback:
        def emit(self, *args):
            print(f"Callback: {args}")
    
    mock_callback = MockCallback()
    
    # Test with small parameters
    try:
        model = train_diffusion_model_v2(
            epochs=2,
            batch_size=2,
            accumulation_steps=2,
            save_interval=1,
            progress_callback=mock_callback,
            loss_callback=mock_callback,
            checkpoint_saved_callback=mock_callback,
            use_multi_gpu=True
        )
        print("✅ Training setup working!")
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
