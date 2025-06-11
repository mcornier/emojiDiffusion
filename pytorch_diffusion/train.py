import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from datetime import datetime

# Assuming model.py, dataset.py, utils.py are in the same directory (pytorch_diffusion)
from .model import DiffusionModel
from .dataset import EmojiDataset # Access global EMOJI_VOCAB for vocab size after dataset creation
from .utils import get_device, forward_noise, TIMESTEPS as DEFAULT_TIMESTEPS # Use TIMESTEPS from utils

# --- Configuration ---
IMG_SIZE = 16
IMG_CHANNELS = 3
LATENT_DIM = 256
TIME_DIM = 64
CONTEXT_DIM = 64
NUM_TRANSFORMER_BLOCKS = 3
NUM_HEADS = 4
INITIAL_CONV_FILTERS = 32
CONV_DIM_MULTS = (1, 2)

DEFAULT_FONT_PATH = None
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Training Loop ---
def train_diffusion_model(
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    sample_emojis: list[str] = None,
    font_path: str = DEFAULT_FONT_PATH,
    model_checkpoint_path: str = None,
    save_checkpoint_prefix: str = "diffusion_emoji_ckpt",
    save_interval: int = 10,
    progress_callback = None,
    loss_callback = None,
    checkpoint_saved_callback = None
):
    device = get_device()
    if progress_callback:
        progress_callback(f"Using device: {device}", 0)

    if sample_emojis is None:
        sample_emojis = ['😀', '😂', '😍', '👍', '🎉', '🚀', '🌟', '💡', '💻', '🤖', '🎨', '🎵']

    if progress_callback:
        progress_callback("Setting up dataset...", 1)
    emoji_dataset = EmojiDataset(emoji_list=sample_emojis, image_size=IMG_SIZE, font_path=font_path)
    dataloader = DataLoader(emoji_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    current_emoji_vocab_size = emoji_dataset.get_vocab_size()
    if progress_callback:
        progress_callback(f"Dataset created with vocabulary size: {current_emoji_vocab_size}", 2)

    if progress_callback:
        progress_callback("Initializing model...", 3)
    model = DiffusionModel(
        img_size=IMG_SIZE,
        img_channels=IMG_CHANNELS,
        latent_dim=LATENT_DIM,
        time_dim=TIME_DIM,
        context_dim=CONTEXT_DIM,
        emoji_vocab_size=current_emoji_vocab_size,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
        num_heads=NUM_HEADS,
        initial_conv_filters=INITIAL_CONV_FILTERS,
        conv_dim_mults=CONV_DIM_MULTS
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    start_epoch = 0
    if model_checkpoint_path and os.path.exists(model_checkpoint_path):
        if progress_callback:
            progress_callback(f"Loading checkpoint from {model_checkpoint_path}...", 4)
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if progress_callback:
            progress_callback(f"Resuming training from epoch {start_epoch}", 5)

    model.train()
    if progress_callback:
        progress_callback("Starting training...", 5)

    total_epochs_to_run = epochs - start_epoch
    if total_epochs_to_run <= 0:
        if progress_callback:
            progress_callback("Training already completed or no epochs to run.", 100)
        return model

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0

        # Calculate base percentage for the start of this epoch's progress segment
        # 5% initial setup, 90% for epochs, 5% for final wrap-up
        progress_base_for_epochs = 5
        progress_range_for_epochs = 90

        current_epoch_progress_start_perc = progress_base_for_epochs + \
            int(((epoch - start_epoch) / total_epochs_to_run) * progress_range_for_epochs)

        if progress_callback:
            progress_callback(f"Epoch {epoch+1}/{epochs} starting...", current_epoch_progress_start_perc)

        for i, (batch_images, batch_context_indices) in enumerate(dataloader):
            optimizer.zero_grad()
            x_0 = batch_images.to(device)
            context_idxs = batch_context_indices.to(device)
            t = torch.randint(0, DEFAULT_TIMESTEPS, (x_0.shape[0],), device=device).long()
            x_t, noise_added = forward_noise(x_0, t, device=device)
            predicted_noise = model(x_t, t, context_idxs)
            loss = criterion(predicted_noise, noise_added)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if progress_callback and (i + 1) % (max(1, len(dataloader) // 5)) == 0 : # Report ~5 times per epoch
                # Calculate percentage within the current epoch's allocated range
                perc_within_epoch_range = ((i + 1) / len(dataloader)) * (progress_range_for_epochs / total_epochs_to_run)
                current_total_perc = current_epoch_progress_start_perc + int(perc_within_epoch_range)
                current_total_perc = min(current_total_perc, progress_base_for_epochs + progress_range_for_epochs -1) # Cap before next epoch start
                progress_callback(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Batch Loss: {loss.item():.6f}", current_total_perc)

        avg_epoch_loss = epoch_loss / len(dataloader)

        if loss_callback:
            loss_callback(epoch + 1, avg_epoch_loss)

        current_epoch_finished_perc = progress_base_for_epochs + \
            int(((epoch - start_epoch + 1) / total_epochs_to_run) * progress_range_for_epochs)
        current_epoch_finished_perc = min(current_epoch_finished_perc, 99) # Cap before final 100%

        if progress_callback:
            progress_callback(f"Epoch {epoch+1}/{epochs} finished. Avg Loss: {avg_epoch_loss:.6f}", current_epoch_finished_perc)

        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_name = f"{save_checkpoint_prefix}_epoch{epoch+1}_{timestamp}.pth"
            ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'emoji_vocab_size': current_emoji_vocab_size,
                'img_size': IMG_SIZE,
                'latent_dim': LATENT_DIM
            }, ckpt_path)
            if checkpoint_saved_callback:
                checkpoint_saved_callback(ckpt_path)

    if progress_callback:
        progress_callback("Training finished.", 100)
    return model

if __name__ == '__main__':
    train_emojis = [
        '😀', '😁', '😂', '🤣', '😃', '😄', '😅', '😆', '😉', '😊',
        '😋', '😎', '😍', '😘', '🥰', '😗', '😙', '🥲', '🤔', '🤩',
        '🎨', '🎵', '💻', '💡', '🌟', '🚀', '🎉'
    ]

    def my_progress_cb(message, percentage):
        print(f"PROGRESS: {percentage}% - {message}")

    def my_loss_cb(epoch_num, loss_value):
        print(f"LOSS: Epoch {epoch_num}, Loss: {loss_value:.4f}")

    def my_checkpoint_cb(path):
        print(f"CHECKPOINT: Saved to {path}")

    print("Starting training script example with callbacks...")
    trained_model = train_diffusion_model(
        epochs=5,
        batch_size=4, # Smaller batch for faster example
        learning_rate=1e-4,
        sample_emojis=train_emojis[:5], # Smaller dataset for example
        font_path=DEFAULT_FONT_PATH,
        save_interval=2,
        progress_callback=my_progress_cb,
        loss_callback=my_loss_cb,
        checkpoint_saved_callback=my_checkpoint_cb
    )
    print("Example training script with callbacks finished.")
