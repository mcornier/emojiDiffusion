import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from datetime import datetime

# Assuming model.py, dataset.py, utils.py are in the same directory (pytorch_diffusion)
from .model import DiffusionModel
from .dataset import EmojiDataset, EMOJI_VOCAB # Access global EMOJI_VOCAB for vocab size after dataset creation
from .utils import get_device, forward_noise, TIMESTEPS as DEFAULT_TIMESTEPS # Use TIMESTEPS from utils

# --- Configuration ---
# These would typically be arguments to a main function or loaded from a config file
IMG_SIZE = 16
IMG_CHANNELS = 3
LATENT_DIM = 256
TIME_DIM = 64
CONTEXT_DIM = 64
# EMOJI_VOCAB_SIZE will be determined from the dataset
NUM_TRANSFORMER_BLOCKS = 3
NUM_HEADS = 4
INITIAL_CONV_FILTERS = 32
CONV_DIM_MULTS = (1, 2) # Matches model.py default for 16x16

DEFAULT_FONT_PATH = None # User might need to set this, e.g., "path/to/NotoColorEmoji.ttf"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Training Loop ---
def train_diffusion_model(
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4, # Adjusted from 2e-4 in JS
    sample_emojis: list[str] = None,
    font_path: str = DEFAULT_FONT_PATH,
    model_checkpoint_path: str = None, # Path to load a checkpoint
    save_checkpoint_prefix: str = "diffusion_emoji_ckpt",
    save_interval: int = 10 # Save checkpoint every N epochs
):
    """
    Trains the diffusion model.
    """
    device = get_device()
    print(f"Using device: {device}")

    if sample_emojis is None:
        sample_emojis = ['😀', '😂', '😍', '👍', '🎉', '🚀', '🌟', '💡', '💻', '🤖', '🎨', '🎵']

    # 1. Dataset
    print("Setting up dataset...")
    emoji_dataset = EmojiDataset(emoji_list=sample_emojis, image_size=IMG_SIZE, font_path=font_path)
    dataloader = DataLoader(emoji_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # num_workers=0 for simplicity

    # Update EMOJI_VOCAB_SIZE based on the created dataset
    current_emoji_vocab_size = emoji_dataset.get_vocab_size()
    print(f"Dataset created with vocabulary size: {current_emoji_vocab_size}")

    # 2. Model
    print("Initializing model...")
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

    # 3. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate) # AdamW is common for transformers
    criterion = nn.MSELoss() # To predict the noise (epsilon)

    start_epoch = 0
    # Load checkpoint if provided
    if model_checkpoint_path and os.path.exists(model_checkpoint_path):
        print(f"Loading checkpoint from {model_checkpoint_path}...")
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        # Note: emoji_vocab_size in checkpoint should ideally match current.
        # For simplicity, we assume it does or that the user handles this.
        print(f"Resuming training from epoch {start_epoch}")

    model.train()
    print("Starting training...")

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        for i, (batch_images, batch_context_indices) in enumerate(dataloader):
            optimizer.zero_grad()

            current_batch_size = batch_images.shape[0]

            # Move data to device
            x_0 = batch_images.to(device) # Clean images
            context_idxs = batch_context_indices.to(device) # Context indices

            # Sample random timesteps for each image in the batch
            # DEFAULT_TIMESTEPS is from utils.py (e.g., 200)
            t = torch.randint(0, DEFAULT_TIMESTEPS, (current_batch_size,), device=device).long()

            # Add noise to images (forward process)
            x_t, noise_added = forward_noise(x_0, t, device=device)

            # Predict noise using the model
            # Model expects time tensor `t` and context indices `context_idxs`
            predicted_noise = model(x_t, t, context_idxs)

            # Calculate loss
            loss = criterion(predicted_noise, noise_added)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (i + 1) % (len(dataloader) // 2 if len(dataloader) > 1 else 1) == 0 : # Log a few times per epoch
                 print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.6f}")

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] completed. Average Loss: {avg_epoch_loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_name = f"{save_checkpoint_prefix}_epoch{epoch+1}_{timestamp}.pth"
            ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'emoji_vocab_size': current_emoji_vocab_size, # Save vocab size for reference
                'img_size': IMG_SIZE, # Save key model params
                'latent_dim': LATENT_DIM
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    print("Training finished.")
    return model # Return the trained model

# --- Main execution ---
if __name__ == '__main__':
    # These are example emojis. Create a more diverse and larger list for actual training.
    # Ensure your font (DEFAULT_FONT_PATH) supports these.
    train_emojis = [
        '😀', '😁', '😂', '🤣', '😃', '😄', '😅', '😆', '😉', '😊',
        '😋', '😎', '😍', '😘', '🥰', '😗', '😙', '🥲', '🤔', '🤩',
        '🤗', '🙂', '😚', '🤨', '😐', '😑', '😶', '🫥', '😮', '😥',
        '😣', '😏', '🙄', '😤', '😠', '😡', '🤬', '🤯', '🥵', '🥶',
        '😳', '🤪', '😵', '😵‍💫', '😲', '😱', '😨', '😰', '😢', '😭',
        '😮‍💨', '🥱', '😴', '🤤', '😪', '🤢', '🤮', '🤧', '😇', '🤠',
        '🥳', '🥺', '🥸', '🧐', '😕', '😟', '🙁', '😮', '😯', '😲',
        '😳', '🥺', '😦', '😧', '😨', '😰', '😥', '😢', '😭', '😱',
        '😖', '😣', '😞', '😓', '😩', '😫', '🥱', '😤', '😡', '😠',
        '🤬', '😈', '👿', '💀', '☠️', '💩', '🤡', '👹', '👺', '👻',
        '👽', '👾', '🤖', '😺', '😸', '😹', '😻', '😼', '😽', '🙀',
        '😿', '😾', '🙈', '🙉', '🙊', '👋', '🤚', '🖐️', '✋', '🖖',
        '👌', '🤌', '🤏', '✌️', '🤞', '🤟', '🤘', '🤙', '👈', '👉',
        '👆', '🖕', '👇', '☝️', '👍', '👎', '✊', '👊', '🤛', '🤜',
        '👏', '🙌', '🫶', '👐', '🤲', '🤝', '🙏', '✍️', '💅', '🤳',
        '💪', '🦾', '🦿', '🦵', '🦶', '👂', '🦻', '👃', '🧠', '🫀',
        '🫁', '🦷', '🦴', '👀', '👁️', '👅', '👄', '👶', '🧒', '🧑',
        '🎨', '🎵', '💻', '💡', '🌟', '🚀', '🎉' # Some objects/symbols
    ]

    print("Starting training script example...")
    # To resume from a checkpoint, set model_checkpoint_path, e.g.:
    # trained_model = train_diffusion_model(epochs=200, batch_size=64, sample_emojis=train_emojis, model_checkpoint_path="checkpoints/diffusion_emoji_ckpt_epoch50_xxxx.pth")
    trained_model = train_diffusion_model(
        epochs=50, # Adjust epochs as needed
        batch_size=64, # Adjust batch size based on memory
        learning_rate=1e-4,
        sample_emojis=train_emojis,
        font_path=DEFAULT_FONT_PATH, # Set this if your system default font doesn't render emojis well
        save_interval=10
    )
    print("Example training script finished.")

    # After training, you can use the 'trained_model' for inference.
    # For example, using p_sample_loop from utils.py:
    # from .utils import p_sample_loop, tensor_to_pil
    # if trained_model:
    #     print("Generating a sample image with the trained model...")
    #     trained_model.eval() # Set model to evaluation mode
    #     shape = (1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE) # Batch size 1
    #     # For unconditional generation, context_embedding can be None or a MASK token embedding
    #     # For conditional, get the context_embedding for a specific emoji index
    #     # Example: unconditional or MASK
    #     mask_idx = torch.tensor([EMOJI_TO_INDEX['[MASK]']], device=get_device())
    #     # The model's context_embedding_layer and context_mlp handle the index to embedding
    #     # So, for p_sample_loop, we need to pass the context_idx if model expects it,
    #     # or pre-compute the embedding if p_sample_loop expects the embedding directly.
    #     # The p_sample in utils.py expects context_embedding.
    #     # Let's assume for now p_sample_loop is adapted or we pass the index.
    #     # For simplicity, if p_sample_loop handles context_idx:
    #     # generated_img_tensor, _ = p_sample_loop(trained_model, shape, context_idx_for_generation=mask_idx)
    #     # For now, let's assume unconditional for this example snippet as p_sample_loop takes context_embedding
    #     # To do this properly, one would need to get the embedding from model.context_embedding_layer(mask_idx) etc.
    #     # The p_sample_loop in utils.py expects context_embedding, not index.
    #     # So, if conditional:
    #     #   cond_idx = torch.tensor([EMOJI_TO_INDEX['😀']], device=get_device())
    #     #   cond_emb = trained_model.context_embedding_layer(cond_idx)
    #     #   cond_emb = trained_model.context_mlp(cond_emb)
    #     #   generated_img_tensor, _ = p_sample_loop(trained_model, shape, context_embedding=cond_emb)
    #     # Else for unconditional (if model supports it by passing None to context_embedding):
    #     #   generated_img_tensor, _ = p_sample_loop(trained_model, shape, context_embedding=None)
    #     #
    #     # tensor_to_pil(generated_img_tensor.squeeze(0)).save("trained_sample.png")
    #     # print("Saved trained_sample.png")
