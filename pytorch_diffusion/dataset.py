import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF
import os

# --- Global Emoji Vocabulary ---
# These will be managed by functions, but defined here for module-level access if needed.
EMOJI_VOCAB = ['[MASK]'] # Start with MASK token, similar to JS
EMOJI_TO_INDEX = {'[MASK]': 0}

# --- Dataset Generation ---
def update_emoji_vocab(emoji_list):
    """
    Updates the global EMOJI_VOCAB and EMOJI_TO_INDEX based on a list of emojis.
    Returns the new vocab and index map.
    """
    global EMOJI_VOCAB, EMOJI_TO_INDEX

    # Reset and add MASK token first
    current_vocab = ['[MASK]']
    current_emoji_to_index = {'[MASK]': 0}

    # Add unique emojis from the input list
    for emoji in emoji_list:
        if emoji not in current_emoji_to_index:
            current_emoji_to_index[emoji] = len(current_vocab)
            current_vocab.append(emoji)

    EMOJI_VOCAB = current_vocab
    EMOJI_TO_INDEX = current_emoji_to_index
    return EMOJI_VOCAB, EMOJI_TO_INDEX

def generate_emoji_image(emoji: str, image_size: int, font_path: str = None):
    """
    Generates a PIL Image of an emoji.

    Args:
        emoji (str): The emoji character to render.
        image_size (int): The size (width and height) of the output image.
        font_path (str, optional): Path to a .ttf font file.
                                   If None, attempts to use a default system font.
    Returns:
        PIL.Image: The generated image.
    """
    image = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(image)
    # Debug: dessiner un carré de bordure pour visualiser le cadre
    draw.rectangle([0, 0, image_size-1, image_size-1], outline="red")

    try:
        import platform
        font_size = int(image_size * 0.5)
        font_loaded = False
        tried_fonts = []

        # 1. User-provided font
        if font_path and os.path.exists(font_path):
            print(f"[DEBUG] generate_emoji_image: Trying font_path={font_path} font_size={font_size}")
            try:
                font = ImageFont.truetype(font_path, font_size)
                print(f"[DEBUG] generate_emoji_image: Loaded font {font_path} successfully")
                font_loaded = True
            except Exception as e:
                print(f"[DEBUG] generate_emoji_image: Failed to load user font {font_path}: {e}")
                tried_fonts.append(font_path)

        # 2. System emoji fonts by OS
        if not font_loaded:
            system = platform.system()
            font_candidates = []
            if system == "Windows":
                font_candidates = [
                    "seguiemj.ttf",  # Segoe UI Emoji
                    "arial.ttf"
                ]
                font_dirs = [
                    os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
                ]
            elif system == "Darwin":
                font_candidates = [
                    "Apple Color Emoji.ttc",
                    "Arial Unicode.ttf"
                ]
                font_dirs = [
                    "/System/Library/Fonts",
                    "/Library/Fonts"
                ]
            else:  # Assume Linux
                font_candidates = [
                    "NotoColorEmoji.ttf",
                    "DejaVuSans.ttf"
                ]
                font_dirs = [
                    "/usr/share/fonts/truetype/noto",
                    "/usr/share/fonts/truetype/dejavu",
                    "/usr/share/fonts/truetype/freefont",
                    "/usr/share/fonts"
                ]
            # Try to find and load a font
            for font_name in font_candidates:
                for font_dir in font_dirs:
                    font_file = os.path.join(font_dir, font_name)
                    if os.path.exists(font_file):
                        tried_fonts.append(font_file)
                        try:
                            font = ImageFont.truetype(font_file, font_size)
                            print(f"[DEBUG] generate_emoji_image: Loaded system font {font_file} successfully")
                            font_loaded = True
                            break
                        except Exception as e:
                            print(f"[DEBUG] generate_emoji_image: Failed to load system font {font_file}: {e}")
                if font_loaded:
                    break

        # 3. Fallbacks
        if not font_loaded:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
                print(f"[DEBUG] generate_emoji_image: Loaded fallback arial.ttf")
                font_loaded = True
            except Exception:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                    print(f"[DEBUG] generate_emoji_image: Loaded fallback DejaVuSans.ttf")
                    font_loaded = True
                except Exception:
                    font = ImageFont.load_default()
                    print(f"[DEBUG] generate_emoji_image: Loaded PIL default font. Tried: {tried_fonts}")

        # Get text dimensions using textbbox (Pillow 8.0.0+) or textsize (older)
        if hasattr(draw, 'textbbox'): # Pillow 8.0.0+
            # xy argument for textbbox is the top-left corner of the text. (0,0) is fine for getting size.
            # Anchor 'mm' for text might be better if available with the font for centering.
            bbox = draw.textbbox((0, 0), emoji, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            # Calculate position to center the text (using bbox[0] and bbox[1] as offsets from (0,0))
            x = (image_size - text_width) / 2 - bbox[0]
            y = (image_size - text_height) / 2 - bbox[1]
        else: # Older Pillow versions
            text_width, text_height = draw.textsize(emoji, font=font)
            x = (image_size - text_width) / 2
            y = (image_size - text_height) / 2

        try:
            draw.text((x, y), emoji, fill="black", font=font)
            print(f"[DEBUG] generate_emoji_image: Drew emoji '{emoji}' at ({x},{y}) with font {font_path}")
        except Exception as e:
            print(f"[DEBUG] generate_emoji_image: ERROR drawing emoji '{emoji}' at ({x},{y}) with font {font_path}: {e}")

    except Exception as e:
        print(f"Error loading font or drawing emoji {emoji}: {e}. Drawing simple text.")
        # Fallback if font loading fails catastrophically
        fallback_font_size = image_size // 2
        fallback_font = ImageFont.load_default() # Should always work
        text_width, text_height = draw.textsize(emoji, font=fallback_font)
        x = (image_size - text_width) / 2
        y = (image_size - text_height) / 2
        draw.text((x, y), emoji, fill="black", font=fallback_font)

    return image

def create_emoji_dataset(emoji_list: list[str], image_size: int, font_path: str = None):
    """
    Creates a dataset of emoji images and their corresponding context indices.

    Args:
        emoji_list (list[str]): A list of unique emoji characters.
        image_size (int): The size of the images to generate.
        font_path (str, optional): Path to a .ttf font file for rendering emojis.

    Returns:
        tuple: (images_tensor, context_indices_tensor, updated_vocab, updated_emoji_to_index)
               - images_tensor: A PyTorch tensor of shape (N, C, H, W), normalized to [-1, 1].
               - context_indices_tensor: A PyTorch tensor of shape (N,) with context indices.
               - updated_vocab: The updated EMOJI_VOCAB list.
               - updated_emoji_to_index: The updated EMOJI_TO_INDEX map.
    """
    updated_vocab, updated_emoji_to_index = update_emoji_vocab(emoji_list)

    print(f"[DEBUG] create_emoji_dataset: font_path={font_path}, emoji_list_len={len(emoji_list)}")

    pil_images = []
    context_indices = []

    for emoji_char in emoji_list: # Iterate through the user-provided list to maintain order and get indices
        if emoji_char in updated_emoji_to_index: # Should always be true due to update_emoji_vocab
            pil_img = generate_emoji_image(emoji_char, image_size, font_path)
            pil_images.append(pil_img)
            context_indices.append(updated_emoji_to_index[emoji_char])
        else:
            print(f"Warning: Emoji '{emoji_char}' not found in updated vocabulary. Skipping.")


    if not pil_images:
        print(f"[DEBUG] create_emoji_dataset: NO IMAGES GENERATED for font_path={font_path}")
        # Return empty tensors and current vocab if no images were generated
        return torch.empty(0, 3, image_size, image_size), torch.empty(0, dtype=torch.long), updated_vocab, updated_emoji_to_index

    # Convert PIL images to tensor and normalize
    # Stack expects a list of tensors of the same size.
    # TF.to_tensor converts a PIL Image or numpy.ndarray to tensor (scales to [0,1]).
    # Then normalize to [-1, 1].
    img_tensors = [TF.to_tensor(img) * 2.0 - 1.0 for img in pil_images]
    images_tensor = torch.stack(img_tensors) # (N, C, H, W)

    context_indices_tensor = torch.tensor(context_indices, dtype=torch.long) # (N,)

    return images_tensor, context_indices_tensor, updated_vocab, updated_emoji_to_index


# --- PyTorch Dataset Class (Optional but good practice) ---
class EmojiDataset(torch.utils.data.Dataset):
    def __init__(self, emoji_list: list[str], image_size: int, font_path: str = None):
        """
        Args:
            emoji_list (list[str]): List of emoji characters to form the dataset.
            image_size (int): Size of the generated emoji images.
            font_path (str, optional): Path to a .ttf font for rendering.
        """
        super().__init__()
        self.image_size = image_size
        self.font_path = font_path

        # Generate the dataset immediately on instantiation
        self.images, self.context_indices, self.vocab, self.emoji_to_idx = \
            create_emoji_dataset(emoji_list, image_size, font_path)

        self.EMOJI_VOCAB = self.vocab
        self.EMOJI_TO_INDEX = self.emoji_to_idx

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        context_idx = self.context_indices[idx]
        # Return image and its context label. Timestep t will be chosen during training.
        return image, context_idx

    def get_vocab_size(self):
        return len(self.EMOJI_VOCAB)

# --- Example Usage ---
if __name__ == '__main__':
    sample_emojis = ['😀', '😂', '😍', '👍', '🎉', '🚀']
    img_size = 16 # For diffusion model, typically small like 16x16 or 32x32

    # You might need to provide a path to a font that supports these emojis.
    # Common Noto fonts: "NotoColorEmoji.ttf", "NotoSansSymbols-Regular.ttf"
    # On Linux, common path: /usr/share/fonts/truetype/noto/NotoColorEmoji.ttf
    # On Windows: "seguiemj.ttf" (Segoe UI Emoji) usually in C:\Windows\Fonts
    # For testing, we'll let it try defaults or use a placeholder if you don't have one handy.
    font_file_path = None # Replace with actual path if needed, e.g., "NotoColorEmoji.ttf"

    print(f"Attempting to use font: {font_file_path if font_file_path else 'System Defaults'}")

    # Test direct dataset creation
    images_tensor, context_indices_tensor, vocab, emoji_to_idx = \
        create_emoji_dataset(sample_emojis, img_size, font_path=font_file_path)

    print(f"Generated images tensor shape: {images_tensor.shape}")
    print(f"Context indices tensor shape: {context_indices_tensor.shape}")
    print(f"Vocabulary (size {len(vocab)}): {vocab}")
    print(f"Emoji to Index map: {emoji_to_idx}")

    if images_tensor.numel() > 0:
        print(f"Image tensor min/max: {images_tensor.min()}, {images_tensor.max()}")
        # Save one image for visual inspection
        try:
            from torchvision.utils import save_image
            save_image((images_tensor[0] + 1) / 2, "sample_emoji_from_create_dataset.png") # Denormalize for saving
            print("Saved sample_emoji_from_create_dataset.png")
        except ImportError:
            print("torchvision.utils.save_image not available. Skipping image save test.")
            # Alternative: convert to PIL and save
            # TF.to_pil_image((images_tensor[0] + 1) / 2).save("sample_emoji_pil.png")


    # Test PyTorch Dataset class
    print("\nTesting EmojiDataset class...")
    emoji_pytorch_dataset = EmojiDataset(sample_emojis, img_size, font_path=font_file_path)
    print(f"Dataset size: {len(emoji_pytorch_dataset)}")
    print(f"Dataset vocab size: {emoji_pytorch_dataset.get_vocab_size()}")

    if len(emoji_pytorch_dataset) > 0:
        img, ctx_idx = emoji_pytorch_dataset[0]
        print(f"Sample from dataset - Image shape: {img.shape}, Context index: {ctx_idx}")

        # Test with DataLoader
        try:
            from torch.utils.data import DataLoader
            dataloader = DataLoader(emoji_pytorch_dataset, batch_size=2, shuffle=True)
            batch_images, batch_ctx_indices = next(iter(dataloader))
            print(f"DataLoader batch images shape: {batch_images.shape}")
            print(f"DataLoader batch context indices: {batch_ctx_indices}")
        except ImportError:
            print("torch.utils.data.DataLoader not available for full test.")

    # Example of how vocab might be updated and used by the model
    # model_vocab_size = emoji_pytorch_dataset.get_vocab_size()
    # diffusion_model = DiffusionModel(..., emoji_vocab_size=model_vocab_size, ...)
    # print(f"\nModel would be initialized with emoji_vocab_size = {model_vocab_size}")
