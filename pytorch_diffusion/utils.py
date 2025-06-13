import torch
import torchvision
from PIL import Image
import numpy as np

# --- Device Selection ---
def get_device():
    """Get the best available device for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# --- Diffusion Schedule ---
TIMESTEPS = 200

def linear_beta_schedule(timesteps):
    """Create a linear beta schedule for the diffusion process."""
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

# --- Diffusion Schedule Constants ---
BETAS = linear_beta_schedule(TIMESTEPS)
ALPHAS = 1. - BETAS
ALPHAS_CUMPROD = torch.cumprod(ALPHAS, axis=0)
ALPHAS_CUMPROD_PREV = torch.nn.functional.pad(ALPHAS_CUMPROD[:-1], (1, 0), value=1.0)

# For q_sample (forward_noise)
SQRT_ALPHAS_CUMPROD = torch.sqrt(ALPHAS_CUMPROD)
SQRT_ONE_MINUS_ALPHAS_CUMPROD = torch.sqrt(1. - ALPHAS_CUMPROD)

# For q_posterior (denoising step)
POSTERIOR_VARIANCE = BETAS * (1. - ALPHAS_CUMPROD_PREV) / (1. - ALPHAS_CUMPROD)

def forward_noise(x_0, t, device):
    """
    Add noise to clean images according to the diffusion schedule.
    
    Args:
        x_0: Clean images (B, C, H, W)
        t: Timesteps (B,)
        device: Device to use
    
    Returns:
        x_t: Noisy images
        noise: The noise that was added
    """
    noise = torch.randn_like(x_0)
    
    sqrt_alphas_cumprod_t = SQRT_ALPHAS_CUMPROD.to(device)[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = SQRT_ONE_MINUS_ALPHAS_CUMPROD.to(device)[t].view(-1, 1, 1, 1)
    
    x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    return x_t, noise

def tensor_to_pil(image_tensor: torch.Tensor):
    """Converts a [0, 1] normalized tensor to a PIL Image."""
    # Tensor is already in [0, 1], no conversion needed!
    image_tensor = image_tensor.clamp(0, 1)
    # If batch, take the first image
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]
    return torchvision.transforms.ToPILImage()(image_tensor.cpu())

def pil_to_tensor(pil_image, target_size=None):
    """Converts a PIL Image to a [0, 1] normalized tensor."""
    if target_size:
        pil_image = pil_image.resize(target_size, Image.LANCZOS)
    tensor = torchvision.transforms.ToTensor()(pil_image)
    if tensor.shape[0] == 1:  # Grayscale to RGB
        tensor = tensor.repeat(3, 1, 1)
    elif tensor.shape[0] == 4:  # RGBA to RGB
        tensor = tensor[:3]
    return tensor.unsqueeze(0)  # Add batch dim - already [0, 1]!

@torch.no_grad()
def p_sample(model, x_t: torch.Tensor, t: torch.Tensor, t_index: int, visual_context: torch.Tensor):
    """
    Samples x_{t-1} from x_t using the model's prediction of the noise.
    
    Args:
        model: The DiffusionModel
        x_t: The current noised image (B, 3, 64, 64)
        t: The current timestep (scalar tensor)
        t_index: The integer index for t
        visual_context: Visual conditioning (B, 3, 8, 8)
    Returns:
        x_prev: The denoised image x_{t-1}
    """
    device = x_t.device
    betas_t = BETAS.to(device)[t_index]
    sqrt_one_minus_alphas_cumprod_t = SQRT_ONE_MINUS_ALPHAS_CUMPROD.to(device)[t_index]
    sqrt_recip_alphas_t = torch.sqrt(1.0 / ALPHAS.to(device)[t_index])

    # Model predicts the noise (epsilon)
    predicted_noise = model(x_t, t, visual_context)

    # Equation 11 from DDPM paper:
    # x_{t-1} = 1/sqrt(alpha_t) * (x_t - (beta_t / sqrt(1-alpha_cumprod_t)) * epsilon_t) + sigma_t * z
    model_mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean  # No noise added at the last step
    else:
        posterior_variance_t = POSTERIOR_VARIANCE.to(device)[t_index]
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, visual_context: torch.Tensor, num_timesteps: int = TIMESTEPS, device: torch.device = None):
    """
    Generates a 64x64 image by sampling from the diffusion model, starting from random noise.
    
    Args:
        model: The DiffusionModel
        visual_context: Visual conditioning (B, 3, 8, 8)
        num_timesteps: The total number of diffusion timesteps
        device: The device to perform calculations on
    Returns:
        img: The generated image (B, 3, 64, 64)
        imgs_history: List of PIL images showing the denoising process
    """
    if device is None:
        device = get_device()

    model.to(device)
    model.eval()
    
    batch_size = visual_context.shape[0]
    shape = (batch_size, 3, 64, 64)  # 64x64 for new architecture
    
    visual_context = visual_context.to(device)

    img = torch.randn(shape, device=device)  # Start with random noise x_T
    imgs_history = [tensor_to_pil(img.clone())]  # Store initial noise

    for i in reversed(range(0, num_timesteps)):
        t_int = i
        t = torch.full((batch_size,), t_int, device=device, dtype=torch.long)
        img = p_sample(model, img, t, t_int, visual_context)
        
        # Store some intermediate images
        if i % (num_timesteps // 10) == 0 or i == 0:
            imgs_history.append(tensor_to_pil(img.clone()))

    return img, imgs_history

def load_model(checkpoint_path: str, device: torch.device = None):
    """
    Load a trained DiffusionModel from checkpoint.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        device: Device to load the model on
    Returns:
        model: Loaded DiffusionModel
    """
    if device is None:
        device = get_device()
    
    # Import here to avoid circular import
    try:
        from .model import DiffusionModel
    except ImportError:
        from model import DiffusionModel
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config
    if 'model_config' not in checkpoint:
        # Use default config
        print("WARNING: Checkpoint missing 'model_config'. Using default configuration...")
        config = {
            'img_size': 64,
            'patch_size': 8,
            'img_channels': 3,
            'latent_dim': 256,
            'time_dim': 64,
            'num_transformer_blocks': 4,
            'num_heads': 8,
            'ff_dim_multiplier': 4,
            'dropout': 0.1
        }
    else:
        config = checkpoint['model_config']
        print("Found model_config in checkpoint ✅")
    
    # Initialize model
    try:
        model = DiffusionModel(**config)
        print(f"Initialized DiffusionModel with config: {config}")
    except Exception as e:
        raise ValueError(f"Failed to initialize DiffusionModel with config {config}: {e}")
    
    # Load state dict
    try:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)
            
        model.to(device)
        model.eval()
        
        print(f"✅ Successfully loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Training loss: {checkpoint.get('loss', 'unknown')}")
        
        return model
        
    except Exception as e:
        error_msg = f"Failed to load state_dict into DiffusionModel: {str(e)}\n\n"
        error_msg += "This usually means:\n"
        error_msg += "1. The checkpoint is from a different model architecture\n"
        error_msg += "2. The model configuration doesn't match the saved weights\n"
        error_msg += "3. You're trying to load an old model checkpoint into the new architecture\n\n"
        error_msg += "Solutions:\n"
        error_msg += "- Train a new model using the new architecture\n"
        error_msg += "- Check that the checkpoint file is not corrupted"
        
        raise ValueError(error_msg)

def generate_from_visual_prompt(model, prompt_image: Image.Image, num_samples: int = 1, device: torch.device = None):
    """
    Generate images using a visual prompt.
    
    Args:
        model: Trained DiffusionModel
        prompt_image: PIL Image to use as visual conditioning
        num_samples: Number of images to generate
        device: Device to run inference on
    Returns:
        generated_images: List of PIL Images (64x64)
        conditioning_image: PIL Image showing the 8x8 conditioning
    """
    if device is None:
        device = get_device()
    
    model.to(device)
    model.eval()
    
    # Prepare visual context (8x8)
    context_tensor = pil_to_tensor(prompt_image, target_size=(8, 8))  # (1, 3, 8, 8)
    context_batch = context_tensor.repeat(num_samples, 1, 1, 1).to(device)  # (num_samples, 3, 8, 8)
    
    # Generate images
    with torch.no_grad():
        generated_tensor, history = p_sample_loop(model, context_batch, device=device)
    
    # Convert to PIL images
    generated_images = []
    for i in range(num_samples):
        img = tensor_to_pil(generated_tensor[i])
        generated_images.append(img)
    
    # Also return the conditioning image
    conditioning_image = tensor_to_pil(context_tensor[0])
    
    return generated_images, conditioning_image

# Test inference functionality
if __name__ == '__main__':
    device = get_device()
    print(f"Testing inference on device: {device}")
    
    # Import here to avoid circular import
    try:
        from .model import DiffusionModel
    except ImportError:
        from model import DiffusionModel
    
    # Create a dummy model for testing
    model = DiffusionModel(
        img_size=64,
        patch_size=8,
        img_channels=3,
        latent_dim=256,
        time_dim=64,
        num_transformer_blocks=4,
        num_heads=8
    ).to(device)
    
    # Create dummy visual context
    dummy_context = torch.randn(1, 3, 8, 8, device=device)
    
    print("Testing sampling loop...")
    try:
        generated, history = p_sample_loop(model, dummy_context, num_timesteps=10, device=device)  # Short test
        print(f"✅ Generated image shape: {generated.shape}")
        print(f"✅ History length: {len(history)}")
        
        # Test PIL conversion
        pil_img = tensor_to_pil(generated[0])
        print(f"✅ PIL image size: {pil_img.size}")
        
        # Test with dummy PIL image
        dummy_pil = Image.new('RGB', (64, 64), color=(128, 128, 128))
        generated_images, conditioning = generate_from_visual_prompt(model, dummy_pil, num_samples=2, device=device)
        print(f"✅ Generated {len(generated_images)} images from visual prompt")
        print(f"✅ Conditioning image size: {conditioning.size}")
        
        # Test forward noise
        dummy_clean = torch.randn(2, 3, 64, 64, device=device)
        dummy_t = torch.randint(0, TIMESTEPS, (2,), device=device)
        noisy, noise = forward_noise(dummy_clean, dummy_t, device)
        print(f"✅ Forward noise: {dummy_clean.shape} -> {noisy.shape}")
        
        print("✅ New inference system working correctly!")
        
    except Exception as e:
        print(f"❌ Inference error: {e}")
        import traceback
        traceback.print_exc()
