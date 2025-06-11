import torch
import torchvision
from PIL import Image
import numpy as np

try:
    from .model_v2 import DiffusionModelV2
    from .utils import get_device, linear_beta_schedule
except ImportError:
    # For direct execution
    from model_v2 import DiffusionModelV2
    from utils import get_device, linear_beta_schedule

# --- Diffusion Schedule for V2 ---
TIMESTEPS_V2 = 200
BETAS_V2 = linear_beta_schedule(TIMESTEPS_V2)
ALPHAS_V2 = 1. - BETAS_V2
ALPHAS_CUMPROD_V2 = torch.cumprod(ALPHAS_V2, axis=0)
ALPHAS_CUMPROD_PREV_V2 = torch.nn.functional.pad(ALPHAS_CUMPROD_V2[:-1], (1, 0), value=1.0)

# For q_sample (forward_noise)
SQRT_ALPHAS_CUMPROD_V2 = torch.sqrt(ALPHAS_CUMPROD_V2)
SQRT_ONE_MINUS_ALPHAS_CUMPROD_V2 = torch.sqrt(1. - ALPHAS_CUMPROD_V2)

# For q_posterior (denoising step)
POSTERIOR_VARIANCE_V2 = BETAS_V2 * (1. - ALPHAS_CUMPROD_PREV_V2) / (1. - ALPHAS_CUMPROD_V2)

def tensor_to_pil_v2(image_tensor: torch.Tensor):
    """Converts a [-1, 1] normalized tensor to a PIL Image (for 64x64)."""
    image_tensor = (image_tensor + 1) / 2  # Denormalize to [0, 1]
    image_tensor = image_tensor.clamp(0, 1)
    # If batch, take the first image
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]
    return torchvision.transforms.ToPILImage()(image_tensor.cpu())

def pil_to_tensor_v2(pil_image, target_size=None):
    """Converts a PIL Image to a [-1, 1] normalized tensor."""
    if target_size:
        pil_image = pil_image.resize(target_size, Image.LANCZOS)
    tensor = torchvision.transforms.ToTensor()(pil_image)
    if tensor.shape[0] == 1:  # Grayscale to RGB
        tensor = tensor.repeat(3, 1, 1)
    elif tensor.shape[0] == 4:  # RGBA to RGB
        tensor = tensor[:3]
    return (tensor * 2.0 - 1.0).unsqueeze(0)  # Add batch dim and normalize to [-1, 1]

@torch.no_grad()
def p_sample_v2(model, x_t: torch.Tensor, t: torch.Tensor, t_index: int, visual_context: torch.Tensor):
    """
    Samples x_{t-1} from x_t using the model V2's prediction of the noise.
    
    Args:
        model: The DiffusionModelV2
        x_t: The current noised image (B, 3, 64, 64)
        t: The current timestep (scalar tensor)
        t_index: The integer index for t
        visual_context: Visual conditioning (B, 3, 8, 8)
    Returns:
        x_prev: The denoised image x_{t-1}
    """
    device = x_t.device
    betas_t = BETAS_V2.to(device)[t_index]
    sqrt_one_minus_alphas_cumprod_t = SQRT_ONE_MINUS_ALPHAS_CUMPROD_V2.to(device)[t_index]
    sqrt_recip_alphas_t = torch.sqrt(1.0 / ALPHAS_V2.to(device)[t_index])

    # Model predicts the noise (epsilon)
    predicted_noise = model(x_t, t, visual_context)

    # Equation 11 from DDPM paper:
    # x_{t-1} = 1/sqrt(alpha_t) * (x_t - (beta_t / sqrt(1-alpha_cumprod_t)) * epsilon_t) + sigma_t * z
    model_mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean  # No noise added at the last step
    else:
        posterior_variance_t = POSTERIOR_VARIANCE_V2.to(device)[t_index]
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop_v2(model, visual_context: torch.Tensor, num_timesteps: int = TIMESTEPS_V2, device: torch.device = None):
    """
    Generates a 64x64 image by sampling from the diffusion model V2, starting from random noise.
    
    Args:
        model: The DiffusionModelV2
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
    shape = (batch_size, 3, 64, 64)  # 64x64 for V2
    
    visual_context = visual_context.to(device)

    img = torch.randn(shape, device=device)  # Start with random noise x_T
    imgs_history = [tensor_to_pil_v2(img.clone())]  # Store initial noise

    for i in reversed(range(0, num_timesteps)):
        t_int = i
        t = torch.full((batch_size,), t_int, device=device, dtype=torch.long)
        img = p_sample_v2(model, img, t, t_int, visual_context)
        
        # Store some intermediate images
        if i % (num_timesteps // 10) == 0 or i == 0:
            imgs_history.append(tensor_to_pil_v2(img.clone()))

    return img, imgs_history

def load_model_v2(checkpoint_path: str, device: torch.device = None):
    """
    Load a trained DiffusionModelV2 from checkpoint.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        device: Device to load the model on
    Returns:
        model: Loaded DiffusionModelV2
    """
    if device is None:
        device = get_device()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if this is a V2 checkpoint by looking for model_config or V2-specific keys
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Check for V2-specific keys
    v2_keys = ['visual_encoder.conv1.weight', 'context_embedding_layer.weight', 'fusion_layer.weight']
    is_v2_checkpoint = any(key in state_dict for key in v2_keys)
    
    if not is_v2_checkpoint:
        raise ValueError(
            f"This checkpoint appears to be from an older model architecture, not compatible with DiffusionModelV2.\n"
            f"Please use a checkpoint that was saved during V2 training.\n"
            f"V2 checkpoints should be in 'pytorch_diffusion_qt/checkpoints/' folder with names like 'diffusion_v2_ckpt_*'"
        )
    
    # Get model config - if not present, this might still be incompatible
    if 'model_config' not in checkpoint:
        # Try to infer config from state_dict shape, but warn user
        print("WARNING: Checkpoint missing 'model_config'. Attempting to infer configuration...")
        
        # Try to get dimensions from state_dict
        try:
            # Check visual encoder input
            if 'visual_encoder.conv1.weight' in state_dict:
                conv1_shape = state_dict['visual_encoder.conv1.weight'].shape
                print(f"Visual encoder conv1 shape: {conv1_shape}")
            
            # Use default V2 config but warn it might not work
            config = {
                'img_size': 64,
                'img_channels': 3,
                'latent_dim': 256,
                'time_dim': 64,
                'visual_context_dim': 192,
                'num_transformer_blocks': 3,
                'num_heads': 4,
                'initial_conv_filters': 32,
                'conv_dim_mults': (1, 2, 4, 8)
            }
            print("Using default V2 configuration. This may cause loading errors if the checkpoint was saved with different parameters.")
        except Exception as e:
            raise ValueError(f"Cannot infer model configuration from checkpoint: {e}")
    else:
        config = checkpoint['model_config']
        print("Found model_config in checkpoint ✅")
    
    # Initialize model
    try:
        model = DiffusionModelV2(**config)
        print(f"Initialized DiffusionModelV2 with config: {config}")
    except Exception as e:
        raise ValueError(f"Failed to initialize DiffusionModelV2 with config {config}: {e}")
    
    # Load state dict with error checking
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
        # Provide helpful error message
        error_msg = f"Failed to load state_dict into DiffusionModelV2: {str(e)}\n\n"
        error_msg += "This usually means:\n"
        error_msg += "1. The checkpoint is from a different model architecture\n"
        error_msg += "2. The model configuration doesn't match the saved weights\n"
        error_msg += "3. You're trying to load an old model checkpoint into the new V2 architecture\n\n"
        error_msg += "Solutions:\n"
        error_msg += "- Use a checkpoint from Training V2 (look in pytorch_diffusion_qt/checkpoints/)\n"
        error_msg += "- Train a new model using Training V2 tab\n"
        error_msg += "- Check that the checkpoint file is not corrupted"
        
        raise ValueError(error_msg)

def generate_from_visual_prompt(model, prompt_image: Image.Image, num_samples: int = 1, device: torch.device = None):
    """
    Generate images using a visual prompt.
    
    Args:
        model: Trained DiffusionModelV2
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
    context_tensor = pil_to_tensor_v2(prompt_image, target_size=(8, 8))  # (1, 3, 8, 8)
    context_batch = context_tensor.repeat(num_samples, 1, 1, 1).to(device)  # (num_samples, 3, 8, 8)
    
    # Generate images
    with torch.no_grad():
        generated_tensor, history = p_sample_loop_v2(model, context_batch, device=device)
    
    # Convert to PIL images
    generated_images = []
    for i in range(num_samples):
        img = tensor_to_pil_v2(generated_tensor[i])
        generated_images.append(img)
    
    # Also return the conditioning image
    conditioning_image = tensor_to_pil_v2(context_tensor[0])
    
    return generated_images, conditioning_image

# Test inference functionality
if __name__ == '__main__':
    device = get_device()
    print(f"Testing inference on device: {device}")
    
    # Create a dummy model for testing
    model = DiffusionModelV2(
        img_size=64,
        img_channels=3,
        latent_dim=256,
        time_dim=64,
        visual_context_dim=192,
        num_transformer_blocks=3,
        num_heads=4,
        initial_conv_filters=32,
        conv_dim_mults=(1, 2, 4, 8)
    ).to(device)
    
    # Create dummy visual context
    dummy_context = torch.randn(1, 3, 8, 8, device=device)
    
    print("Testing sampling loop...")
    try:
        generated, history = p_sample_loop_v2(model, dummy_context, num_timesteps=10, device=device)  # Short test
        print(f"✅ Generated image shape: {generated.shape}")
        print(f"✅ History length: {len(history)}")
        
        # Test PIL conversion
        pil_img = tensor_to_pil_v2(generated[0])
        print(f"✅ PIL image size: {pil_img.size}")
        
        # Test with dummy PIL image
        dummy_pil = Image.new('RGB', (64, 64), color=(128, 128, 128))
        generated_images, conditioning = generate_from_visual_prompt(model, dummy_pil, num_samples=2, device=device)
        print(f"✅ Generated {len(generated_images)} images from visual prompt")
        print(f"✅ Conditioning image size: {conditioning.size}")
        
        print("✅ Inference V2 working correctly!")
        
    except Exception as e:
        print(f"❌ Inference error: {e}")
        import traceback
        traceback.print_exc()
