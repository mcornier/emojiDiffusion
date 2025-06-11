import torch
import torchvision

# --- Device Helper ---
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Diffusion Schedule ---
def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02):
    """
    Generates a linear schedule for beta values.
    """
    return torch.linspace(beta_start, beta_end, timesteps)

# Pre-calculate diffusion constants
TIMESTEPS = 200 # Default, can be configured
BETAS = linear_beta_schedule(TIMESTEPS)
ALPHAS = 1. - BETAS
ALPHAS_CUMPROD = torch.cumprod(ALPHAS, axis=0)
ALPHAS_CUMPROD_PREV = torch.nn.functional.pad(ALPHAS_CUMPROD[:-1], (1, 0), value=1.0) # F.pad in PyTorch

# For q_sample (forward_noise)
SQRT_ALPHAS_CUMPROD = torch.sqrt(ALPHAS_CUMPROD)
SQRT_ONE_MINUS_ALPHAS_CUMPROD = torch.sqrt(1. - ALPHAS_CUMPROD)

# For q_posterior (denoising step)
POSTERIOR_VARIANCE = BETAS * (1. - ALPHAS_CUMPROD_PREV) / (1. - ALPHAS_CUMPROD)


def forward_noise(x_0: torch.Tensor, t: torch.Tensor, device: torch.device = None):
    """
    Adds noise to an image x_0 at timestep t.
    q(x_t | x_0) = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * epsilon
    where epsilon is random noise.

    Args:
        x_0: The initial clean image (batch_size, channels, height, width).
        t: The timestep (batch_size,) or a single integer timestep.
        device: The device to perform calculations on.
    Returns:
        noised_image: The noised image at timestep t.
        noise: The noise that was added.
    """
    if device is None:
        device = get_device()

    noise = torch.randn_like(x_0, device=device)

    # Ensure t is correctly shaped for gather
    # gather needs t to be of the same number of dimensions as the tensor being gathered from,
    # but with the relevant dimension size matching the number of indices.
    # For 1D tensors like SQRT_ALPHAS_CUMPROD, t should be 1D.

    # If t is a scalar, make it a 1D tensor for gather
    if t.ndim == 0:
        t = t.unsqueeze(0)

    # Ensure t is on the correct device
    t = t.to(device)

    # Make sure x_0 is on the correct device
    x_0 = x_0.to(device)

    sqrt_alphas_cumprod_t = SQRT_ALPHAS_CUMPROD.to(device)[t]
    sqrt_one_minus_alphas_cumprod_t = SQRT_ONE_MINUS_ALPHAS_CUMPROD.to(device)[t]

    # Reshape for broadcasting: (batch_size, 1, 1, 1)
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1)

    noised_image = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    return noised_image, noise

# --- Tensor to PIL Image and back (for visualization) ---
def tensor_to_pil(image_tensor: torch.Tensor):
    """Converts a [-1, 1] normalized tensor to a PIL Image."""
    image_tensor = (image_tensor + 1) / 2 # Denormalize to [0, 1]
    image_tensor = image_tensor.clamp(0, 1)
    # If batch, take the first image
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]
    return torchvision.transforms.ToPILImage()(image_tensor.cpu())

def pil_to_tensor(pil_image):
    """Converts a PIL Image to a [-1, 1] normalized tensor."""
    return torchvision.transforms.ToTensor()(pil_image).unsqueeze(0) * 2.0 - 1.0 # Add batch dim and normalize

# --- Diffusion Sampler (denoising step) ---
@torch.no_grad()
def p_sample(model, x_t: torch.Tensor, t: torch.Tensor, t_index: int, context_embedding: torch.Tensor = None):
    """
    Samples x_{t-1} from x_t using the model's prediction of the noise.
    This is one step in the reverse diffusion process.

    Args:
        model: The diffusion model.
        x_t: The current noised image (batch_size, channels, height, width).
        t: The current timestep (scalar tensor).
        t_index: The integer index for t (used for indexing precomputed values).
        context_embedding: Optional context embedding.
    Returns:
        x_prev: The denoised image x_{t-1}.
    """
    device = x_t.device
    betas_t = BETAS.to(device)[t_index]
    sqrt_one_minus_alphas_cumprod_t = SQRT_ONE_MINUS_ALPHAS_CUMPROD.to(device)[t_index]
    sqrt_recip_alphas_t = torch.sqrt(1.0 / ALPHAS.to(device)[t_index])

    # Model predicts the noise (epsilon)
    if context_embedding is not None:
        predicted_noise = model(x_t, t, context_embedding)
    else:
        predicted_noise = model(x_t, t) # Or model(x_t, t, None) if model expects 3 args

    # Equation 11 from DDPM paper:
    # x_{t-1} = 1/sqrt(alpha_t) * (x_t - (beta_t / sqrt(1-alpha_cumprod_t)) * epsilon_t) + sigma_t * z
    model_mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean # No noise added at the last step
    else:
        posterior_variance_t = POSTERIOR_VARIANCE.to(device)[t_index]
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, shape, num_timesteps:int = TIMESTEPS, context_embedding: torch.Tensor = None, device: torch.device = None):
    """
    Generates an image by sampling from the diffusion model, starting from random noise.
    This is the full reverse diffusion process (DDPM sampling).

    Args:
        model: The diffusion model.
        shape: The shape of the image to generate (batch_size, channels, height, width).
        num_timesteps: The total number of diffusion timesteps.
        context_embedding: Optional context embedding.
        device: The device to perform calculations on.
    Returns:
        img: The generated image.
    """
    if device is None:
        device = get_device()

    model.to(device)
    model.eval()

    img = torch.randn(shape, device=device) # Start with random noise x_T
    imgs_history = [tensor_to_pil(img.clone())] # Store initial noise

    for i in reversed(range(0, num_timesteps)):
        t_int = i
        t = torch.full((shape[0],), t_int, device=device, dtype=torch.long) # Tensor for timestep
        img = p_sample(model, img, t, t_int, context_embedding)
        if i % (num_timesteps // 10) == 0 or i == 0: # Store some intermediate images
             imgs_history.append(tensor_to_pil(img.clone()))

    # The user wants image_t, t, predicted_noise, image_t-1.
    # This loop is for full generation. The inference step-by-step will be slightly different.
    # For now, this function returns the final image and a history for dev purposes.
    return img, imgs_history

# Note: The TIMESTEPS variable is defined globally in this file.
# If it needs to be configurable per model instance or per run,
# it should be passed as an argument to functions or stored in a class.
# For now, using the global TIMESTEPS for simplicity.

@torch.no_grad()
def denoise_single_step(model, x_t: torch.Tensor, t_tensor: torch.Tensor, t_index: int, context_idx_tensor: torch.Tensor = None):
    device = x_t.device
    model.to(device)
    model.eval()

    betas_t = BETAS.to(device)[t_index]
    sqrt_one_minus_alphas_cumprod_t = SQRT_ONE_MINUS_ALPHAS_CUMPROD.to(device)[t_index]
    sqrt_recip_alphas_t = torch.sqrt(1.0 / ALPHAS.to(device)[t_index])

    predicted_noise = model(x_t, t_tensor, context_idx_tensor)
    model_mean = sqrt_recip_alphas_t * (x_t - (betas_t / sqrt_one_minus_alphas_cumprod_t) * predicted_noise)

    if t_index == 0:
        x_prev = model_mean
    else:
        posterior_variance_t = POSTERIOR_VARIANCE.to(device)[t_index]
        noise_for_sampling = torch.randn_like(x_t)
        x_prev = model_mean + torch.sqrt(posterior_variance_t) * noise_for_sampling

    return predicted_noise, x_prev
