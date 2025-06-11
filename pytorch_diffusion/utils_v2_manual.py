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

# --- Manual Step-by-Step Diffusion ---
class ManualDiffusionSampler:
    """Manual step-by-step diffusion sampler for educational purposes"""
    
    def __init__(self, model, visual_context, timesteps=200, device=None):
        """
        Args:
            model: DiffusionModelV2
            visual_context: (B, 3, 8, 8) visual conditioning tensor
            timesteps: Number of diffusion steps
            device: Device for computation
        """
        if device is None:
            device = get_device()
        
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.timesteps = timesteps
        
        # Ensure visual context is on correct device
        self.visual_context = visual_context.to(device)
        self.batch_size = visual_context.shape[0]
        
        # Diffusion schedule
        self.betas = linear_beta_schedule(timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # For sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
        # State
        self.current_t = timesteps - 1  # Start at t=T-1 (199 for 200 steps)
        self.current_image = None
        self.last_predicted_noise = None
        self.history = []
        
        self.reset()
    
    def reset(self):
        """Reset to initial noise state"""
        self.current_t = self.timesteps - 1
        shape = (self.batch_size, 3, 64, 64)
        self.current_image = torch.randn(shape, device=self.device)
        self.last_predicted_noise = None
        self.history = []
        
        # Store initial state
        self.history.append({
            'timestep': self.current_t,
            'image': self.current_image.clone(),
            'predicted_noise': None,
            'info': 'Initial random noise'
        })
    
    def step(self):
        """Perform one denoising step"""
        if self.current_t < 0:
            return False  # Already finished
        
        # Current timestep tensor
        t_tensor = torch.full((self.batch_size,), self.current_t, device=self.device, dtype=torch.long)
        
        # Predict noise with model
        with torch.no_grad():
            predicted_noise = self.model(self.current_image, t_tensor, self.visual_context)
        
        self.last_predicted_noise = predicted_noise.clone()
        
        # Perform denoising step
        if self.current_t > 0:
            # Not the final step
            beta_t = self.betas[self.current_t]
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[self.current_t]
            sqrt_recip_alpha_t = torch.sqrt(1.0 / self.alphas[self.current_t])
            
            # Equation 11 from DDPM paper
            model_mean = sqrt_recip_alpha_t * (self.current_image - beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t)
            
            # Add noise for non-final steps
            posterior_variance_t = self.posterior_variance[self.current_t]
            noise = torch.randn_like(self.current_image)
            self.current_image = model_mean + torch.sqrt(posterior_variance_t) * noise
            
            info = f"Denoising step t={self.current_t} → t={self.current_t-1}"
        else:
            # Final step (t=0) - no noise added
            beta_t = self.betas[self.current_t]
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[self.current_t]
            sqrt_recip_alpha_t = torch.sqrt(1.0 / self.alphas[self.current_t])
            
            self.current_image = sqrt_recip_alpha_t * (self.current_image - beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t)
            info = f"Final step t={self.current_t} → DONE"
        
        # Store step in history
        self.history.append({
            'timestep': self.current_t,
            'image': self.current_image.clone(),
            'predicted_noise': predicted_noise.clone(),
            'info': info
        })
        
        # Move to next timestep
        self.current_t -= 1
        
        return True  # Step successful
    
    def step_multiple(self, num_steps):
        """Perform multiple denoising steps"""
        results = []
        for _ in range(num_steps):
            if not self.step():
                break  # Finished
            results.append(self.get_current_state())
        return results
    
    def get_current_state(self):
        """Get current state information"""
        return {
            'timestep': self.current_t + 1,  # +1 because we decremented after step
            'image': self.current_image.clone(),
            'predicted_noise': self.last_predicted_noise.clone() if self.last_predicted_noise is not None else None,
            'progress_percent': ((self.timesteps - 1 - self.current_t) / self.timesteps) * 100,
            'is_finished': self.current_t < 0
        }
    
    def is_finished(self):
        """Check if sampling is complete"""
        return self.current_t < 0
    
    def get_final_image(self):
        """Get the final denoised image"""
        if not self.is_finished():
            return None
        return self.current_image.clone()
    
    def get_history_images(self, every_n_steps=10):
        """Get history of images at regular intervals"""
        images = []
        for i, state in enumerate(self.history):
            if i % every_n_steps == 0 or i == len(self.history) - 1:
                images.append({
                    'timestep': state['timestep'],
                    'image': state['image'],
                    'info': state['info']
                })
        return images

def tensor_to_pil_v2(image_tensor: torch.Tensor):
    """Converts a [-1, 1] normalized tensor to a PIL Image (for 64x64)."""
    image_tensor = (image_tensor + 1) / 2  # Denormalize to [0, 1]
    image_tensor = image_tensor.clamp(0, 1)
    # If batch, take the first image
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]
    return torchvision.transforms.ToPILImage()(image_tensor.cpu())

def create_noise_visualization(noise_tensor):
    """Create a visualization of predicted noise"""
    # Normalize noise for visualization
    noise = noise_tensor.clone()
    if noise.ndim == 4:
        noise = noise[0]  # Take first batch
    
    # Normalize to [0, 1] for visualization
    noise_min = noise.min()
    noise_max = noise.max()
    if noise_max > noise_min:
        noise = (noise - noise_min) / (noise_max - noise_min)
    else:
        noise = torch.zeros_like(noise)
    
    return torchvision.transforms.ToPILImage()(noise.cpu())

# Test the manual sampler
if __name__ == '__main__':
    device = get_device()
    print(f"Testing manual sampler on device: {device}")
    
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
    visual_context = torch.randn(1, 3, 8, 8, device=device)
    
    print("Testing manual diffusion sampler...")
    try:
        # Create sampler
        sampler = ManualDiffusionSampler(model, visual_context, timesteps=10, device=device)  # Short test
        
        print(f"Initial state: t={sampler.current_t}, finished={sampler.is_finished()}")
        
        # Perform a few steps
        for i in range(5):
            if sampler.step():
                state = sampler.get_current_state()
                print(f"Step {i+1}: t={state['timestep']}, progress={state['progress_percent']:.1f}%, finished={state['is_finished']}")
            else:
                print(f"Step {i+1}: Sampling finished")
                break
        
        # Test multiple steps
        print("\nTesting multiple steps...")
        remaining_steps = 5
        results = sampler.step_multiple(remaining_steps)
        print(f"Performed {len(results)} more steps")
        
        # Final state
        final_state = sampler.get_current_state()
        print(f"Final state: finished={final_state['is_finished']}, progress={final_state['progress_percent']:.1f}%")
        
        # Test image conversion
        if sampler.history:
            test_image = sampler.history[0]['image']
            pil_img = tensor_to_pil_v2(test_image)
            print(f"PIL conversion: {pil_img.size}")
        
        # Test noise visualization
        if sampler.last_predicted_noise is not None:
            noise_vis = create_noise_visualization(sampler.last_predicted_noise)
            print(f"Noise visualization: {noise_vis.size}")
        
        print("✅ Manual sampler working correctly!")
        
    except Exception as e:
        print(f"❌ Manual sampler error: {e}")
        import traceback
        traceback.print_exc()
