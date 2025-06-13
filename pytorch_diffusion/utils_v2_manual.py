import torch
import numpy as np
from PIL import Image
import torchvision
from .utils import (
    get_device, tensor_to_pil, TIMESTEPS, BETAS, ALPHAS, ALPHAS_CUMPROD, 
    SQRT_ALPHAS_CUMPROD, SQRT_ONE_MINUS_ALPHAS_CUMPROD, POSTERIOR_VARIANCE
)

def create_noise_visualization(noise_tensor):
    """Create a visualization of predicted noise for display"""
    if noise_tensor.dim() == 4:
        noise_tensor = noise_tensor[0]  # Take first batch item if needed
    
    # Normalize noise to [0, 1] for visualization
    noise_normalized = (noise_tensor - noise_tensor.min()) / (noise_tensor.max() - noise_tensor.min() + 1e-8)
    
    # Convert to PIL
    return torchvision.transforms.ToPILImage()(noise_normalized.cpu())

class ManualDiffusionSampler:
    """
    A class for manual step-by-step diffusion sampling, allowing users to
    observe each denoising step in the process.
    """
    
    def __init__(self, model, visual_context, timesteps=TIMESTEPS, device=None):
        """
        Initialize the manual sampler.
        
        Args:
            model: The trained diffusion model
            visual_context: Visual conditioning tensor (B, 3, 8, 8)
            timesteps: Total number of timesteps (default 200)
            device: Device to run on
        """
        self.model = model
        self.visual_context = visual_context
        self.timesteps = timesteps
        self.device = device or get_device()
        
        # Move everything to device
        self.model.to(self.device)
        self.visual_context = self.visual_context.to(self.device)
        
        # Initialize the sampling
        self.reset()
    
    def reset(self):
        """Reset to initial noise state"""
        batch_size = self.visual_context.shape[0]
        shape = (batch_size, 3, 64, 64)
        
        # Start with random noise
        self.current_image = torch.randn(shape, device=self.device)
        self.current_timestep = self.timesteps - 1  # Start from T-1
        self.current_step = 0
        self.predicted_noise = None
        self.is_complete = False
        
        print(f"Manual sampler reset: Starting from timestep {self.current_timestep}")
    
    def step(self):
        """
        Perform one denoising step.
        
        Returns:
            bool: True if step was successful, False if already finished
        """
        if self.is_complete:
            return False
        
        # Current timestep for model input
        t_int = self.current_timestep
        batch_size = self.current_image.shape[0]
        t = torch.full((batch_size,), t_int, device=self.device, dtype=torch.long)
        
        # Get model prediction
        with torch.no_grad():
            self.predicted_noise = self.model(self.current_image, t, self.visual_context)
        
        # Perform denoising step
        device = self.device
        betas_t = BETAS.to(device)[t_int]
        sqrt_one_minus_alphas_cumprod_t = SQRT_ONE_MINUS_ALPHAS_CUMPROD.to(device)[t_int]
        sqrt_recip_alphas_t = torch.sqrt(1.0 / ALPHAS.to(device)[t_int])

        # DDPM denoising equation
        model_mean = sqrt_recip_alphas_t * (
            self.current_image - betas_t * self.predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_int == 0:
            # Last step - no noise added
            self.current_image = model_mean
            self.is_complete = True
        else:
            # Add posterior noise
            posterior_variance_t = POSTERIOR_VARIANCE.to(device)[t_int]
            noise = torch.randn_like(self.current_image)
            self.current_image = model_mean + torch.sqrt(posterior_variance_t) * noise
        
        # Update counters
        self.current_timestep -= 1
        self.current_step += 1
        
        print(f"Step {self.current_step}: t={t_int} -> t={self.current_timestep}")
        
        return True
    
    def is_finished(self):
        """Check if sampling is complete"""
        return self.is_complete
    
    def get_current_state(self):
        """
        Get the current state of the sampler for display.
        
        Returns:
            dict: Contains current image, timestep, progress, etc.
        """
        progress_percent = (self.current_step / self.timesteps) * 100
        
        # Calculate current variance for display
        if self.current_timestep >= 0 and self.current_timestep < len(POSTERIOR_VARIANCE):
            variance = POSTERIOR_VARIANCE[self.current_timestep].item()
        else:
            variance = 0.0
        
        return {
            'image': self.current_image[0].clone(),  # First batch item
            'timestep': self.current_timestep,
            'step': self.current_step,
            'progress_percent': progress_percent,
            'predicted_noise': self.predicted_noise[0].clone() if self.predicted_noise is not None else None,
            'is_finished': self.is_complete,
            'variance': f"{variance:.6f}"
        }
    
    def get_final_image(self):
        """Get the final generated image as PIL"""
        if not self.is_complete:
            print("Warning: Sampling not complete yet")
        
        return tensor_to_pil(self.current_image[0])
    
    def skip_to_step(self, target_step):
        """
        Skip ahead to a specific step (for slider interaction).
        
        Args:
            target_step: Target step number (0 to timesteps-1)
        """
        if target_step < self.current_step:
            # Need to reset and step forward
            self.reset()
        
        # Step forward to target
        while self.current_step < target_step and not self.is_complete:
            self.step()
    
    def get_progress_images(self, num_images=10):
        """
        Generate a sequence of images showing the denoising progress.
        
        Args:
            num_images: Number of intermediate images to generate
            
        Returns:
            List of PIL images showing denoising progression
        """
        # Save current state
        original_step = self.current_step
        original_timestep = self.current_timestep
        original_image = self.current_image.clone()
        original_complete = self.is_complete
        
        # Reset and generate progression
        self.reset()
        images = [tensor_to_pil(self.current_image[0])]  # Initial noise
        
        step_interval = max(1, self.timesteps // num_images)
        
        for step in range(0, self.timesteps, step_interval):
            while self.current_step < step and not self.is_complete:
                self.step()
            if not self.is_complete or step == self.timesteps - 1:
                images.append(tensor_to_pil(self.current_image[0]))
        
        # Restore original state
        self.current_step = original_step
        self.current_timestep = original_timestep
        self.current_image = original_image
        self.is_complete = original_complete
        
        return images


# Test the manual sampler
if __name__ == '__main__':
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    from pytorch_diffusion.model import DiffusionModel
    
    device = get_device()
    print(f"Testing ManualDiffusionSampler on device: {device}")
    
    # Create a dummy model
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
    visual_context = torch.randn(1, 3, 8, 8, device=device)
    
    # Test manual sampler
    try:
        sampler = ManualDiffusionSampler(model, visual_context, timesteps=10, device=device)
        
        print("Testing manual steps...")
        for i in range(5):
            if not sampler.is_finished():
                success = sampler.step()
                state = sampler.get_current_state()
                print(f"Step {i+1}: timestep={state['timestep']}, progress={state['progress_percent']:.1f}%")
        
        print("Testing reset...")
        sampler.reset()
        state = sampler.get_current_state()
        print(f"After reset: timestep={state['timestep']}, step={state['step']}")
        
        print("Testing progress images...")
        progress_imgs = sampler.get_progress_images(num_images=3)
        print(f"Generated {len(progress_imgs)} progress images")
        
        print("✅ ManualDiffusionSampler working correctly!")
        
    except Exception as e:
        print(f"❌ Error testing ManualDiffusionSampler: {e}")
        import traceback
        traceback.print_exc()
