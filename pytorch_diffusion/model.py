import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Device Selection ---
def get_device():
    """Get the best available device for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# --- Time Embedding ---
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        if time.ndim == 1:
            time = time.unsqueeze(1)
        embeddings = time * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0,1))
        return embeddings

# --- Unified Patch Encoder (8x8x3 -> 256D) ---
class UnifiedPatchEncoder(nn.Module):
    """Unified encoder for both image patches and visual prompt (8x8x3 → 256D)"""
    def __init__(self, output_dim=256):
        super().__init__()
        self.output_dim = output_dim
        
        # 8x8x3 → 8x8x32 → 4x4x64 → 2x2x128 → 256
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # 8x8x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 4x4x64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 2x2x128
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 2 * 2, output_dim)  # 512 → 256
        
        self.activation = nn.ReLU()
        self.norm1 = nn.InstanceNorm2d(32)
        self.norm2 = nn.InstanceNorm2d(64)
        self.norm3 = nn.InstanceNorm2d(128)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # x: (B, 3, 8, 8) ou (B*num_patches, 3, 8, 8)
        x = self.activation(self.norm1(self.conv1(x)))  # (B, 32, 8, 8)
        x = self.activation(self.norm2(self.conv2(x)))  # (B, 64, 4, 4)
        x = self.activation(self.norm3(self.conv3(x)))  # (B, 128, 2, 2)
        
        x = self.flatten(x)  # (B, 512)
        x = self.dropout(x)
        x = self.fc(x)       # (B, 256)
        return x

# --- Patch Decoder (256D -> 8x8x3) ---
class PatchDecoder(nn.Module):
    """Decoder from 256D features back to 8x8x3 patch"""
    def __init__(self, input_dim=256):
        super().__init__()
        self.input_dim = input_dim
        
        # 256 → 512 → 2x2x128 → 4x4x64 → 8x8x32 → 8x8x3
        self.fc = nn.Linear(input_dim, 128 * 2 * 2)  # 256 → 512
        self.unflatten = nn.Unflatten(1, (128, 2, 2))
        
        self.tconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  # 2x2 → 4x4
        self.tconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)   # 4x4 → 8x8
        self.tconv3 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)  # 8x8x3 final
        
        self.activation = nn.ReLU()
        self.norm1 = nn.InstanceNorm2d(64)
        self.norm2 = nn.InstanceNorm2d(32)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # x: (B, 256)
        x = self.fc(x)  # (B, 512)
        x = self.unflatten(x)  # (B, 128, 2, 2)
        
        x = self.activation(self.norm1(self.tconv1(x)))  # (B, 64, 4, 4)
        x = self.activation(self.norm2(self.tconv2(x)))  # (B, 32, 8, 8)
        x = self.tconv3(x)  # (B, 3, 8, 8)
        
        return x

# --- Transformer Block ---
class TransformerBlock(nn.Module):
    def __init__(self, latent_dim, num_heads, ff_dim_multiplier=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * ff_dim_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * ff_dim_multiplier, latent_dim)
        )

    def forward(self, x):
        # Self-attention
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output

        # Feed forward
        ff_output = self.ff(self.norm2(x))
        x = x + ff_output
        return x

# --- Main Diffusion Model (Patch-based) ---
class DiffusionModel(nn.Module):
    def __init__(self,
                 img_size=64,
                 patch_size=8,
                 img_channels=3,
                 latent_dim=256,
                 time_dim=64,
                 num_transformer_blocks=4,
                 num_heads=8,
                 ff_dim_multiplier=4,
                 dropout=0.1
                ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.img_channels = img_channels
        self.latent_dim = latent_dim
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2  # 64x64 / 8x8 = 8x8 = 64 patches
        
        # 1. Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, latent_dim)  # Map to latent_dim for easier addition
        )

        # 2. Unified Patch Encoder (same weights for patches and prompt)
        self.patch_encoder = UnifiedPatchEncoder(output_dim=latent_dim)
        
        # 3. Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(latent_dim, num_heads, ff_dim_multiplier, dropout) 
            for _ in range(num_transformer_blocks)
        ])
        
        # 4. Patch Decoder
        self.patch_decoder = PatchDecoder(input_dim=latent_dim)
        
        # 5. Final layer norm
        self.final_norm = nn.LayerNorm(latent_dim)

    def extract_patches(self, x):
        """
        Extract 8x8 patches from 64x64 image
        Args:
            x: (B, 3, 64, 64)
        Returns:
            patches: (B, 64, 3, 8, 8)
        """
        B, C, H, W = x.shape
        # Use unfold to extract patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # Reshape: (B, C, num_patches_h, num_patches_w, patch_size, patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        # Permute to: (B, num_patches, C, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4)
        return patches

    def reconstruct_from_patches(self, patches):
        """
        Reconstruct 64x64 image from 8x8 patches
        Args:
            patches: (B, 64, 3, 8, 8)
        Returns:
            image: (B, 3, 64, 64)
        """
        B, num_patches, C, patch_h, patch_w = patches.shape
        patches_per_row = self.img_size // self.patch_size  # 8
        
        # Reshape patches to grid: (B, C, 8, 8, 8, 8)
        patches = patches.view(B, patches_per_row, patches_per_row, C, patch_h, patch_w)
        patches = patches.permute(0, 3, 1, 4, 2, 5)  # (B, C, 8, 8, 8, 8)
        
        # Reconstruct: (B, C, 64, 64)
        image = patches.contiguous().view(B, C, self.img_size, self.img_size)
        return image

    def forward(self, x_t: torch.Tensor, time: torch.Tensor, visual_context: torch.Tensor):
        """
        Args:
            x_t: (B, 3, 64, 64) - Noisy image
            time: (B,) - Timestep
            visual_context: (B, 3, 8, 8) - Visual conditioning prompt
        Returns:
            predicted_noise: (B, 3, 64, 64) - Predicted noise
        """
        B = x_t.shape[0]
        
        # 1. Extract patches from main image
        image_patches = self.extract_patches(x_t)  # (B, 64, 3, 8, 8)
        
        # 2. Encode all patches (batch processing)
        # Reshape for batch encoding: (B*64, 3, 8, 8)
        patches_flat = image_patches.reshape(-1, 3, self.patch_size, self.patch_size)
        encoded_patches = self.patch_encoder(patches_flat)  # (B*64, 256)
        encoded_patches = encoded_patches.reshape(B, self.num_patches, self.latent_dim)  # (B, 64, 256)
        
        # 3. Encode visual context (prompt)
        encoded_prompt = self.patch_encoder(visual_context)  # (B, 256)
        encoded_prompt = encoded_prompt.unsqueeze(1)  # (B, 1, 256)
        
        # 4. Encode time
        time_emb = self.time_mlp(time)  # (B, 256)
        time_emb = time_emb.unsqueeze(1)  # (B, 1, 256)
        
        # 5. Create sequence: [patches (64), prompt (1), time (1)] = 66 tokens
        sequence = torch.cat([encoded_patches, encoded_prompt, time_emb], dim=1)  # (B, 66, 256)
        
        # 6. Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            sequence = transformer_block(sequence)
        
        # 7. Extract patches from sequence (ignore prompt and time tokens)
        processed_patches = sequence[:, :self.num_patches, :]  # (B, 64, 256)
        processed_patches = self.final_norm(processed_patches)
        
        # 8. Decode patches back to image patches
        # Reshape for batch decoding: (B*64, 256)
        patches_flat = processed_patches.view(-1, self.latent_dim)
        decoded_patches = self.patch_decoder(patches_flat)  # (B*64, 3, 8, 8)
        decoded_patches = decoded_patches.view(B, self.num_patches, 3, self.patch_size, self.patch_size)  # (B, 64, 3, 8, 8)
        
        # 9. Reconstruct full image
        predicted_noise = self.reconstruct_from_patches(decoded_patches)  # (B, 3, 64, 64)
        
        return predicted_noise

# Test the model
if __name__ == '__main__':
    device = get_device()
    print(f"Using device: {device}")

    # Test the new patch-based model
    model = DiffusionModel(
        img_size=64,
        patch_size=8,
        img_channels=3,
        latent_dim=256,
        time_dim=64,
        num_transformer_blocks=4,
        num_heads=8
    ).to(device)

    batch_size = 2
    dummy_x_t = torch.randn(batch_size, 3, 64, 64, device=device)
    dummy_time = torch.randint(0, 200, (batch_size,), device=device)
    dummy_visual_context = torch.randn(batch_size, 3, 8, 8, device=device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input shapes:")
    print(f"  x_t: {dummy_x_t.shape}")
    print(f"  time: {dummy_time.shape}")
    print(f"  visual_context: {dummy_visual_context.shape}")

    try:
        # Test patch extraction and reconstruction
        patches = model.extract_patches(dummy_x_t)
        print(f"Extracted patches shape: {patches.shape}")
        
        reconstructed = model.reconstruct_from_patches(patches)
        print(f"Reconstructed shape: {reconstructed.shape}")
        
        # Test if reconstruction is perfect
        diff = torch.abs(dummy_x_t - reconstructed).max()
        print(f"Reconstruction error: {diff.item():.6f}")
        
        # Test full forward pass
        predicted_noise = model(dummy_x_t, dummy_time, dummy_visual_context)
        print(f"Predicted noise shape: {predicted_noise.shape}")
        assert predicted_noise.shape == dummy_x_t.shape
        print("✅ Patch-based model forward pass successful!")
        
        # Test patch encoder
        test_patch = torch.randn(1, 3, 8, 8, device=device)
        encoded = model.patch_encoder(test_patch)
        print(f"Patch encoding: {test_patch.shape} → {encoded.shape}")
        
        # Test patch decoder
        decoded = model.patch_decoder(encoded)
        print(f"Patch decoding: {encoded.shape} → {decoded.shape}")
        
        print("✅ All tests passed! New patch-based architecture working correctly!")

    except Exception as e:
        print(f"❌ Error during model test: {e}")
        import traceback
        traceback.print_exc()
