import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .utils import get_device
except ImportError:
    # For direct execution
    from utils import get_device

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

# --- Visual Conditioning CNN Encoder ---
class VisualConditioningEncoder(nn.Module):
    """Encode 8x8x3 visual prompt to 192-dimensional features"""
    def __init__(self, output_dim=192):
        super().__init__()
        self.output_dim = output_dim
        
        # 8x8x3 → 8x8x32 → 4x4x64 → 2x2x64 → 192
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # 8x8x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 4x4x64
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)  # 2x2x64
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 2 * 2, output_dim)  # 256 → 192
        
        self.activation = nn.ReLU()
        self.norm1 = nn.InstanceNorm2d(32)
        self.norm2 = nn.InstanceNorm2d(64)
        self.norm3 = nn.InstanceNorm2d(64)
    
    def forward(self, x):
        # x: (B, 3, 8, 8)
        x = self.activation(self.norm1(self.conv1(x)))  # (B, 32, 8, 8)
        x = self.activation(self.norm2(self.conv2(x)))  # (B, 64, 4, 4)
        x = self.activation(self.norm3(self.conv3(x)))  # (B, 64, 2, 2)
        
        x = self.flatten(x)  # (B, 256)
        x = self.fc(x)       # (B, 192)
        return x

# --- Building Blocks ---
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_norm=True, activation='relu'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels) if use_norm else nn.Identity()
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation is None:
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class TransposeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, activation='relu'):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        x = self.tconv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

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
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output

        ff_output = self.ff(self.norm2(x))
        x = x + ff_output
        return x

# --- Diffusion Model V2 (64x64 with Visual Conditioning) ---
class DiffusionModelV2(nn.Module):
    def __init__(self,
                 img_size=64,
                 img_channels=3,
                 latent_dim=256,
                 time_dim=64,
                 visual_context_dim=192,  # Output from VisualConditioningEncoder
                 num_transformer_blocks=3,
                 num_heads=4,
                 initial_conv_filters=32,
                 conv_dim_mults=(1, 2, 4, 8)  # For 64x64: 64→32→16→8→4
                ):
        super().__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.latent_dim = latent_dim
        self.visual_context_dim = visual_context_dim

        # 1. Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        # 2. Visual Conditioning Encoder
        self.visual_encoder = VisualConditioningEncoder(output_dim=visual_context_dim)
        self.context_mlp = nn.Sequential(
            nn.Linear(visual_context_dim, visual_context_dim * 2),
            nn.GELU(),
            nn.Linear(visual_context_dim * 2, visual_context_dim)
        )

        # 3. Image Encoder
        current_channels_enc = img_channels
        encoder_blocks_list = []

        encoder_blocks_list.append(ConvBlock(current_channels_enc, initial_conv_filters, kernel_size=3, padding=1))
        current_channels_enc = initial_conv_filters

        for mult_factor in conv_dim_mults:
            out_filters = initial_conv_filters * mult_factor
            encoder_blocks_list.append(ConvBlock(current_channels_enc, out_filters, kernel_size=3, stride=2, padding=1))
            current_channels_enc = out_filters

        self.encoder_blocks = nn.Sequential(*encoder_blocks_list)

        self.final_encoder_img_size = img_size // (2**len(conv_dim_mults))
        self.encoder_output_channels = current_channels_enc
        flattened_size = self.encoder_output_channels * (self.final_encoder_img_size**2)

        self.to_latent_projection = nn.Linear(flattened_size, latent_dim)

        # 4. Transformer Blocks
        self.transformer_blocks_list = nn.ModuleList([
            TransformerBlock(latent_dim, num_heads) for _ in range(num_transformer_blocks)
        ])

        fusion_input_dim = latent_dim + time_dim + visual_context_dim

        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim)
        )

        # 5. Decoder
        self.from_latent_projection = nn.Linear(latent_dim, flattened_size)

        decoder_blocks_list = []
        current_channels_dec = self.encoder_output_channels

        for i, mult_factor in enumerate(reversed(conv_dim_mults)):
            if i == len(conv_dim_mults) - 1:
                out_dec_filters = initial_conv_filters
            else:
                prev_enc_mult_factor_idx = len(conv_dim_mults) - 1 - i - 1
                out_dec_filters = initial_conv_filters * conv_dim_mults[prev_enc_mult_factor_idx]

            decoder_blocks_list.append(TransposeConvBlock(current_channels_dec, out_dec_filters, kernel_size=3, stride=2, padding=1, output_padding=1))
            current_channels_dec = out_dec_filters

        decoder_blocks_list.append(ConvBlock(initial_conv_filters, img_channels, kernel_size=3, padding=1, activation=None, use_norm=False))
        self.decoder_blocks = nn.Sequential(*decoder_blocks_list)

    def forward(self, x_t: torch.Tensor, time: torch.Tensor, visual_context: torch.Tensor):
        """
        Args:
            x_t: (B, 3, 64, 64) - Noisy image
            time: (B,) - Timestep
            visual_context: (B, 3, 8, 8) - Visual conditioning prompt
        """
        # Encode main image
        img_encoded = self.encoder_blocks(x_t)
        img_encoded_flat = img_encoded.view(img_encoded.size(0), -1)
        img_latent = self.to_latent_projection(img_encoded_flat)

        # Encode time
        time_emb = self.time_mlp(time)

        # Encode visual context
        visual_features = self.visual_encoder(visual_context)
        visual_emb = self.context_mlp(visual_features)

        # Fusion
        fused_features = torch.cat([img_latent, time_emb, visual_emb], dim=-1)
        transformer_input = self.fusion_layer(fused_features)
        transformer_input = transformer_input.unsqueeze(1)

        # Transformer processing
        transformer_output = transformer_input
        for block in self.transformer_blocks_list:
            transformer_output = block(transformer_output)

        transformer_output = transformer_output.squeeze(1)

        # Decode
        decoder_input_flat = self.from_latent_projection(transformer_output)
        decoder_input_spatial = decoder_input_flat.view(
            -1,
            self.encoder_output_channels,
            self.final_encoder_img_size,
            self.final_encoder_img_size
        )

        predicted_noise = self.decoder_blocks(decoder_input_spatial)
        return predicted_noise

# Example Usage and Test:
if __name__ == '__main__':
    device = get_device()
    print(f"Using device: {device}")

    # Test the new model
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

    batch_size = 2
    dummy_x_t = torch.randn(batch_size, 3, 64, 64, device=device)
    dummy_time = torch.randint(0, 200, (batch_size,), device=device)
    dummy_visual_context = torch.randn(batch_size, 3, 8, 8, device=device)

    print(f"Input shapes:")
    print(f"  x_t: {dummy_x_t.shape}")
    print(f"  time: {dummy_time.shape}")
    print(f"  visual_context: {dummy_visual_context.shape}")

    try:
        predicted_noise = model(dummy_x_t, dummy_time, dummy_visual_context)
        print(f"Predicted noise shape: {predicted_noise.shape}")
        assert predicted_noise.shape == dummy_x_t.shape
        print("✅ Model V2 forward pass successful!")
        
        # Test visual encoder separately
        visual_features = model.visual_encoder(dummy_visual_context)
        print(f"Visual conditioning features: {visual_features.shape}")
        assert visual_features.shape == (batch_size, 192)
        print("✅ Visual conditioning encoder working!")

    except Exception as e:
        print(f"❌ Error during model test: {e}")
        import traceback
        traceback.print_exc()
