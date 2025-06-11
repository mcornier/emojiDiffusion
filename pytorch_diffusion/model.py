import torch
import torch.nn as nn
import torch.nn.functional as F
# Assuming utils.py is in pytorch_diffusion directory, and model.py is also there,
# the import should be:
from .utils import get_device

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
        # Ensure time is correctly shaped for broadcasting, e.g., (batch_size, 1)
        if time.ndim == 1:
            time = time.unsqueeze(1)
        embeddings = time * embeddings.unsqueeze(0) # time: (B,1), embeddings: (1, half_dim) -> (B, half_dim)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1: # Zero pad if dim is odd
            embeddings = F.pad(embeddings, (0,1))
        return embeddings

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
        elif activation is None: # Allow no activation
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

    def forward(self, x): # x is (batch_size, seq_len, latent_dim)
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm) # Q, K, V
        x = x + attn_output

        ff_output = self.ff(self.norm2(x))
        x = x + ff_output
        return x

# --- Diffusion Model ---
class DiffusionModel(nn.Module):
    def __init__(self,
                 img_size=16,
                 img_channels=3,
                 latent_dim=256,
                 time_dim=64,
                 context_dim=64,
                 emoji_vocab_size=10, # Must be set based on actual dataset
                 num_transformer_blocks=3,
                 num_heads=4,
                 initial_conv_filters=32,
                 conv_dim_mults=(1, 2) # Default for 16x16 to get 4x4 spatial: 16->8 (via mult 1), 8->4 (via mult 2)
                ):
        super().__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.latent_dim = latent_dim
        # self.device = get_device() # Model should be device-agnostic until .to(device) is called

        # 1. Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim), # Output: (B, time_dim)
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim) # Output: (B, time_dim)
        )

        # 2. Context Embedding (Optional)
        self.use_context = emoji_vocab_size > 0 and context_dim > 0
        if self.use_context:
            self.context_embedding_layer = nn.Embedding(emoji_vocab_size, context_dim)
            self.context_mlp = nn.Sequential(
                nn.Linear(context_dim, context_dim * 4),
                nn.GELU(),
                nn.Linear(context_dim * 4, context_dim)
            )
            actual_context_dim = context_dim
        else:
            actual_context_dim = 0
            self.context_embedding_layer = None # Explicitly set to None
            self.context_mlp = None


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

        fusion_input_dim = latent_dim + time_dim + actual_context_dim

        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim)
        )

        # 5. Decoder
        self.from_latent_projection = nn.Linear(latent_dim, flattened_size)

        decoder_blocks_list = []
        current_channels_dec = self.encoder_output_channels # Start with channels from end of encoder

        for i, mult_factor in enumerate(reversed(conv_dim_mults)):
            # Determine the output channels for this upsampling block
            # It should match the channel count of the corresponding encoder stage (before downsampling)
            if i == len(conv_dim_mults) - 1: # This is the last upsampling block (corresponds to first conv_dim_mults factor)
                out_dec_filters = initial_conv_filters
            else:
                # Corresponds to encoder block whose output mult was conv_dim_mults[len(conv_dim_mults) - 1 - i - 1]
                prev_enc_mult_factor_idx = len(conv_dim_mults) - 1 - i - 1
                out_dec_filters = initial_conv_filters * conv_dim_mults[prev_enc_mult_factor_idx]

            decoder_blocks_list.append(TransposeConvBlock(current_channels_dec, out_dec_filters, kernel_size=3, stride=2, padding=1, output_padding=1))
            current_channels_dec = out_dec_filters

        # After upsampling, current_channels_dec should be initial_conv_filters.
        # Final convolution to restore image channels.
        decoder_blocks_list.append(ConvBlock(initial_conv_filters, img_channels, kernel_size=3, padding=1, activation=None, use_norm=False))
        self.decoder_blocks = nn.Sequential(*decoder_blocks_list)


    def forward(self, x_t: torch.Tensor, time: torch.Tensor, context_idx: torch.Tensor = None):
        # x_t: (B, C, H, W)
        # time: (B,)
        # context_idx: (B,) or (B, 1)

        img_encoded = self.encoder_blocks(x_t)
        img_encoded_flat = img_encoded.view(img_encoded.size(0), -1)
        img_latent = self.to_latent_projection(img_encoded_flat)

        time_emb = self.time_mlp(time)

        fused_elements = [img_latent, time_emb]
        if self.use_context and context_idx is not None and self.context_embedding_layer is not None:
            context_idx = context_idx.long() # Ensure long type for embedding
            if context_idx.ndim == 2 and context_idx.shape[1] == 1:
                context_idx = context_idx.squeeze(1)

            context_emb_raw = self.context_embedding_layer(context_idx)
            context_emb = self.context_mlp(context_emb_raw)
            fused_elements.append(context_emb)

        fused_features = torch.cat(fused_elements, dim=-1)
        transformer_input = self.fusion_layer(fused_features)
        transformer_input = transformer_input.unsqueeze(1)

        transformer_output = transformer_input
        for block in self.transformer_blocks_list:
            transformer_output = block(transformer_output)

        transformer_output = transformer_output.squeeze(1)

        decoder_input_flat = self.from_latent_projection(transformer_output)
        decoder_input_spatial = decoder_input_flat.view(
            -1,
            self.encoder_output_channels,
            self.final_encoder_img_size,
            self.final_encoder_img_size
        )

        predicted_noise = self.decoder_blocks(decoder_input_spatial)
        return predicted_noise

# Example Usage:
if __name__ == '__main__':
    device = get_device()
    print(f"Using device: {device}")

    IMG_S, IMG_C, LATENT_D, TIME_D, CONTEXT_D = 16, 3, 256, 64, 64
    EMOJI_VOCAB_LEN, NUM_TRANSFORMER_BLOCKS, NUM_HEADS = 20, 3, 4
    INITIAL_FILTERS = 32
    CONV_MULTS = (1, 2)

    test_model = DiffusionModel(
        img_size=IMG_S, img_channels=IMG_C, latent_dim=LATENT_D, time_dim=TIME_D,
        context_dim=CONTEXT_D, emoji_vocab_size=EMOJI_VOCAB_LEN,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS, num_heads=NUM_HEADS,
        initial_conv_filters=INITIAL_FILTERS, conv_dim_mults=CONV_MULTS
    ).to(device)

    batch_size = 4
    dummy_x_t = torch.randn(batch_size, IMG_C, IMG_S, IMG_S, device=device)
    dummy_time = torch.randint(0, 200, (batch_size,), device=device)
    dummy_context = torch.randint(0, EMOJI_VOCAB_LEN, (batch_size,1), device=device)

    print(f"Model 'emoji_vocab_size': {test_model.context_embedding_layer.num_embeddings if test_model.use_context and test_model.context_embedding_layer else 'N/A'}")
    print(f"Input x_t: {dummy_x_t.shape}, time: {dummy_time.shape}, context: {dummy_context.shape}")

    try:
        predicted_noise = test_model(dummy_x_t, dummy_time, dummy_context)
        print(f"Predicted noise shape (with context): {predicted_noise.shape}")
        assert predicted_noise.shape == dummy_x_t.shape
        print("Model forward pass successful (with context)!")

        test_model_no_ctx_cfg = DiffusionModel(
            img_size=IMG_S, img_channels=IMG_C, latent_dim=LATENT_D, time_dim=TIME_D,
            context_dim=0, emoji_vocab_size=0,
            initial_conv_filters=INITIAL_FILTERS, conv_dim_mults=CONV_MULTS,
            num_transformer_blocks=NUM_TRANSFORMER_BLOCKS, num_heads=NUM_HEADS
        ).to(device)

        predicted_noise_no_ctx = test_model_no_ctx_cfg(dummy_x_t, dummy_time, None)
        print(f"Predicted noise shape (no context): {predicted_noise_no_ctx.shape}")
        assert predicted_noise_no_ctx.shape == dummy_x_t.shape
        print("Model forward pass successful (no context)!")

    except Exception as e:
        print(f"Error during model test: {e}")
        import traceback
        traceback.print_exc()
