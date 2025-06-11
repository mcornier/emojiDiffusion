import torch
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import os
import glob
from pathlib import Path

class ImageDatasetV2(torch.utils.data.Dataset):
    """
    Dataset for loading 256x256 images and creating pairs:
    - Main image: 64x64x3 (target for diffusion)
    - Visual context: 8x8x3 (conditioning prompt)
    """
    def __init__(self, dataset_root, image_size=64, context_size=8, extensions=('*.jpg', '*.jpeg', '*.png', '*.bmp')):
        """
        Args:
            dataset_root: Path to H:/Repositories/ImgDatasetGen_256
            image_size: Target size for main image (default 64)
            context_size: Target size for visual context (default 8)
            extensions: Image file extensions to search for
        """
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.image_size = image_size
        self.context_size = context_size
        
        # Find all image files in L01-5k, L02-5k, etc.
        self.image_paths = []
        self._discover_images(extensions)
        
        print(f"Found {len(self.image_paths)} images in dataset")
        
        # Transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
        
    def _discover_images(self, extensions):
        """Discover all images in L##-5k directories"""
        pattern_dirs = list(self.dataset_root.glob("L*-5k"))
        print(f"Found {len(pattern_dirs)} L##-5k directories")
        
        for directory in pattern_dirs:
            if directory.is_dir():
                for ext in extensions:
                    files = list(directory.glob(ext))
                    self.image_paths.extend(files)
                    print(f"  {directory.name}: {len(files)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        try:
            pil_image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            pil_image = Image.new('RGB', (256, 256), color=(0, 0, 0))
        
        # Resize to target sizes
        main_image = pil_image.resize((self.image_size, self.image_size), Image.LANCZOS)
        context_image = pil_image.resize((self.context_size, self.context_size), Image.LANCZOS)
        
        # Convert to tensors and normalize
        main_tensor = self.normalize(self.to_tensor(main_image))      # (3, 64, 64) in [-1, 1]
        context_tensor = self.normalize(self.to_tensor(context_image)) # (3, 8, 8) in [-1, 1]
        
        return main_tensor, context_tensor, str(img_path)
    
    def get_sample_batch(self, batch_size=4):
        """Get a sample batch for testing"""
        indices = torch.randint(0, len(self), (batch_size,))
        batch_main = []
        batch_context = []
        batch_paths = []
        
        for idx in indices:
            main, context, path = self[idx]
            batch_main.append(main)
            batch_context.append(context)
            batch_paths.append(path)
        
        return torch.stack(batch_main), torch.stack(batch_context), batch_paths

def create_visual_conditioning_dataset(dataset_root, image_size=64, context_size=8):
    """
    Create dataset for visual conditioning diffusion model
    
    Returns:
        dataset: ImageDatasetV2 instance
        dataloader: DataLoader for training
    """
    dataset = ImageDatasetV2(dataset_root, image_size, context_size)
    
    # Create DataLoader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=8,  # Will be adjusted for multi-GPU
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    return dataset, dataloader

# Test dataset discovery and loading
if __name__ == '__main__':
    # Test dataset loading
    dataset_path = "H:/Repositories/ImgDatasetGen_256"
    
    if os.path.exists(dataset_path):
        print(f"Testing dataset loading from: {dataset_path}")
        
        try:
            dataset = ImageDatasetV2(dataset_path)
            print(f"Dataset size: {len(dataset)}")
            
            if len(dataset) > 0:
                # Test loading a few samples
                main, context, path = dataset[0]
                print(f"Sample 0:")
                print(f"  Main image: {main.shape}, range: [{main.min():.3f}, {main.max():.3f}]")
                print(f"  Context: {context.shape}, range: [{context.min():.3f}, {context.max():.3f}]")
                print(f"  Path: {path}")
                
                # Test batch loading
                main_batch, context_batch, paths = dataset.get_sample_batch(4)
                print(f"\nBatch test:")
                print(f"  Main batch: {main_batch.shape}")
                print(f"  Context batch: {context_batch.shape}")
                
                # Test DataLoader
                from torch.utils.data import DataLoader
                dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
                batch_main, batch_context, batch_paths = next(iter(dataloader))
                print(f"\nDataLoader test:")
                print(f"  Main batch: {batch_main.shape}")
                print(f"  Context batch: {batch_context.shape}")
                print("✅ Dataset V2 working correctly!")
                
            else:
                print("❌ No images found in dataset")
                
        except Exception as e:
            print(f"❌ Error testing dataset: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ Dataset path not found: {dataset_path}")
        print("Creating a simulated test...")
        
        # Create minimal test with dummy data
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size=100):
                self.size = size
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                main = torch.randn(3, 64, 64)
                context = torch.randn(3, 8, 8)
                return main, context, f"dummy_{idx}.jpg"
        
        dummy_dataset = DummyDataset(100)
        main, context, path = dummy_dataset[0]
        print(f"Dummy test:")
        print(f"  Main: {main.shape}")
        print(f"  Context: {context.shape}")
        print("✅ Dataset structure working!")
