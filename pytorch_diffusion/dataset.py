import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random

class ImageDataset(Dataset):
    """
    Dataset for training the patch-based diffusion model.
    Loads 64x64 images and creates 8x8 visual context from the same image.
    """
    
    def __init__(self, root_dir, img_size=64, context_size=8, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
        """
        Args:
            root_dir: Root directory containing images
            img_size: Target size for main images (64)
            context_size: Size for visual context (8)
            extensions: Valid image file extensions
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.context_size = context_size
        
        # Find all image files
        self.image_paths = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(extensions):
                    self.image_paths.append(os.path.join(root, file))
        
        print(f"Found {len(self.image_paths)} images in {root_dir}")
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")
        
        # Main image transform (64x64) - SIMPLE [0, 1] NORMALIZATION
        self.main_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),  # [0, 1] - PERFECT for ReLU!
        ])
        
        # Context transform (8x8) - SIMPLE [0, 1] NORMALIZATION  
        self.context_transform = transforms.Compose([
            transforms.Resize((context_size, context_size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),  # [0, 1] - PERFECT for ReLU!
        ])
        
        # Augmentation for context (optional variety)
        self.context_augment = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.RandomHorizontalFlip(p=0.3),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Returns:
            main_image: Tensor (3, 64, 64) normalized to [0, 1]
            context_image: Tensor (3, 8, 8) normalized to [0, 1] 
            image_path: String path to the image file
        """
        image_path = self.image_paths[idx]
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply some augmentation before creating context (for variety)
            if random.random() < 0.5:
                image = self.context_augment(image)
            
            # Create main image (64x64)
            main_image = self.main_transform(image)
            
            # Create context image (8x8) - same image but different processing path
            context_image = self.context_transform(image)
            
            return main_image, context_image, image_path
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a random other image if this one fails
            new_idx = random.randint(0, len(self.image_paths) - 1)
            return self.__getitem__(new_idx)

# Test dataset functionality
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Test with a sample directory (adjust path as needed)
    dataset_path = "H:/Repositories/ImgDatasetGen_256"  # Adjust this path
    
    if os.path.exists(dataset_path):
        print(f"Testing dataset with path: {dataset_path}")
        
        dataset = ImageDataset(dataset_path, img_size=64, context_size=8)
        print(f"Dataset size: {len(dataset)}")
        
        # Test a few samples
        for i in range(min(3, len(dataset))):
            main_img, context_img, path = dataset[i]
            
            print(f"Sample {i}:")
            print(f"  Main image shape: {main_img.shape}")
            print(f"  Context image shape: {context_img.shape}")
            print(f"  Main image range: [{main_img.min():.3f}, {main_img.max():.3f}]")
            print(f"  Context image range: [{context_img.min():.3f}, {context_img.max():.3f}]")
            print(f"  Path: {os.path.basename(path)}")
            
            # Test tensor to PIL conversion
            def tensor_to_pil_test(tensor):
                # tensor is already [0,1], no conversion needed!
                tensor = tensor.clamp(0, 1)
                return transforms.ToPILImage()(tensor)
            
            main_pil = tensor_to_pil_test(main_img)
            context_pil = tensor_to_pil_test(context_img)
            
            print(f"  Main PIL size: {main_pil.size}")
            print(f"  Context PIL size: {context_pil.size}")
            print()
        
        # Test DataLoader
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
        
        for batch_idx, (main_batch, context_batch, paths) in enumerate(dataloader):
            print(f"Batch {batch_idx}:")
            print(f"  Main batch shape: {main_batch.shape}")
            print(f"  Context batch shape: {context_batch.shape}")
            print(f"  Batch size: {len(paths)}")
            
            if batch_idx >= 2:  # Test only a few batches
                break
        
        print("✅ Dataset working correctly!")
        
    else:
        print(f"❌ Dataset path {dataset_path} not found. Please adjust the path in the test.")
        print("Creating a simple test with dummy data...")
        
        # Create a minimal test without real images
        try:
            # This will fail but we can catch it
            dataset = ImageDataset("nonexistent_path")
        except ValueError as e:
            print(f"Expected error: {e}")
            print("✅ Error handling working correctly!")
