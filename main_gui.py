#!/usr/bin/env python3
"""
Main launcher for the PyTorch Diffusion GUI application
Run this from the root directory to launch the complete application with V2 tabs
"""

import sys
import os

# Add the current directory to Python path so modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import and run the main application
from pytorch_diffusion_qt.main_window import main

if __name__ == '__main__':
    print("🚀 Launching PyTorch Diffusion GUI with V2 Components...")
    print("Features:")
    print("  - Training V2: High batch sizes (1-16k), loss graph, multi-GPU")
    print("  - Inference V2: Paint 8x8 widget, manual step-by-step diffusion")
    print("  - Original tabs: Dataset, Training, Weights, Inference")
    print("")
    main()
