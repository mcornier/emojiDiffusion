import sys
import os
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QDesktopWidget,
    QFileDialog, QMessageBox, QLabel # QLabel might be used if a tab fails to load
)
from PyQt5.QtCore import Qt, pyqtSlot

# Main Tab Imports (V2 versions are now the main ones)
try:
    from .training_tab_v2 import TrainingTabV2
    from .inference_tab_v2 import InferenceTabV2
except ImportError:
    # Fallback for direct execution
    from training_tab_v2 import TrainingTabV2
    from inference_tab_v2 import InferenceTabV2

# Backend imports
from pytorch_diffusion.model_v2 import DiffusionModelV2
from pytorch_diffusion.utils import get_device


class DiffusionAppMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyTorch Emoji Diffusion GUI")

        # Model and dataset related attributes
        self.diffusion_model = None
        self.model_img_size = 16  # Default, should be updated on model load
        self.model_vocab_size = None
        self.model_emoji_list = None # List of emojis in the current vocab
        self.model_emoji_to_idx = None # Mapping for the current vocab
        self.model_font_path = None    # Font path used for the current model's dataset basis
        self.timesteps = 200  # V2 uses 200 timesteps

        # Center window on screen
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())
        self.setMinimumSize(800, 700) # Increased min height for more content

        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        self._create_tabs()
        self._connect_signals()


    def _create_tabs(self):
        """Initializes and adds all the primary tabs to the main tab widget."""
        try:
            self.training_tab = TrainingTabV2()
            self.tab_widget.addTab(self.training_tab, "Training")
        except Exception as e:
            print(f"Error loading TrainingTab: {e}")
            self.tab_widget.addTab(QLabel(f"Error loading TrainingTab: {e}"), "Training")

        try:
            self.inference_tab = InferenceTabV2()
            self.tab_widget.addTab(self.inference_tab, "Inference")
        except Exception as e:
            print(f"Error loading InferenceTab: {e}")
            self.tab_widget.addTab(QLabel(f"Error loading InferenceTab: {e}"), "Inference")


    def _connect_signals(self):
        """Connects signals from tabs to their respective handlers in the main window or other tabs."""
        # V2 tabs are self-contained and don't need complex signal connections
        pass


def main():
    app = QApplication(sys.argv)
    # Apply a basic style
    try:
        import qdarkstyle
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    except ImportError:
        print("qdarkstyle not found. Using default Qt style. Consider 'pip install qdarkstyle'.")
        # app.setStyle("Fusion") # Or another built-in style

    main_window = DiffusionAppMainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
