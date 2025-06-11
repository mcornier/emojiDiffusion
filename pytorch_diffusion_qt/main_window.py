import sys
import os
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QDesktopWidget,
    QFileDialog, QMessageBox, QLabel # QLabel might be used if a tab fails to load
)
from PyQt5.QtCore import Qt, pyqtSlot

# Actual Tab Imports
from .dataset_tab import DatasetTab
from .training_tab import TrainingTab
from .weights_tab import WeightsTab
from .inference_tab import InferenceTab

# Backend imports
from pytorch_diffusion.model import DiffusionModel
from pytorch_diffusion.utils import get_device

# Attempt to get TIMESTEPS from utils, with a fallback
try:
    from pytorch_diffusion.utils import TIMESTEPS as UTILS_TIMESTEPS
except ImportError:
    print("Warning: Could not import TIMESTEPS from pytorch_diffusion.utils. Using default value 200.")
    UTILS_TIMESTEPS = 200


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
        self.timesteps = UTILS_TIMESTEPS

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
            self.dataset_tab = DatasetTab()
            self.tab_widget.addTab(self.dataset_tab, "Dataset")
        except Exception as e:
            print(f"Error loading DatasetTab: {e}")
            self.tab_widget.addTab(QLabel(f"Error loading DatasetTab: {e}"), "Dataset")

        try:
            self.training_tab = TrainingTab()
            self.tab_widget.addTab(self.training_tab, "Training")
        except Exception as e:
            print(f"Error loading TrainingTab: {e}")
            self.tab_widget.addTab(QLabel(f"Error loading TrainingTab: {e}"), "Training")

        try:
            self.weights_tab = WeightsTab()
            self.tab_widget.addTab(self.weights_tab, "Weights")
        except Exception as e:
            print(f"Error loading WeightsTab: {e}")
            self.tab_widget.addTab(QLabel(f"Error loading WeightsTab: {e}"), "Weights")

        try:
            self.inference_tab = InferenceTab()
            self.tab_widget.addTab(self.inference_tab, "Inference")
        except Exception as e:
            print(f"Error loading InferenceTab: {e}")
            self.tab_widget.addTab(QLabel(f"Error loading InferenceTab: {e}"), "Inference")


    def _connect_signals(self):
        """Connects signals from tabs to their respective handlers in the main window or other tabs."""
        # DatasetTab -> TrainingTab & MainWindow
        if hasattr(self, 'dataset_tab') and hasattr(self, 'training_tab'):
            self.dataset_tab.dataset_updated.connect(self.training_tab.update_active_dataset_info)
            self.dataset_tab.dataset_updated.connect(self._update_main_dataset_info)

        # WeightsTab -> MainWindow
        if hasattr(self, 'weights_tab'):
            self.weights_tab.request_model_export.connect(self.handle_export_weights)
            self.weights_tab.request_model_import.connect(self.handle_import_weights)

    @pyqtSlot(list, dict, str, int) # emojis, emoji_to_idx, font_path, vocab_size
    def _update_main_dataset_info(self, emojis, emoji_to_idx, font_path, vocab_size):
        self.model_emoji_list = emojis
        self.model_emoji_to_idx = emoji_to_idx
        self.model_font_path = font_path
        self.model_vocab_size = vocab_size # Usually len(emojis) or len(emoji_to_idx)
        print(f"Main window updated with dataset: {len(emojis)} emojis, vocab size {vocab_size}. Font: {font_path if font_path else 'Default'}")

        # If a model is loaded, its parameters might now be inconsistent with this new dataset.
        # This is a point of complex interaction. For now, we just update the main window's state.
        # The inference tab might need to be updated if it was using the old dataset context.
        if self.diffusion_model and hasattr(self, 'inference_tab'):
             # If the new vocab size from dataset doesn't match loaded model, it's an issue.
            if self.diffusion_model.emoji_vocab_size != vocab_size:
                QMessageBox.warning(self, "Dataset/Model Vocab Mismatch",
                                    f"The newly generated dataset has a vocabulary size ({vocab_size}) "
                                    f"that differs from the loaded model's vocabulary size ({self.diffusion_model.emoji_vocab_size}). "
                                    "Inference may not work as expected. Consider re-training or loading a compatible model.")
            # Update inference tab with potentially new emoji list/map from the new dataset,
            # but keep the existing loaded model.
            self.inference_tab.set_model_and_vocab(
                self.diffusion_model, # existing model
                self.model_emoji_list, # new emoji list from dataset
                self.model_emoji_to_idx, # new map from dataset
                self.timesteps
            )


    def handle_export_weights(self, suggested_path_ignored): # Path from signal is ignored for export
        if not self.diffusion_model:
            if hasattr(self, 'weights_tab'): self.weights_tab.update_export_status("No model loaded to export.")
            QMessageBox.warning(self, "Export Error", "No model is currently loaded.")
            return

        # Default filename suggestion
        default_filename = f"diffusion_model_vocab{self.model_vocab_size}_img{self.model_img_size}.pth"

        filePath, _ = QFileDialog.getSaveFileName(
            self, "Export Model Weights", default_filename, "PyTorch Model Files (*.pth *.pt)"
        )
        if filePath:
            try:
                # Save more metadata with the model for easier reloading
                save_dict = {
                    'model_state_dict': self.diffusion_model.state_dict(),
                    'img_size': self.model_img_size,
                    'emoji_vocab_size': self.model_vocab_size,
                    'img_channels': self.diffusion_model.img_channels,
                    'latent_dim': self.diffusion_model.latent_dim,
                    'time_dim': self.diffusion_model.time_dim,
                    'context_dim': self.diffusion_model.context_dim,
                    'num_transformer_blocks': self.diffusion_model.num_transformer_blocks,
                    'num_heads': self.diffusion_model.num_heads,
                    'initial_conv_filters': self.diffusion_model.initial_conv_filters,
                    # Optionally, also save self.model_emoji_list and self.model_emoji_to_idx
                    # 'emoji_list': self.model_emoji_list,
                    # 'emoji_to_idx': self.model_emoji_to_idx
                }
                torch.save(save_dict, filePath)
                if hasattr(self, 'weights_tab'): self.weights_tab.update_export_status(f"Model exported to {os.path.basename(filePath)}")
                QMessageBox.information(self, "Export Successful", f"Model weights and metadata exported to {filePath}")
            except Exception as e:
                if hasattr(self, 'weights_tab'): self.weights_tab.update_export_status(f"Export failed: {str(e)}")
                QMessageBox.critical(self, "Export Error", f"Failed to export model: {str(e)}")
        else:
            if hasattr(self, 'weights_tab'): self.weights_tab.update_export_status("Export cancelled.")

    def handle_import_weights(self, filePath):
        if not filePath:
            if hasattr(self, 'weights_tab'): self.weights_tab.update_import_status("Import path is invalid.")
            return

        device = get_device()
        try:
            checkpoint = torch.load(filePath, map_location=device)

            # Determine if it's a state_dict directly or a dictionary containing it and metadata
            if 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
                loaded_img_size = checkpoint.get('img_size', self.model_img_size) # Use current default if not in ckpt
                loaded_vocab_size = checkpoint.get('emoji_vocab_size')
                # Other parameters for model instantiation
                img_channels = checkpoint.get('img_channels', 3)
                latent_dim = checkpoint.get('latent_dim', 256)
                time_dim = checkpoint.get('time_dim', 64) # Assuming model.py defaults
                context_dim = checkpoint.get('context_dim', 64)
                num_transformer_blocks = checkpoint.get('num_transformer_blocks', 3)
                num_heads = checkpoint.get('num_heads', 4)
                initial_conv_filters = checkpoint.get('initial_conv_filters', 32)
                # Potentially load emoji list/map if saved
                # self.model_emoji_list = checkpoint.get('emoji_list', self.model_emoji_list)
                # self.model_emoji_to_idx = checkpoint.get('emoji_to_idx', self.model_emoji_to_idx)

            else: # Assume it's just a state_dict
                model_state_dict = checkpoint
                # We lack metadata, try to use current settings or defaults. This is risky.
                loaded_img_size = self.model_img_size # Use current setting
                loaded_vocab_size = self.model_vocab_size # Use current setting (could be from dataset)
                img_channels = 3 # Default
                latent_dim = 256 # Default
                # ... set other params to defaults ...
                time_dim, context_dim, num_transformer_blocks, num_heads, initial_conv_filters = 64, 64, 3, 4, 32
                QMessageBox.warning(self, "Import Warning",
                                  "Loading a raw state_dict. Model parameters (img_size, vocab_size, etc.) "
                                  "are assumed from current settings or defaults. This may fail if incompatible.")

            if loaded_vocab_size is None:
                # Attempt to infer from current dataset if available
                if self.model_emoji_to_idx:
                    loaded_vocab_size = len(self.model_emoji_to_idx)
                    QMessageBox.warning(self, "Import Warning",
                                      f"Checkpoint did not contain 'emoji_vocab_size'. "
                                      f"Inferred from current dataset: {loaded_vocab_size}. "
                                      "Ensure this is correct for the loaded model.")
                else:
                    raise ValueError("Cannot determine vocabulary size for the imported model. "
                                     "Checkpoint must include 'emoji_vocab_size' or a dataset providing this context must be generated first.")

            self.diffusion_model = DiffusionModel(
                img_size=loaded_img_size,
                emoji_vocab_size=loaded_vocab_size,
                img_channels=img_channels,
                latent_dim=latent_dim,
                time_dim=time_dim,
                context_dim=context_dim,
                num_transformer_blocks=num_transformer_blocks,
                num_heads=num_heads,
                initial_conv_filters=initial_conv_filters
            ).to(device)

            self.diffusion_model.load_state_dict(model_state_dict)
            self.diffusion_model.eval()

            # Update main window's understanding of the current model
            self.model_img_size = loaded_img_size
            self.model_vocab_size = loaded_vocab_size
            # Note: self.model_emoji_list and self.model_emoji_to_idx might be from the *current dataset*,
            # not necessarily the one the model was trained on, unless they were also in the checkpoint.
            # This is a crucial point for data consistency.

            if hasattr(self, 'weights_tab'): self.weights_tab.update_import_status(f"Model imported from {os.path.basename(filePath)}")
            QMessageBox.information(self, "Import Successful", f"Model weights imported from {filePath} (Vocab: {loaded_vocab_size}, Img: {loaded_img_size}x{loaded_img_size})")

            if hasattr(self, 'inference_tab'):
                self.inference_tab.set_model_and_vocab(
                    self.diffusion_model,
                    self.model_emoji_list, # This is from the current dataset, ensure user knows
                    self.model_emoji_to_idx, # Same as above
                    self.timesteps
                )
            # TODO: Notify TrainingTab if it needs to know about the loaded model for fine-tuning
            # self.training_tab.set_loaded_model_info(self.diffusion_model, loaded_vocab_size, loaded_img_size)

        except Exception as e:
            import traceback
            print(f"Import Error: {str(e)}\n{traceback.format_exc()}")
            self.diffusion_model = None
            if hasattr(self, 'weights_tab'): self.weights_tab.update_import_status(f"Import failed: {str(e)}")
            QMessageBox.critical(self, "Import Error", f"Failed to import model: {str(e)}\nEnsure the checkpoint is valid and compatible. It's best if checkpoints contain metadata like 'emoji_vocab_size' and 'img_size'.")
            if hasattr(self, 'inference_tab'):
                self.inference_tab.set_model_and_vocab(None, None, None, self.timesteps)


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
