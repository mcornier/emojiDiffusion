from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton,
    QScrollArea, QGridLayout, QFileDialog, QLineEdit, QGroupBox, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import sys
import os

# Make sure the pytorch_diffusion directory is in Python's search path
# This might be needed if running this tab's code directly for testing,
# or if the main app structure doesn't handle it.
# For a structured app, this kind of path manipulation should ideally be in the main entry point.
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

try:
    from pytorch_diffusion.dataset import create_emoji_dataset, generate_emoji_image
    from pytorch_diffusion.utils import tensor_to_pil # For previewing
except ImportError as e:
    # This is a fallback for direct execution or if imports fail.
    # In a real app, ensure python path is set correctly.
    print(f"Error importing backend modules: {e}. GUI will have limited functionality.")
    # Define dummy functions if imports fail, so GUI can still load
    def create_emoji_dataset(*args, **kwargs):
        print("Backend 'create_emoji_dataset' not available.")
        return None, None, [], {}
    def generate_emoji_image(*args, **kwargs):
        print("Backend 'generate_emoji_image' not available.")
        return None # Or a dummy PIL image
    def tensor_to_pil(*args, **kwargs):
        print("Backend 'tensor_to_pil' not available.")
        return None


# --- Worker Thread for Dataset Generation ---
class DatasetGenerationThread(QThread):
    # Signals: finished(images_tensor, context_indices_tensor, vocab, emoji_to_idx)
    #          error(str_message)
    #          progress(str_message)
    finished = pyqtSignal(object, object, list, dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, emoji_string_list, image_size, font_path):
        super().__init__()
        self.emoji_string_list = emoji_string_list
        self.image_size = image_size
        self.font_path = font_path

    def run(self):
        try:
            self.progress.emit(f"Generating dataset for {len(self.emoji_string_list)} emojis...")
            if not self.emoji_string_list:
                self.error.emit("Emoji list is empty.")
                return

            images_tensor, context_indices_tensor, vocab, emoji_to_idx = \
                create_emoji_dataset(self.emoji_string_list, self.image_size, self.font_path)

            if images_tensor is None or images_tensor.numel() == 0:
                 self.progress.emit(f"Dataset generation resulted in no images. Vocab size: {len(vocab)}")
            else:
                self.progress.emit(f"Dataset generated: {images_tensor.shape[0]} images. Vocab size: {len(vocab)}")

            self.finished.emit(images_tensor, context_indices_tensor, vocab, emoji_to_idx)
        except Exception as e:
            self.error.emit(f"Error during dataset generation: {str(e)}")


class DatasetTab(QWidget):
    # Signal to notify other parts of the app (e.g., training tab) about new dataset/vocab
    dataset_updated = pyqtSignal(object, int) # dataset_info (e.g. path or object), vocab_size

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_size = 16 # Default, can be configurable later
        self.generated_images_tensor = None
        self.generated_vocab = None
        self.generated_emoji_to_idx = None

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # 1. Controls Group
        controls_group = QGroupBox("Dataset Generation Controls")
        controls_layout = QVBoxLayout()

        # Emoji Input
        self.emoji_input_label = QLabel("Enter emojis (each character will be one image):")
        controls_layout.addWidget(self.emoji_input_label)
        self.emoji_input_text = QTextEdit()
        self.emoji_input_text.setPlaceholderText("e.g., 😀😂😍👍🎉🚀")
        self.emoji_input_text.setMinimumHeight(100)
        # Populate with a default list for convenience
        default_emojis = "😀😁😂🤣😃😄😅😆😉😊😋😎😍😘🥰😗😙🥲🤔🤩🤗🙂😚🤨😐😑😶🫥😮😥😣😏🙄😤😠😡🤬🤯🥵🥶😳🤪😵😵‍💫😲😱😨😰😢😭😮‍💨🥱😴🤤😪🤢🤮🤧😇🤠🥳🥺🥸🧐😕😟🙁😮😯😲😳🥺😦😧😨😰😥😢😭😱😖😣😞😓😩😫🥱😤😡😠🤬😈👿💀☠️💩🤡👹👺👻👽👾🤖😺😸😹😻😼😽🙀😿😾🙈🙉🙊👋🤚🖐️✋🖖👌🤌🤏✌️🤞🤟🤘🤙👈👉👆🖕👇☝️👍👎✊👊🤛🤜👏🙌🫶👐🤲🤝🙏✍️💅🤳💪🦾🦿🦵🦶👂🦻👃🧠🫀🫁🦷🦴👀👁️👅👄👶🧒🧑🎨🎵💻💡🌟🚀🎉"
        self.emoji_input_text.setText(default_emojis)
        controls_layout.addWidget(self.emoji_input_text)

        # Font Path Input (Optional)
        font_layout = QHBoxLayout()
        self.font_path_label = QLabel("Font Path (Optional, .ttf):")
        font_layout.addWidget(self.font_path_label)
        self.font_path_edit = QLineEdit()
        self.font_path_edit.setPlaceholderText("e.g., /path/to/NotoColorEmoji.ttf")
        font_layout.addWidget(self.font_path_edit)
        self.font_browse_button = QPushButton("Browse...")
        self.font_browse_button.clicked.connect(self._browse_font_file)
        font_layout.addWidget(self.font_browse_button)
        controls_layout.addLayout(font_layout)

        # Generate Button
        self.generate_button = QPushButton("Generate Dataset")
        self.generate_button.clicked.connect(self._start_dataset_generation)
        controls_layout.addWidget(self.generate_button)

        controls_group.setLayout(controls_layout)
        main_layout.addWidget(controls_group)

        # 2. Status and Stats Group
        status_stats_group = QGroupBox("Status & Statistics")
        status_stats_layout = QVBoxLayout()
        self.status_label = QLabel("Status: Ready")
        status_stats_layout.addWidget(self.status_label)
        self.stats_label = QLabel("Generated Images: 0 | Vocabulary Size: 0")
        status_stats_layout.addWidget(self.stats_label)
        status_stats_group.setLayout(status_stats_layout)
        main_layout.addWidget(status_stats_group)

        # 3. Preview Area Group
        preview_group = QGroupBox("Dataset Preview")
        preview_main_layout = QVBoxLayout()

        self.preview_scroll_area = QScrollArea()
        self.preview_scroll_area.setWidgetResizable(True)
        self.preview_widget = QWidget() # Container for grid layout
        self.preview_grid_layout = QGridLayout(self.preview_widget)
        self.preview_grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.preview_scroll_area.setWidget(self.preview_widget)

        preview_main_layout.addWidget(self.preview_scroll_area)
        preview_group.setLayout(preview_main_layout)
        main_layout.addWidget(preview_group)

        main_layout.addStretch(1) # Add stretch at the end
        self.setLayout(main_layout)

    def _browse_font_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        font_file, _ = QFileDialog.getOpenFileName(self, "Select Font File", "",
                                                  "Font Files (*.ttf *.otf);;All Files (*)", options=options)
        if font_file:
            self.font_path_edit.setText(font_file)

    def _start_dataset_generation(self):
        emoji_text = self.emoji_input_text.toPlainText()
        # Use set to get unique characters, then convert back to list
        # This also helps in removing duplicates if user pastes them.
        # Need to handle multi-character graphemes if they are treated as single "emojis"
        # For now, Python's string iteration gives individual Unicode characters.
        unique_emojis = sorted(list(set(c for c in emoji_text if c.strip()))) # Filter out whitespace

        if not unique_emojis:
            QMessageBox.warning(self, "Input Error", "Please enter some emojis to generate the dataset.")
            return

        font_path = self.font_path_edit.text()
        if not font_path or not os.path.exists(font_path):
            font_path = None # Pass None to use default font search in backend
            self.status_label.setText("Status: Font path not provided or invalid. Using default font search.")

        self.generate_button.setEnabled(False)
        self.status_label.setText("Status: Generating dataset...")
        self._clear_preview_area()

        self.generation_thread = DatasetGenerationThread(unique_emojis, self.image_size, font_path)
        self.generation_thread.finished.connect(self._on_dataset_generation_finished)
        self.generation_thread.error.connect(self._on_dataset_generation_error)
        self.generation_thread.progress.connect(self._update_progress)
        self.generation_thread.start()

    def _update_progress(self, message):
        self.status_label.setText(f"Status: {message}")

    def _on_dataset_generation_finished(self, images_tensor, context_indices_tensor, vocab, emoji_to_idx):
        self.generate_button.setEnabled(True)
        self.status_label.setText("Status: Dataset generation finished.")

        self.generated_images_tensor = images_tensor
        self.generated_vocab = vocab
        self.generated_emoji_to_idx = emoji_to_idx

        if images_tensor is not None and images_tensor.numel() > 0:
            self.stats_label.setText(f"Generated Images: {images_tensor.shape[0]} | Vocabulary Size: {len(vocab)}")
            self._update_preview_area(images_tensor)
            # Emit signal that dataset has been updated
            # Pass some info, e.g., the tensor itself or path if saved, and vocab size
            self.dataset_updated.emit(images_tensor, len(vocab))
        else:
            self.stats_label.setText(f"Generated Images: 0 | Vocabulary Size: {len(vocab)}")
            QMessageBox.information(self, "Dataset Generation", "No images were generated. Check emoji input and font.")
            self._clear_preview_area()


    def _on_dataset_generation_error(self, error_message):
        self.generate_button.setEnabled(True)
        self.status_label.setText(f"Status: Error!")
        QMessageBox.critical(self, "Generation Error", error_message)
        self._clear_preview_area()
        self.stats_label.setText("Generated Images: 0 | Vocabulary Size: 0")


    def _clear_preview_area(self):
        while self.preview_grid_layout.count():
            child = self.preview_grid_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        # Add a placeholder if needed
        # placeholder = QLabel("Dataset preview will appear here.")
        # placeholder.setAlignment(Qt.AlignCenter)
        # self.preview_grid_layout.addWidget(placeholder, 0, 0)


    def _update_preview_area(self, images_tensor):
        self._clear_preview_area()

        if images_tensor is None or images_tensor.numel() == 0:
            return

        max_cols = 10 # Max images per row in preview
        row, col = 0, 0

        for i in range(images_tensor.shape[0]):
            img_tensor = images_tensor[i] # (C, H, W)

            # Convert tensor to QPixmap for display
            pil_img = tensor_to_pil((img_tensor.cpu() + 1) / 2) # Denormalize from [-1,1] to [0,1] for PIL
            if pil_img:
                q_image = QImage(pil_img.tobytes("raw", "RGB"), pil_img.width, pil_img.height, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)

                img_label = QLabel()
                img_label.setPixmap(pixmap.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)) # Scaled preview
                img_label.setFixedSize(64,64) # Ensure consistent size for grid
                self.preview_grid_layout.addWidget(img_label, row, col)

                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1
            if row * max_cols + col > 100 : # Limit previews to avoid performance issues on huge datasets
                limit_label = QLabel("Preview limited to first 100 images...")
                limit_label.setAlignment(Qt.AlignCenter)
                self.preview_grid_layout.addWidget(limit_label, row + 1, 0, 1, max_cols) # Span across columns
                break


# Example for running this tab standalone (for testing UI)
if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication, QMainWindow
    app = QApplication(sys.argv)
    main_win = QMainWindow()
    dataset_tab_widget = DatasetTab()
    main_win.setCentralWidget(dataset_tab_widget)
    main_win.setWindowTitle("Dataset Tab Test")
    main_win.resize(600, 800)
    main_win.show()
    sys.exit(app.exec_())
