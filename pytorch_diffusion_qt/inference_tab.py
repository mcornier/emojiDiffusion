import sys
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QScrollArea, QGroupBox, QFrame, QMessageBox, QMainWindow
)
from PyQt5.QtGui import QPixmap, QImage # For displaying images
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QThread

import torch
from torchvision import transforms

# Backend imports
from pytorch_diffusion.utils import denoise_single_step, TIMESTEPS as UTILS_TIMESTEPS

# Updated tensor_to_qpixmap function
def tensor_to_qpixmap(image_tensor: torch.Tensor, size=(128, 128)):
    if image_tensor is None:
        dummy_image = QImage(size[0], size[1], QImage.Format_RGB888)
        dummy_image.fill(Qt.darkGray) # type: ignore
        return QPixmap.fromImage(dummy_image)

    img_tensor_cpu = image_tensor.detach().cpu()

    # Normalize if it seems to be in [-1, 1] range
    if img_tensor_cpu.min() < -0.1: # A simple heuristic
        img_tensor_cpu = (img_tensor_cpu + 1) / 2.0
    img_tensor_cpu = img_tensor_cpu.clamp(0, 1) # Ensure it's in [0, 1]

    # Handle batch dimension if present
    if img_tensor_cpu.ndim == 4 and img_tensor_cpu.shape[0] > 0:
        img_tensor_cpu = img_tensor_cpu[0] # Take the first image in batch
    elif img_tensor_cpu.ndim != 3: # Expect 3D tensor (C, H, W)
        # Invalid dimensions, return a placeholder
        dummy_image = QImage(size[0], size[1], QImage.Format_RGB888)
        dummy_image.fill(Qt.red) # type: ignore # Red indicates error
        return QPixmap.fromImage(dummy_image)

    try:
        pil_img = transforms.ToPILImage()(img_tensor_cpu)
    except Exception as e:
        print(f"Error converting tensor to PIL Image: {e}")
        dummy_image = QImage(size[0], size[1], QImage.Format_RGB888)
        dummy_image.fill(Qt.blue) # type: ignore # Blue for PIL conversion error
        return QPixmap.fromImage(dummy_image)

    try:
        # Ensure PIL image is in RGB format for QImage
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        # Convert PIL Image to QImage
        q_image = QImage(pil_img.tobytes("raw", "RGB"), pil_img.width, pil_img.height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        return pixmap.scaled(size[0], size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation) # type: ignore
    except Exception as e:
        print(f"Error converting PIL to QPixmap: {e}")
        dummy_image = QImage(size[0], size[1], QImage.Format_RGB888)
        dummy_image.fill(Qt.green) # type: ignore # Green for QImage/Pixmap conversion error
        return QPixmap.fromImage(dummy_image)


class InferenceStepWidget(QFrame):
    def __init__(self, step_data, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self)

        self.step_label = QLabel(f"<b>Step #{step_data.get('t_original', 'N/A')}</b> (t_index: {step_data.get('t_idx', 'N/A')})")
        layout.addWidget(self.step_label)

        images_layout = QHBoxLayout()

        # image_t (current noisy image)
        img_t_group = QGroupBox(f"x_t (Input to step {step_data.get('t_idx', 'N/A')})")
        img_t_layout = QVBoxLayout()
        self.img_t_view = QLabel("img_t")
        self.img_t_view.setFixedSize(128, 128)
        self.img_t_view.setAlignment(Qt.AlignCenter) # type: ignore
        self.img_t_view.setStyleSheet("border: 1px solid gray;")
        img_t_pixmap = step_data.get('image_t_pixmap')
        if img_t_pixmap:
            self.img_t_view.setPixmap(img_t_pixmap.scaled(128,128, Qt.KeepAspectRatio)) # type: ignore
        else:
            self.img_t_view.setText(f"Image @ t={step_data.get('t_idx', 'N/A')}")
        img_t_layout.addWidget(self.img_t_view)
        img_t_group.setLayout(img_t_layout)
        images_layout.addWidget(img_t_group)


        # epsilon_pred (predicted noise)
        epsilon_group = QGroupBox("ε_pred (Predicted Noise)")
        epsilon_layout = QVBoxLayout()
        self.epsilon_view = QLabel("epsilon_pred")
        self.epsilon_view.setFixedSize(128, 128)
        self.epsilon_view.setAlignment(Qt.AlignCenter) # type: ignore
        self.epsilon_view.setStyleSheet("border: 1px solid gray;")
        epsilon_pixmap = step_data.get('epsilon_pred_pixmap')
        if epsilon_pixmap:
            self.epsilon_view.setPixmap(epsilon_pixmap.scaled(128,128, Qt.KeepAspectRatio)) # type: ignore
        else:
            self.epsilon_view.setText("Predicted Noise ε")
        epsilon_layout.addWidget(self.epsilon_view)
        epsilon_group.setLayout(epsilon_layout)
        images_layout.addWidget(epsilon_group)

        # image_t-1 (denoised image)
        img_t_prev_group = QGroupBox(f"x_(t-1) (Output of step {step_data.get('t_idx', 'N/A')})")
        img_t_prev_layout = QVBoxLayout()
        self.img_t_prev_view = QLabel("img_t_prev")
        self.img_t_prev_view.setFixedSize(128, 128)
        self.img_t_prev_view.setAlignment(Qt.AlignCenter) # type: ignore
        self.img_t_prev_view.setStyleSheet("border: 1px solid gray;")
        img_t_prev_pixmap = step_data.get('image_t_minus_1_pixmap')
        if img_t_prev_pixmap:
            self.img_t_prev_view.setPixmap(img_t_prev_pixmap.scaled(128,128, Qt.KeepAspectRatio)) # type: ignore
        else:
            self.img_t_prev_view.setText(f"Result x_{step_data.get('t_idx', 'N/A')-1}")
        img_t_prev_layout.addWidget(self.img_t_prev_view)
        img_t_prev_group.setLayout(img_t_prev_layout)
        images_layout.addWidget(img_t_prev_group)

        layout.addLayout(images_layout)
        self.setLayout(layout)


class DenoiseStepThread(QThread):
    # (Content of DenoiseStepThread as provided in the previous attempt)
    step_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    finished_all_steps = pyqtSignal()

    def __init__(self, model_ref, x_t_tensor, t_start_idx, num_steps, context_idx_tensor, parent=None): # MODIFIED
        super().__init__(parent)
        self.model = model_ref
        self.current_x_t = x_t_tensor
        self.t_start_idx = t_start_idx
        self.num_steps_to_run = num_steps
        self.context_idx_tensor = context_idx_tensor # MODIFIED
        self._is_running = True

    def run(self):
        if self.model is None:
            self.error_occurred.emit("Model not available for inference.")
            return
        try:
            # torch should already be imported at the top level of the file
            current_t_idx = self.t_start_idx
            for _i in range(self.num_steps_to_run):
                if not self._is_running:
                    self.error_occurred.emit("Denoising stopped by user/system.")
                    return

                if current_t_idx < 0:
                    self.error_occurred.emit("Time index fell below zero.")
                    break

                if self.current_x_t is None: # Should not happen if initialized correctly
                     self.error_occurred.emit("Current x_t tensor is None.")
                     return

                model_device = self.current_x_t.device

                current_x_t_on_device = self.current_x_t.to(model_device)
                context_idx_tensor_on_device = self.context_idx_tensor.to(model_device) if self.context_idx_tensor is not None else None

                # Create t_tensor for the current step
                # Ensure t_tensor has the correct shape (batch_size,) if model expects that,
                # or just a scalar if model handles broadcasting.
                # denoise_single_step expects t_tensor to be a 1D tensor for model input.
                t_tensor = torch.full((current_x_t_on_device.shape[0],), current_t_idx, device=model_device, dtype=torch.long)

                predicted_noise_tensor, x_prev_tensor = denoise_single_step(
                    self.model,
                    current_x_t_on_device,
                    t_tensor, # Pass the tensor 't'
                    current_t_idx, # Pass the integer index 't_index'
                    context_idx_tensor_on_device
                )

                step_data = {
                    't_original': current_t_idx + 1, # User-facing step number (e.g., T down to 1)
                    't_idx': current_t_idx,          # Actual time index (e.g., T-1 down to 0)
                    'image_t_tensor': current_x_t_on_device.clone(),
                    'predicted_noise_tensor': predicted_noise_tensor.clone(),
                    'image_t_minus_1_tensor': x_prev_tensor.clone()
                }
                self.step_completed.emit(step_data)

                self.current_x_t = x_prev_tensor # For next iteration if num_steps_to_run > 1
                current_t_idx -= 1

                if current_t_idx < 0:
                    self.finished_all_steps.emit()
                    break
        except Exception as e:
            import traceback
            self.error_occurred.emit(f"Error in denoising thread: {str(e)}\n{traceback.format_exc()}")

    def stop(self):
        self._is_running = False


class InferenceTab(QWidget):
    # (Content of InferenceTab as provided in the previous attempt, with necessary imports and definitions)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_model = None
        self.current_vocab = None
        self.current_emoji_to_idx = None

        self.TIMESTEPS = UTILS_TIMESTEPS # Use TIMESTEPS from utils

        self.x_T_tensor = None
        self.current_x_t_tensor = None
        self.current_t_idx = -1
        self.current_context_idx_tensor = None # Renamed from current_context_embedding

        self._setup_ui()
        self.denoise_thread = None

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        controls_group = QGroupBox("Inference Controls")
        controls_layout = QVBoxLayout(controls_group)
        context_layout = QHBoxLayout()
        context_layout.addWidget(QLabel("Emoji Context:"))
        self.emoji_context_input = QLineEdit()
        self.emoji_context_input.setPlaceholderText("e.g., 😀 (single emoji for now)")
        context_layout.addWidget(self.emoji_context_input)
        controls_layout.addLayout(context_layout)
        buttons_layout = QHBoxLayout()
        self.start_reset_button = QPushButton("Start / Reset Inference")
        self.start_reset_button.clicked.connect(self._start_reset_inference)
        buttons_layout.addWidget(self.start_reset_button)
        self.denoise_step_button = QPushButton("Denoise (1 Step)")
        self.denoise_step_button.clicked.connect(self._denoise_one_step)
        self.denoise_step_button.setEnabled(False)
        buttons_layout.addWidget(self.denoise_step_button)
        controls_layout.addLayout(buttons_layout)
        main_layout.addWidget(controls_group)
        self.status_label = QLabel("Status: Ready. Load a model and enter an emoji.")
        self.status_label.setAlignment(Qt.AlignCenter) # type: ignore
        main_layout.addWidget(self.status_label)
        display_group = QGroupBox("Inference Progress (Step-by-Step)")
        display_main_layout = QVBoxLayout(display_group)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(400)
        self.scroll_content_widget = QWidget()
        self.steps_layout = QVBoxLayout(self.scroll_content_widget)
        self.steps_layout.setAlignment(Qt.AlignTop) # type: ignore
        self.scroll_area.setWidget(self.scroll_content_widget)
        display_main_layout.addWidget(self.scroll_area)
        main_layout.addWidget(display_group)
        self.setLayout(main_layout)

    def set_model_and_vocab(self, model, vocab, emoji_to_idx, timesteps_from_utils):
        self.current_model = model
        self.current_vocab = vocab
        self.current_emoji_to_idx = emoji_to_idx
        self.TIMESTEPS = timesteps_from_utils
        if self.current_model:
            self.update_status("Model loaded. Ready for inference.")
        else:
            self.update_status("Model removed or not available.")

    def _start_reset_inference(self):
        if self.current_model is None:
            QMessageBox.warning(self, "No Model", "Please ensure a model is loaded via the Weights tab.")
            return
        emoji_char = self.emoji_context_input.text().strip()
        if not emoji_char:
            QMessageBox.warning(self, "Input Error", "Please enter a single emoji character for context.")
            return
        if self.current_emoji_to_idx is None or emoji_char not in self.current_emoji_to_idx:
            QMessageBox.warning(self, "Vocabulary Error", f"Emoji '{emoji_char}' not in the model's vocabulary. Please use an emoji from the training set.")
            return
        if self.denoise_thread and self.denoise_thread.isRunning():
            self.denoise_thread.stop()
            self.denoise_thread.wait()
        self.reset_inference_display()
        self.update_status(f"Initializing inference for emoji: {emoji_char}")
        try:
            # torch should be imported at the top of the file
            context_idx = self.current_emoji_to_idx[emoji_char]
            model_device = self.current_model.device if hasattr(self.current_model, 'device') else \
                           (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')) # type: ignore

            self.current_context_idx_tensor = torch.tensor([context_idx], device=model_device, dtype=torch.long) # type: ignore

            img_channels, img_size = 3, 16 # Defaults
            if hasattr(self.current_model, 'img_channels') and self.current_model.img_channels is not None: # type: ignore
                img_channels = self.current_model.img_channels # type: ignore
            if hasattr(self.current_model, 'img_size') and self.current_model.img_size is not None: # type: ignore
                img_size = self.current_model.img_size # type: ignore

            self.x_T_tensor = torch.randn(1, img_channels, img_size, img_size, device=model_device) # type: ignore
            self.current_x_t_tensor = self.x_T_tensor.clone()
            self.current_t_idx = self.TIMESTEPS - 1
            initial_step_data = {
                't_original': self.TIMESTEPS,
                't_idx': self.current_t_idx,
                'image_t_pixmap': tensor_to_qpixmap(self.current_x_t_tensor, (128,128)),
                'epsilon_pred_pixmap': None,
                'image_t_minus_1_pixmap': None
            }
            self.display_inference_step(initial_step_data)
            self.denoise_step_button.setEnabled(True)
            self.update_status(f"Initialized with noise x_T (t={self.TIMESTEPS}). Ready to denoise.")
        except Exception as e:
            self.update_status(f"Error during init: {str(e)}")
            QMessageBox.critical(self, "Inference Error", f"Failed to initialize inference: {str(e)}")
            self.denoise_step_button.setEnabled(False)

    def _denoise_one_step(self):
        if self.current_model is None or self.current_x_t_tensor is None or self.current_t_idx < 0:
            self.update_status("Error: Not ready for denoising. Start/Reset first.")
            self.denoise_step_button.setEnabled(False)
            return
        if self.denoise_thread and self.denoise_thread.isRunning():
            QMessageBox.information(self, "Busy", "A denoising step is already in progress.")
            return
        self.denoise_step_button.setEnabled(False)
        self.update_status(f"Denoising step for t_index = {self.current_t_idx}...")
        self.denoise_thread = DenoiseStepThread(
            model_ref=self.current_model,
            x_t_tensor=self.current_x_t_tensor.clone(),
            t_start_idx=self.current_t_idx,
            num_steps=1,
            context_idx_tensor=self.current_context_idx_tensor # Pass the idx tensor
        )
        self.denoise_thread.step_completed.connect(self._on_denoise_step_completed)
        self.denoise_thread.error_occurred.connect(self._on_denoise_error)
        self.denoise_thread.finished_all_steps.connect(self._on_finished_all_denoising_steps)
        self.denoise_thread.finished.connect(self._on_denoise_thread_finished)
        self.denoise_thread.start()

    @pyqtSlot(dict)
    def _on_denoise_step_completed(self, step_data_backend):
        img_t_pix = tensor_to_qpixmap(step_data_backend.get('image_t_tensor'), (128,128))
        eps_pred_pix = tensor_to_qpixmap(step_data_backend.get('predicted_noise_tensor'), (128,128))
        img_t_prev_pix = tensor_to_qpixmap(step_data_backend.get('image_t_minus_1_tensor'), (128,128))
        ui_step_data = {
            't_original': step_data_backend.get('t_original'),
            't_idx': step_data_backend.get('t_idx'),
            'image_t_pixmap': img_t_pix,
            'epsilon_pred_pixmap': eps_pred_pix,
            'image_t_minus_1_pixmap': img_t_prev_pix
        }
        self.display_inference_step(ui_step_data)
        self.current_x_t_tensor = step_data_backend.get('image_t_minus_1_tensor')
        self.current_t_idx = step_data_backend.get('t_idx') - 1
        if self.current_t_idx < 0:
            self.update_status("All denoising steps completed.")
            self.denoise_step_button.setEnabled(False)
        else:
            self.update_status(f"Step for t_index={step_data_backend.get('t_idx')} complete. Ready for t_index={self.current_t_idx}.")

    @pyqtSlot()
    def _on_finished_all_denoising_steps(self):
        self.update_status("All denoising steps successfully completed by the thread.")
        self.denoise_step_button.setEnabled(False)

    @pyqtSlot(str)
    def _on_denoise_error(self, message):
        self.update_status(f"Error during denoising: {message}")
        QMessageBox.critical(self, "Denoising Error", message)
        if self.current_t_idx >= 0 :
             self.denoise_step_button.setEnabled(True)

    @pyqtSlot()
    def _on_denoise_thread_finished(self):
        if self.current_t_idx >= 0 and self.denoise_step_button.isEnabled() == False:
            self.denoise_step_button.setEnabled(True)
        self.denoise_thread = None

    @pyqtSlot(str)
    def update_status(self, message):
        self.status_label.setText(f"Status: {message}")

    @pyqtSlot(dict)
    def display_inference_step(self, step_data):
        step_widget = InferenceStepWidget(step_data)
        self.steps_layout.addWidget(step_widget)
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

    @pyqtSlot()
    def reset_inference_display(self):
        while self.steps_layout.count():
            child = self.steps_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.update_status("Display reset. Ready for new inference.")
        self.denoise_step_button.setEnabled(False)

# Example for running this tab standalone
if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication # QApplication already imported
    app = QApplication(sys.argv)
    main_win = QMainWindow()
    inference_tab_widget = InferenceTab()
    class DummyModel:
        def __init__(self):
            self.device = 'cpu'
            self.img_channels = 3
            self.img_size = 16
    dummy_vocab = ['😀', '😂', '😍']
    dummy_emoji_to_idx = {emoji: i for i, emoji in enumerate(dummy_vocab)}
    dummy_timesteps = 150
    inference_tab_widget.set_model_and_vocab(DummyModel(), dummy_vocab, dummy_emoji_to_idx, dummy_timesteps)
    inference_tab_widget.emoji_context_input.setText("😀")
    main_win.setCentralWidget(inference_tab_widget)
    main_win.setWindowTitle("Inference Tab Test")
    main_win.resize(600, 800)
    main_win.show()
    sys.exit(app.exec_())
