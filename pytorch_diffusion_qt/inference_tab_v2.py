import sys
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox,
    QGroupBox, QFileDialog, QMessageBox, QLineEdit, QGridLayout, QTextEdit,
    QProgressBar, QScrollArea, QCheckBox, QSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image
import torch
import numpy as np

# Ensure backend modules can be imported
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

try:
    from pytorch_diffusion.utils_v2 import load_model_v2, generate_from_visual_prompt, tensor_to_pil_v2
    from pytorch_diffusion.utils_v2_manual import ManualDiffusionSampler, create_noise_visualization
    from pytorch_diffusion.model_v2 import DiffusionModelV2
    from pytorch_diffusion.dataset_v2 import ImageDatasetV2
    from pytorch_diffusion.utils import get_device
    try:
        from .paint_8x8_widget import Paint8x8Widget
    except ImportError:
        from paint_8x8_widget import Paint8x8Widget
    import random
except ImportError as e:
    print(f"Error importing V2 modules: {e}")
    load_model_v2 = None

class InferenceThread(QThread):
    """Thread for running inference without blocking the UI"""
    inference_finished = pyqtSignal(list, object)  # generated_images, conditioning_image
    inference_error = pyqtSignal(str)
    progress_updated = pyqtSignal(str)

    def __init__(self, model, prompt_image, num_samples, inference_steps=200):
        super().__init__()
        self.model = model
        self.prompt_image = prompt_image
        self.num_samples = num_samples
        self.inference_steps = inference_steps

    def run(self):
        try:
            self.progress_updated.emit(f"Generating {self.num_samples} images with {self.inference_steps} steps...")
            device = get_device()
            
            # Import needed for custom steps
            from pytorch_diffusion.utils_v2 import p_sample_loop_v2, pil_to_tensor_v2
            
            # Prepare visual context (8x8)
            context_tensor = pil_to_tensor_v2(self.prompt_image, target_size=(8, 8))  # (1, 3, 8, 8)
            context_batch = context_tensor.repeat(self.num_samples, 1, 1, 1).to(device)  # (num_samples, 3, 8, 8)
            
            # Generate images with custom number of steps
            with torch.no_grad():
                generated_tensor, history = p_sample_loop_v2(
                    self.model, 
                    context_batch, 
                    num_timesteps=self.inference_steps,
                    device=device
                )
            
            # Convert to PIL images
            from pytorch_diffusion.utils_v2 import tensor_to_pil_v2
            generated_images = []
            for i in range(self.num_samples):
                img = tensor_to_pil_v2(generated_tensor[i])
                generated_images.append(img)
            
            # Also return the conditioning image
            conditioning_image = tensor_to_pil_v2(context_tensor[0])
            
            self.inference_finished.emit(generated_images, conditioning_image)
            
        except Exception as e:
            self.inference_error.emit(str(e))

class InferenceTabV2(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.prompt_image = None
        self.inference_thread = None
        self.manual_sampler = None
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # 1. Model Loading Group
        model_group = QGroupBox("Model Loading")
        model_layout = QGridLayout(model_group)

        model_layout.addWidget(QLabel("Model Checkpoint:"), 0, 0)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Path to .pth checkpoint file")
        model_layout.addWidget(self.model_path_edit, 0, 1)
        self.browse_model_button = QPushButton("Browse...")
        self.browse_model_button.clicked.connect(self._browse_model)
        model_layout.addWidget(self.browse_model_button, 0, 2)

        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self._load_model)
        model_layout.addWidget(self.load_model_button, 1, 0, 1, 3)

        self.model_status_label = QLabel("Status: No model loaded")
        model_layout.addWidget(self.model_status_label, 2, 0, 1, 3)

        main_layout.addWidget(model_group)

        # 2. Visual Prompt Group - 8x8 Paint Widget
        prompt_group = QGroupBox("8x8 Visual Prompt")
        prompt_layout = QVBoxLayout(prompt_group)

        # Prompt source selection
        source_layout = QHBoxLayout()
        self.random_from_dataset_button = QPushButton("Load from Dataset")
        self.random_from_dataset_button.clicked.connect(self._load_random_from_dataset)
        source_layout.addWidget(self.random_from_dataset_button)
        
        self.upload_prompt_button = QPushButton("Load from File")
        self.upload_prompt_button.clicked.connect(self._upload_prompt_image)
        source_layout.addWidget(self.upload_prompt_button)
        source_layout.addStretch()
        prompt_layout.addLayout(source_layout)

        # 8x8 Paint Widget
        self.paint_widget = Paint8x8Widget()
        self.paint_widget.gridChanged.connect(self._on_grid_changed)
        prompt_layout.addWidget(self.paint_widget)

        # Initialize the grid change to set initial state
        self._on_grid_changed()

        main_layout.addWidget(prompt_group)

        # 3. Generation Mode Group
        mode_group = QGroupBox("Generation Mode")
        mode_layout = QGridLayout(mode_group)

        # Mode selection
        self.auto_mode_checkbox = QCheckBox("Auto Generation")
        self.auto_mode_checkbox.setChecked(True)
        self.auto_mode_checkbox.toggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self.auto_mode_checkbox, 0, 0)

        self.manual_mode_checkbox = QCheckBox("Manual Step-by-Step")
        self.manual_mode_checkbox.toggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self.manual_mode_checkbox, 0, 1)

        # Auto mode parameters
        mode_layout.addWidget(QLabel("Number of Samples:"), 1, 0)
        self.num_samples_spinbox = QSpinBox()
        self.num_samples_spinbox.setRange(1, 8)
        self.num_samples_spinbox.setValue(1)
        self.num_samples_spinbox.setToolTip("Nombre d'images à générer")
        mode_layout.addWidget(self.num_samples_spinbox, 1, 1)

        # Inference parameters
        mode_layout.addWidget(QLabel("Inference Steps:"), 2, 0)
        self.inference_steps_spinbox = QSpinBox()
        self.inference_steps_spinbox.setRange(10, 1000)
        self.inference_steps_spinbox.setValue(200)
        self.inference_steps_spinbox.setToolTip("Nombre d'étapes de débruitage (plus = meilleure qualité mais plus lent)")
        mode_layout.addWidget(self.inference_steps_spinbox, 2, 1)

        main_layout.addWidget(mode_group)

        # 4. Generation Controls
        controls_group = QGroupBox("Generation Controls")
        controls_layout = QVBoxLayout(controls_group)

        # Auto generation button
        self.generate_button = QPushButton("🚀 Generate Images")
        self.generate_button.clicked.connect(self._generate_images)
        self.generate_button.setEnabled(False)
        controls_layout.addWidget(self.generate_button)

        # Manual step controls
        manual_controls_layout = QHBoxLayout()
        self.init_button = QPushButton("🎲 Initialize")
        self.init_button.clicked.connect(self._initialize_manual)
        self.init_button.setEnabled(False)
        self.init_button.setVisible(False)
        manual_controls_layout.addWidget(self.init_button)

        self.step_button = QPushButton("➡️ Next Step")
        self.step_button.clicked.connect(self._manual_step)
        self.step_button.setEnabled(False)
        self.step_button.setVisible(False)
        manual_controls_layout.addWidget(self.step_button)

        self.reset_button = QPushButton("🔄 Reset")
        self.reset_button.clicked.connect(self._reset_manual)
        self.reset_button.setEnabled(False)
        self.reset_button.setVisible(False)
        manual_controls_layout.addWidget(self.reset_button)

        manual_controls_layout.addStretch()
        controls_layout.addLayout(manual_controls_layout)

        # Step info
        self.step_info_label = QLabel("Ready to initialize")
        self.step_info_label.setVisible(False)
        controls_layout.addWidget(self.step_info_label)

        # Progress slider for manual mode
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 200)
        self.progress_slider.setValue(0)
        self.progress_slider.setEnabled(False)
        self.progress_slider.setVisible(False)
        controls_layout.addWidget(self.progress_slider)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Load a model and create a visual prompt to start")
        controls_layout.addWidget(self.status_label)

        main_layout.addWidget(controls_group)

        # 5. Results Group
        results_group = QGroupBox("Generated Images")
        results_layout = QVBoxLayout(results_group)

        # Scroll area for results
        self.results_scroll = QScrollArea()
        self.results_widget = QWidget()
        self.results_layout = QGridLayout(self.results_widget)
        self.results_scroll.setWidget(self.results_widget)
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setMinimumHeight(300)
        results_layout.addWidget(self.results_scroll)

        main_layout.addWidget(results_group)

        self.setLayout(main_layout)

        # Initialize state
        self.manual_sampler = None
        self._update_ui_state()

    def _browse_model(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        
        # Default to V2 checkpoints directory
        default_dir = "pytorch_diffusion_qt/checkpoints/"
        if not os.path.exists(default_dir):
            default_dir = "checkpoints/"
        
        model_file, _ = QFileDialog.getOpenFileName(
            self, "Select V2 Model Checkpoint", default_dir,
            "PyTorch V2 Checkpoint Files (*.pth *.pt);;All Files (*)", options=options
        )
        if model_file:
            self.model_path_edit.setText(model_file)

    def _load_model(self):
        model_path = self.model_path_edit.text()
        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "Error", "Please select a valid model checkpoint file.")
            return

        try:
            self.model_status_label.setText("Loading model...")
            self.load_model_button.setEnabled(False)
            
            # Load model
            device = get_device()
            self.model = load_model_v2(model_path, device)
            
            self.model_status_label.setText(f"✅ Model loaded successfully from {os.path.basename(model_path)}")
            self.load_model_button.setEnabled(True)
            
            # Update UI state
            self._update_generate_button_state()
            
        except Exception as e:
            QMessageBox.critical(self, "Model Loading Error", f"Failed to load model:\n{str(e)}")
            self.model_status_label.setText("❌ Failed to load model")
            self.load_model_button.setEnabled(True)
            self.model = None

    def _upload_prompt_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        image_file, _ = QFileDialog.getOpenFileName(
            self, "Select Prompt Image", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)", options=options
        )
        if image_file:
            try:
                # Load and display image
                self.prompt_image = Image.open(image_file).convert('RGB')
                
                # Display original image
                pil_image = self.prompt_image.copy()
                if pil_image.size[0] > 200 or pil_image.size[1] > 200:
                    pil_image.thumbnail((200, 200), Image.LANCZOS)
                
                # Convert to QPixmap
                qimage = QImage(pil_image.tobytes(), pil_image.width, pil_image.height, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                self.prompt_display_label.setPixmap(pixmap)
                
                # Update UI state
                self._update_generate_button_state()
                
            except Exception as e:
                QMessageBox.critical(self, "Image Loading Error", f"Failed to load image:\n{str(e)}")

    def _update_generate_button_state(self):
        """Enable generate button only if model and prompt image are loaded"""
        can_generate = self.model is not None and self.prompt_image is not None
        self.generate_button.setEnabled(can_generate)
        
        if can_generate:
            self.status_label.setText("Ready to generate! Click 'Generate Images' to start.")
        elif self.model is None:
            self.status_label.setText("Load a model to continue")
        elif self.prompt_image is None:
            self.status_label.setText("Upload a prompt image to continue")

    def _generate_images(self):
        if not self.model or not self.prompt_image:
            return

        if self.inference_thread and self.inference_thread.isRunning():
            QMessageBox.warning(self, "Generation In Progress", "A generation is already running.")
            return

        # Start inference
        num_samples = self.num_samples_spinbox.value()
        
        self.generate_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        self.inference_thread = InferenceThread(self.model, self.prompt_image, num_samples)
        self.inference_thread.inference_finished.connect(self._on_inference_finished)
        self.inference_thread.inference_error.connect(self._on_inference_error)
        self.inference_thread.progress_updated.connect(self._on_progress_updated)
        self.inference_thread.start()

    def _on_progress_updated(self, message):
        self.status_label.setText(message)

    def _on_inference_finished(self, generated_images, conditioning_image):
        self.generate_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"✅ Generated {len(generated_images)} images successfully!")
        
        # Clear previous results
        self._clear_results()
        
        # Display conditioning image
        row = 0
        conditioning_label = QLabel("8x8 Conditioning:")
        self.results_layout.addWidget(conditioning_label, row, 0, 1, 4)
        
        row += 1
        cond_display = QLabel()
        cond_display.setMinimumSize(64, 64)
        cond_display.setMaximumSize(64, 64)
        cond_display.setStyleSheet("border: 1px solid blue;")
        cond_display.setScaledContents(True)
        
        # Convert conditioning PIL to QPixmap
        cond_rgb = conditioning_image.convert('RGB')
        cond_qimage = QImage(cond_rgb.tobytes(), cond_rgb.width, cond_rgb.height, QImage.Format_RGB888)
        cond_pixmap = QPixmap.fromImage(cond_qimage)
        cond_display.setPixmap(cond_pixmap)
        self.results_layout.addWidget(cond_display, row, 0)
        
        # Display generated images
        row += 1
        gen_label = QLabel("Generated 64x64 Images:")
        self.results_layout.addWidget(gen_label, row, 0, 1, 4)
        
        row += 1
        col = 0
        for i, img in enumerate(generated_images):
            img_display = QLabel()
            img_display.setMinimumSize(128, 128)
            img_display.setMaximumSize(128, 128)
            img_display.setStyleSheet("border: 1px solid green;")
            img_display.setScaledContents(True)
            
            # Convert PIL to QPixmap
            img_rgb = img.convert('RGB')
            img_qimage = QImage(img_rgb.tobytes(), img_rgb.width, img_rgb.height, QImage.Format_RGB888)
            img_pixmap = QPixmap.fromImage(img_qimage)
            img_display.setPixmap(img_pixmap)
            
            self.results_layout.addWidget(img_display, row, col)
            
            col += 1
            if col >= 4:  # 4 images per row
                row += 1
                col = 0

    def _on_inference_error(self, error_message):
        self.generate_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("❌ Generation failed")
        QMessageBox.critical(self, "Generation Error", f"Failed to generate images:\n{error_message}")

    def _load_random_from_dataset(self):
        """Load a random image from the dataset"""
        dataset_path = "H:/Repositories/ImgDatasetGen_256"
        
        if not os.path.exists(dataset_path):
            QMessageBox.warning(self, "Dataset Not Found", 
                               f"Dataset not found at: {dataset_path}")
            return
        
        try:
            # Create dataset loader
            dataset = ImageDatasetV2(dataset_path)
            
            if len(dataset) == 0:
                QMessageBox.warning(self, "Empty Dataset", "No images found in dataset")
                return
            
            # Get random image
            random_idx = random.randint(0, len(dataset) - 1)
            main_tensor, context_tensor, img_path = dataset[random_idx]
            
            # Convert tensor back to PIL for display and processing
            # main_tensor is (3, 64, 64) in [-1, 1]
            main_np = ((main_tensor + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy()
            main_np = (main_np * 255).astype(np.uint8)
            self.prompt_image = Image.fromarray(main_np)
            
            # Display image
            display_image = self.prompt_image.copy()
            if display_image.size[0] > 200 or display_image.size[1] > 200:
                display_image.thumbnail((200, 200), Image.LANCZOS)
            
            # Convert to QPixmap
            qimage = QImage(display_image.tobytes(), display_image.width, display_image.height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            self.prompt_display_label.setPixmap(pixmap)
            
            # Update UI state
            self._update_generate_button_state()
            
            # Show info about selected image
            self.status_label.setText(f"Loaded random image: {os.path.basename(img_path)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Dataset Loading Error", f"Failed to load random image:\n{str(e)}")

    def _on_grid_changed(self):
        """Handle changes to the 8x8 paint grid"""
        # Get the current tensor from paint widget
        self.visual_context_tensor = self.paint_widget.get_tensor().unsqueeze(0)  # Add batch dim
        self.prompt_image = self.paint_widget.get_pil_image()  # For compatibility
        self._update_ui_state()

    def _on_mode_changed(self):
        """Handle mode selection changes"""
        # Ensure only one mode is selected
        if hasattr(self, 'auto_mode_checkbox') and hasattr(self, 'manual_mode_checkbox'):
            if self.sender() == self.auto_mode_checkbox:
                if self.auto_mode_checkbox.isChecked():
                    self.manual_mode_checkbox.setChecked(False)
            elif self.sender() == self.manual_mode_checkbox:
                if self.manual_mode_checkbox.isChecked():
                    self.auto_mode_checkbox.setChecked(False)
            
            # If both unchecked, default to auto
            if not self.auto_mode_checkbox.isChecked() and not self.manual_mode_checkbox.isChecked():
                self.auto_mode_checkbox.setChecked(True)
        
        self._update_ui_state()

    def _update_ui_state(self):
        """Update UI state based on current conditions"""
        has_model = self.model is not None
        has_prompt = hasattr(self, 'visual_context_tensor')
        
        # Safe check for mode checkboxes
        is_auto_mode = getattr(self, 'auto_mode_checkbox', None) and self.auto_mode_checkbox.isChecked()
        is_manual_mode = getattr(self, 'manual_mode_checkbox', None) and self.manual_mode_checkbox.isChecked()
        
        # Default to auto mode if widgets don't exist yet
        if not hasattr(self, 'auto_mode_checkbox'):
            is_auto_mode = True
            is_manual_mode = False
        
        # Auto mode controls (safe checks)
        if hasattr(self, 'generate_button'):
            self.generate_button.setVisible(is_auto_mode)
            self.generate_button.setEnabled(has_model and has_prompt and is_auto_mode)
        if hasattr(self, 'num_samples_spinbox'):
            self.num_samples_spinbox.setEnabled(is_auto_mode)
        
        # Manual mode controls (safe checks)
        if hasattr(self, 'init_button'):
            self.init_button.setVisible(is_manual_mode)
            self.init_button.setEnabled(has_model and has_prompt and is_manual_mode)
        if hasattr(self, 'step_button'):
            self.step_button.setVisible(is_manual_mode)
        if hasattr(self, 'reset_button'):
            self.reset_button.setVisible(is_manual_mode)
        if hasattr(self, 'step_info_label'):
            self.step_info_label.setVisible(is_manual_mode)
        if hasattr(self, 'progress_slider'):
            self.progress_slider.setVisible(is_manual_mode)
        
        # Manual sampler state (safe checks)
        if is_manual_mode and self.manual_sampler and hasattr(self, 'step_button') and hasattr(self, 'reset_button'):
            self.step_button.setEnabled(not self.manual_sampler.is_finished())
            self.reset_button.setEnabled(True)
        elif hasattr(self, 'step_button') and hasattr(self, 'reset_button'):
            self.step_button.setEnabled(False)
            self.reset_button.setEnabled(False)
        
        # Status message (safe check)
        if hasattr(self, 'status_label'):
            if not has_model:
                self.status_label.setText("Load a model to continue")
            elif not has_prompt:
                self.status_label.setText("Create or load a visual prompt")
            elif is_auto_mode:
                self.status_label.setText("Ready for auto generation!")
            elif is_manual_mode:
                if not self.manual_sampler:
                    self.status_label.setText("Click Initialize to start manual sampling")
                elif self.manual_sampler.is_finished():
                    self.status_label.setText("Manual sampling completed!")
                else:
                    state = self.manual_sampler.get_current_state()
                    self.status_label.setText(f"Manual mode: Step {200-state['timestep']}/200 ({state['progress_percent']:.1f}%)")

    def _initialize_manual(self):
        """Initialize manual sampling"""
        if not self.model or not hasattr(self, 'visual_context_tensor'):
            return
        
        try:
            device = get_device()
            visual_context = self.visual_context_tensor.to(device)
            
            # Create manual sampler
            self.manual_sampler = ManualDiffusionSampler(
                model=self.model,
                visual_context=visual_context,
                timesteps=200,
                device=device
            )
            
            # Update progress slider
            self.progress_slider.setValue(0)
            self.progress_slider.setEnabled(True)
            
            # Display initial noise
            self._display_manual_state()
            
            self.step_info_label.setText("Initialized with random noise. Click 'Next Step' to denoise.")
            self._update_ui_state()
            
        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", f"Failed to initialize manual sampling:\n{str(e)}")

    def _manual_step(self):
        """Perform one manual denoising step"""
        if not self.manual_sampler or self.manual_sampler.is_finished():
            return
        
        try:
            # Perform one step
            success = self.manual_sampler.step()
            
            if success:
                # Update progress slider
                state = self.manual_sampler.get_current_state()
                progress_value = int(state['progress_percent'] * 2)  # 0-200 range
                self.progress_slider.setValue(progress_value)
                
                # Display current state
                self._display_manual_state()
                
                # Update step info
                if state['is_finished']:
                    self.step_info_label.setText("✅ Manual sampling completed!")
                else:
                    self.step_info_label.setText(f"Step {200-state['timestep']}/200 - Timestep t={state['timestep']}")
            
            self._update_ui_state()
            
        except Exception as e:
            QMessageBox.critical(self, "Step Error", f"Failed to perform manual step:\n{str(e)}")

    def _reset_manual(self):
        """Reset manual sampling to initial state"""
        if self.manual_sampler:
            try:
                self.manual_sampler.reset()
                self.progress_slider.setValue(0)
                self._display_manual_state()
                self.step_info_label.setText("Reset to initial noise. Click 'Next Step' to start denoising.")
                self._update_ui_state()
            except Exception as e:
                QMessageBox.critical(self, "Reset Error", f"Failed to reset manual sampling:\n{str(e)}")

    def _display_manual_state(self):
        """Display current state of manual sampling with step-by-step progression"""
        if not self.manual_sampler:
            return
        
        try:
            # Get current state
            state = self.manual_sampler.get_current_state()
            
            # Create step-by-step display showing progression
            row = 0
            
            # Header showing current step
            step_num = 200 - state['timestep']
            header_label = QLabel(f"Step {step_num}/200 - Timestep t={state['timestep']}")
            header_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #333;")
            self.results_layout.addWidget(header_label, row, 0, 1, 5)
            
            row += 1
            
            # Step equation: img_t -> ε_θ -> img_t-1
            equation_label = QLabel("img_t  →  ε_θ(img_t, t, context)  →  img_t-1")
            equation_label.setStyleSheet("font-family: monospace; font-size: 12px; color: #666;")
            self.results_layout.addWidget(equation_label, row, 0, 1, 5)
            
            row += 1
            
            # Column headers
            headers = ["img_t (Input)", "ε_θ (Predicted Noise)", "img_t-1 (Output)", "Context"]
            for col, header in enumerate(headers):
                header_label = QLabel(header)
                header_label.setStyleSheet("font-weight: bold; text-align: center;")
                header_label.setAlignment(Qt.AlignCenter)
                self.results_layout.addWidget(header_label, row, col)
            
            row += 1
            
            # Images in a row
            # img_t (current noisy image)
            current_display = QLabel()
            current_display.setMinimumSize(128, 128)
            current_display.setMaximumSize(128, 128)
            current_display.setStyleSheet("border: 2px solid orange; background: white;")
            current_display.setScaledContents(True)
            current_pil = tensor_to_pil_v2(state['image'])
            current_qimage = QImage(current_pil.tobytes(), current_pil.width, current_pil.height, QImage.Format_RGB888)
            current_pixmap = QPixmap.fromImage(current_qimage)
            current_display.setPixmap(current_pixmap)
            self.results_layout.addWidget(current_display, row, 0)
            
            # ε_θ (predicted noise)
            if state['predicted_noise'] is not None:
                noise_display = QLabel()
                noise_display.setMinimumSize(128, 128)
                noise_display.setMaximumSize(128, 128)
                noise_display.setStyleSheet("border: 2px solid red; background: white;")
                noise_display.setScaledContents(True)
                noise_pil = create_noise_visualization(state['predicted_noise'])
                noise_qimage = QImage(noise_pil.tobytes(), noise_pil.width, noise_pil.height, QImage.Format_RGB888)
                noise_pixmap = QPixmap.fromImage(noise_qimage)
                noise_display.setPixmap(noise_pixmap)
                self.results_layout.addWidget(noise_display, row, 1)
            else:
                placeholder = QLabel("(No noise yet)")
                placeholder.setAlignment(Qt.AlignCenter)
                placeholder.setStyleSheet("border: 2px solid red; background: #f0f0f0;")
                placeholder.setMinimumSize(128, 128)
                self.results_layout.addWidget(placeholder, row, 1)
            
            # img_t-1 (next step preview - will be shown after step)
            if hasattr(self, '_previous_image') and self._previous_image is not None:
                next_display = QLabel()
                next_display.setMinimumSize(128, 128)
                next_display.setMaximumSize(128, 128)
                next_display.setStyleSheet("border: 2px solid green; background: white;")
                next_display.setScaledContents(True)
                next_pil = tensor_to_pil_v2(self._previous_image)
                next_qimage = QImage(next_pil.tobytes(), next_pil.width, next_pil.height, QImage.Format_RGB888)
                next_pixmap = QPixmap.fromImage(next_qimage)
                next_display.setPixmap(next_pixmap)
                self.results_layout.addWidget(next_display, row, 2)
            else:
                placeholder = QLabel("(Next step)")
                placeholder.setAlignment(Qt.AlignCenter)
                placeholder.setStyleSheet("border: 2px solid green; background: #f0f0f0;")
                placeholder.setMinimumSize(128, 128)
                self.results_layout.addWidget(placeholder, row, 2)
            
            # Visual conditioning (8x8)
            cond_display = QLabel()
            cond_display.setMinimumSize(80, 80)
            cond_display.setMaximumSize(80, 80)
            cond_display.setStyleSheet("border: 2px solid blue; background: white;")
            cond_display.setScaledContents(True)
            cond_pil = self.paint_widget.get_pil_image()
            cond_qimage = QImage(cond_pil.tobytes(), cond_pil.width, cond_pil.height, QImage.Format_RGB888)
            cond_pixmap = QPixmap.fromImage(cond_qimage)
            cond_display.setPixmap(cond_pixmap)
            self.results_layout.addWidget(cond_display, row, 3)
            
            row += 1
            
            # Progress information
            progress_info = QLabel(f"Progress: {state['progress_percent']:.1f}% | Variance: {state.get('variance', 'N/A')}")
            progress_info.setStyleSheet("color: #666; font-size: 11px;")
            self.results_layout.addWidget(progress_info, row, 0, 1, 4)
            
            # Store current image for next step comparison
            self._previous_image = state['image'].clone()
            
        except Exception as e:
            print(f"Error displaying manual state: {e}")

    def _update_generate_button_state(self):
        """Update generate button state (compatibility method)"""
        self._update_ui_state()

    def _generate_images(self):
        """Generate images using the paint widget prompt"""
        if not self.model or not hasattr(self, 'visual_context_tensor'):
            return

        if self.inference_thread and self.inference_thread.isRunning():
            QMessageBox.warning(self, "Generation In Progress", "A generation is already running.")
            return

        # Start inference using paint widget image
        num_samples = self.num_samples_spinbox.value()
        inference_steps = self.inference_steps_spinbox.value()
        prompt_image = self.paint_widget.get_pil_image()
        
        self.generate_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        self.inference_thread = InferenceThread(self.model, prompt_image, num_samples, inference_steps)
        self.inference_thread.inference_finished.connect(self._on_inference_finished)
        self.inference_thread.inference_error.connect(self._on_inference_error)
        self.inference_thread.progress_updated.connect(self._on_progress_updated)
        self.inference_thread.start()

    def _upload_prompt_image(self):
        """Load image from file and set it in paint widget"""
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        image_file, _ = QFileDialog.getOpenFileName(
            self, "Select Prompt Image", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)", options=options
        )
        if image_file:
            try:
                # Load image and resize to 8x8 for paint widget
                pil_image = Image.open(image_file).convert('RGB')
                pil_image_8x8 = pil_image.resize((8, 8), Image.LANCZOS)
                
                # Convert to tensor and set in paint widget
                from torchvision.transforms import ToTensor
                tensor_8x8 = ToTensor()(pil_image_8x8) * 2.0 - 1.0  # Normalize to [-1, 1]
                
                self.paint_widget.set_tensor(tensor_8x8)
                
                # Update status
                self.status_label.setText(f"Loaded and resized image: {os.path.basename(image_file)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Image Loading Error", f"Failed to load image:\n{str(e)}")

    def _load_random_from_dataset(self):
        """Load a random image from the dataset into paint widget"""
        dataset_path = "H:/Repositories/ImgDatasetGen_256"
        
        if not os.path.exists(dataset_path):
            QMessageBox.warning(self, "Dataset Not Found", 
                               f"Dataset not found at: {dataset_path}")
            return
        
        try:
            # Create dataset loader
            dataset = ImageDatasetV2(dataset_path)
            
            if len(dataset) == 0:
                QMessageBox.warning(self, "Empty Dataset", "No images found in dataset")
                return
            
            # Get random image
            random_idx = random.randint(0, len(dataset) - 1)
            main_tensor, context_tensor, img_path = dataset[random_idx]
            
            # Use the context tensor (8x8) directly in paint widget
            self.paint_widget.set_tensor(context_tensor)
            
            # Show info about selected image
            self.status_label.setText(f"Loaded random 8x8 from: {os.path.basename(img_path)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Dataset Loading Error", f"Failed to load random image:\n{str(e)}")

    def _clear_results(self):
        """Clear all widgets from results layout"""
        while self.results_layout.count():
            child = self.results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

# Example for running this tab standalone
if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication, QMainWindow
    
    app = QApplication(sys.argv)
    main_win = QMainWindow()
    inference_tab_widget = InferenceTabV2()
    
    main_win.setCentralWidget(inference_tab_widget)
    main_win.setWindowTitle("Inference Tab V2 Test")
    main_win.resize(800, 900)
    main_win.show()
    sys.exit(app.exec_())
