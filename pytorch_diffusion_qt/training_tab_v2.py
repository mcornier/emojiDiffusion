import sys
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox,
    QGroupBox, QFileDialog, QMessageBox, QLineEdit, QGridLayout, QTextEdit,
    QProgressBar, QCheckBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont
import torch

# Matplotlib for loss plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

# Ensure backend modules can be imported
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

try:
    from pytorch_diffusion.train_v2 import train_diffusion_model_v2
    from pytorch_diffusion.dataset_v2 import ImageDatasetV2
    from pytorch_diffusion.utils import get_device
    import random
except ImportError as e:
    print(f"Error importing V2 training modules: {e}")
    train_diffusion_model_v2 = None

class LossPlotWidget(FigureCanvas):
    """Widget for plotting training loss in real-time"""
    
    def __init__(self, parent=None, width=8, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)
        self.setParent(parent)
        
        # Set up the plot
        self.axes = self.figure.add_subplot(111)
        self.axes.set_title('Training Loss Over Time', fontsize=12, fontweight='bold')
        self.axes.set_xlabel('Epoch')
        self.axes.set_ylabel('Loss')
        self.axes.grid(True, alpha=0.3)
        
        # Initialize empty data
        self.epochs = []
        self.losses = []
        self.line = None
        self.best_line = None
        
        # Styling
        self.figure.patch.set_facecolor('white')
        
    def update_plot(self, epoch, loss):
        """Update the plot with new loss data"""
        self.epochs.append(epoch)
        self.losses.append(loss)
        
        # Clear and replot
        self.axes.clear()
        self.axes.set_title('Training Loss Over Time', fontsize=12, fontweight='bold')
        self.axes.set_xlabel('Epoch')
        self.axes.set_ylabel('Loss')
        self.axes.grid(True, alpha=0.3)
        
        if len(self.epochs) > 0:
            # Plot main loss line
            self.line = self.axes.plot(self.epochs, self.losses, 'b-', linewidth=2, 
                                     label=f'Current: {loss:.6f}')[0]
            
            # Mark best loss
            if len(self.losses) > 0:
                best_loss = min(self.losses)
                best_epoch = self.epochs[self.losses.index(best_loss)]
                self.axes.plot(best_epoch, best_loss, 'ro', markersize=8, 
                             label=f'Best: {best_loss:.6f}')
            
            # Add trend line if we have enough points
            if len(self.epochs) >= 3:
                try:
                    z = np.polyfit(self.epochs, self.losses, 1)
                    p = np.poly1d(z)
                    trend_color = 'green' if z[0] < 0 else 'red'  # Green if decreasing
                    self.axes.plot(self.epochs, p(self.epochs), '--', 
                                 color=trend_color, alpha=0.7, linewidth=1,
                                 label=f'Trend: {"↓" if z[0] < 0 else "↑"}')
                except:
                    pass
            
            self.axes.legend(loc='upper right', fontsize=10)
            
            # Auto-scale with some padding
            if len(self.losses) > 1:
                y_min, y_max = min(self.losses), max(self.losses)
                y_range = y_max - y_min
                if y_range > 0:
                    padding = y_range * 0.1
                    self.axes.set_ylim(y_min - padding, y_max + padding)
        
        # Refresh the canvas
        self.draw()
    
    def clear_plot(self):
        """Clear all data and reset the plot"""
        self.epochs = []
        self.losses = []
        self.axes.clear()
        self.axes.set_title('Training Loss Over Time', fontsize=12, fontweight='bold')
        self.axes.set_xlabel('Epoch')
        self.axes.set_ylabel('Loss')
        self.axes.grid(True, alpha=0.3)
        self.draw()

class TrainingThread(QThread):
    """Thread for running training without blocking the UI"""
    progress_updated = pyqtSignal(str, int)  # message, percentage
    loss_updated = pyqtSignal(int, float)  # epoch, loss
    checkpoint_saved = pyqtSignal(str)  # checkpoint_path
    training_finished = pyqtSignal()
    training_error = pyqtSignal(str)

    def __init__(self, training_params):
        super().__init__()
        self.training_params = training_params
        self.should_stop = False

    def run(self):
        try:
            # Mock callbacks that emit signals
            class ProgressCallback:
                def __init__(self, thread):
                    self.thread = thread
                def emit(self, message, percentage):
                    if not self.thread.should_stop:
                        self.thread.progress_updated.emit(message, percentage)

            class LossCallback:
                def __init__(self, thread):
                    self.thread = thread
                def emit(self, epoch, loss):
                    if not self.thread.should_stop:
                        self.thread.loss_updated.emit(epoch, loss)

            class CheckpointCallback:
                def __init__(self, thread):
                    self.thread = thread
                def emit(self, path):
                    if not self.thread.should_stop:
                        self.thread.checkpoint_saved.emit(path)

            # Start training with callbacks
            model = train_diffusion_model_v2(
                dataset_root=self.training_params['dataset_root'],
                epochs=self.training_params['epochs'],
                batch_size=self.training_params['batch_size'],
                accumulation_steps=self.training_params['accumulation_steps'],
                learning_rate=self.training_params['learning_rate'],
                img_size=self.training_params['img_size'],
                context_size=self.training_params['context_size'],
                model_checkpoint_path=self.training_params.get('model_checkpoint_path'),
                save_checkpoint_prefix=self.training_params['save_checkpoint_prefix'],
                save_interval=self.training_params['save_interval'],
                progress_callback=ProgressCallback(self),
                loss_callback=LossCallback(self),
                checkpoint_saved_callback=CheckpointCallback(self),
                use_multi_gpu=self.training_params['use_multi_gpu']
            )

            if not self.should_stop:
                self.training_finished.emit()

        except Exception as e:
            self.training_error.emit(str(e))

    def stop(self):
        self.should_stop = True

class TrainingTabV2(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.training_thread = None
        self.loss_history = []
        self._setup_ui()
        self._check_dataset()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # 1. Dataset Information Group
        dataset_group = QGroupBox("Dataset Information")
        dataset_layout = QGridLayout(dataset_group)

        dataset_layout.addWidget(QLabel("Dataset Path:"), 0, 0)
        self.dataset_path_edit = QLineEdit("H:/Repositories/ImgDatasetGen_256")
        dataset_layout.addWidget(self.dataset_path_edit, 0, 1)
        self.browse_dataset_button = QPushButton("Browse...")
        self.browse_dataset_button.clicked.connect(self._browse_dataset)
        dataset_layout.addWidget(self.browse_dataset_button, 0, 2)

        self.refresh_dataset_button = QPushButton("Refresh Info")
        self.refresh_dataset_button.clicked.connect(self._check_dataset)
        dataset_layout.addWidget(self.refresh_dataset_button, 1, 0)

        self.dataset_info_label = QLabel("Checking dataset...")
        dataset_layout.addWidget(self.dataset_info_label, 1, 1, 1, 2)

        main_layout.addWidget(dataset_group)

        # 2. Training Parameters Group
        params_group = QGroupBox("Training Parameters")
        params_layout = QGridLayout(params_group)

        # Basic parameters
        params_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 1000)
        self.epochs_spinbox.setValue(50)
        params_layout.addWidget(self.epochs_spinbox, 0, 1)

        params_layout.addWidget(QLabel("Batch Size (per GPU):"), 0, 2)
        self.batch_size_spinbox = QSpinBox()
        self.batch_size_spinbox.setRange(1, 16384)
        self.batch_size_spinbox.setValue(1024)
        self.batch_size_spinbox.setToolTip("Batch size per GPU - vous avez 24GB/GPU donc max ~16k")
        params_layout.addWidget(self.batch_size_spinbox, 0, 3)

        params_layout.addWidget(QLabel("Gradient Accumulation:"), 1, 0)
        self.accumulation_spinbox = QSpinBox()
        self.accumulation_spinbox.setRange(1, 16)
        self.accumulation_spinbox.setValue(4)
        self.accumulation_spinbox.setToolTip("Steps to accumulate gradients")
        params_layout.addWidget(self.accumulation_spinbox, 1, 1)

        params_layout.addWidget(QLabel("Learning Rate:"), 1, 2)
        self.learning_rate_spinbox = QDoubleSpinBox()
        self.learning_rate_spinbox.setRange(1e-6, 1e-2)
        self.learning_rate_spinbox.setValue(1e-4)
        self.learning_rate_spinbox.setDecimals(6)
        self.learning_rate_spinbox.setSingleStep(1e-5)
        params_layout.addWidget(self.learning_rate_spinbox, 1, 3)

        # Model parameters
        params_layout.addWidget(QLabel("Image Size:"), 2, 0)
        self.img_size_spinbox = QSpinBox()
        self.img_size_spinbox.setRange(32, 128)
        self.img_size_spinbox.setValue(64)
        self.img_size_spinbox.setEnabled(False)  # Fixed for V2
        params_layout.addWidget(self.img_size_spinbox, 2, 1)

        params_layout.addWidget(QLabel("Context Size:"), 2, 2)
        self.context_size_spinbox = QSpinBox()
        self.context_size_spinbox.setRange(4, 16)
        self.context_size_spinbox.setValue(8)
        self.context_size_spinbox.setEnabled(False)  # Fixed for V2
        params_layout.addWidget(self.context_size_spinbox, 2, 3)

        # Diffusion parameters
        params_layout.addWidget(QLabel("Training Timesteps:"), 3, 0)
        self.timesteps_spinbox = QSpinBox()
        self.timesteps_spinbox.setRange(100, 2000)
        self.timesteps_spinbox.setValue(200)
        self.timesteps_spinbox.setToolTip("Nombre d'étapes de diffusion pour le training")
        params_layout.addWidget(self.timesteps_spinbox, 3, 1)

        params_layout.addWidget(QLabel("Beta Start:"), 3, 2)
        self.beta_start_spinbox = QDoubleSpinBox()
        self.beta_start_spinbox.setRange(0.0001, 0.01)
        self.beta_start_spinbox.setValue(0.0001)
        self.beta_start_spinbox.setDecimals(6)
        self.beta_start_spinbox.setSingleStep(0.0001)
        self.beta_start_spinbox.setToolTip("Valeur beta initiale pour le scheduler de bruit")
        params_layout.addWidget(self.beta_start_spinbox, 3, 3)

        params_layout.addWidget(QLabel("Beta End:"), 4, 0)
        self.beta_end_spinbox = QDoubleSpinBox()
        self.beta_end_spinbox.setRange(0.001, 0.1)
        self.beta_end_spinbox.setValue(0.02)
        self.beta_end_spinbox.setDecimals(4)
        self.beta_end_spinbox.setSingleStep(0.001)
        self.beta_end_spinbox.setToolTip("Valeur beta finale pour le scheduler de bruit")
        params_layout.addWidget(self.beta_end_spinbox, 4, 1)

        # Checkpoint parameters
        params_layout.addWidget(QLabel("Save Interval:"), 4, 2)
        self.save_interval_spinbox = QSpinBox()
        self.save_interval_spinbox.setRange(1, 50)
        self.save_interval_spinbox.setValue(5)
        self.save_interval_spinbox.setToolTip("Save checkpoint every N epochs")
        params_layout.addWidget(self.save_interval_spinbox, 4, 3)

        params_layout.addWidget(QLabel("Checkpoint Prefix:"), 5, 0)
        self.checkpoint_prefix_edit = QLineEdit("diffusion_v2_ckpt")
        params_layout.addWidget(self.checkpoint_prefix_edit, 5, 1)

        # Multi-GPU option
        self.multi_gpu_checkbox = QCheckBox("Use Multi-GPU (DataParallel)")
        self.multi_gpu_checkbox.setChecked(True)
        params_layout.addWidget(self.multi_gpu_checkbox, 5, 2, 1, 2)

        # Resume training
        params_layout.addWidget(QLabel("Resume from:"), 6, 0)
        self.resume_checkbox = QCheckBox("Resume from checkpoint")
        params_layout.addWidget(self.resume_checkbox, 6, 1, 1, 2)

        main_layout.addWidget(params_group)

        # 3. Resume Training Group
        resume_group = QGroupBox("Resume Training (Optional)")
        resume_layout = QGridLayout(resume_group)

        resume_layout.addWidget(QLabel("Checkpoint Path:"), 0, 0)
        self.resume_path_edit = QLineEdit()
        self.resume_path_edit.setPlaceholderText("Path to checkpoint .pth file")
        resume_layout.addWidget(self.resume_path_edit, 0, 1)
        self.browse_resume_button = QPushButton("Browse...")
        self.browse_resume_button.clicked.connect(self._browse_resume)
        resume_layout.addWidget(self.browse_resume_button, 0, 2)

        main_layout.addWidget(resume_group)

        # 4. Training Controls Group
        controls_group = QGroupBox("Training Controls")
        controls_layout = QVBoxLayout(controls_group)

        # Control buttons
        buttons_layout = QHBoxLayout()
        self.start_training_button = QPushButton("🚀 Start Training")
        self.start_training_button.clicked.connect(self._start_training)
        self.start_training_button.setEnabled(False)
        buttons_layout.addWidget(self.start_training_button)

        self.stop_training_button = QPushButton("⏹️ Stop Training")
        self.stop_training_button.clicked.connect(self._stop_training)
        self.stop_training_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_training_button)

        buttons_layout.addStretch()
        controls_layout.addLayout(buttons_layout)

        # Progress information
        self.progress_label = QLabel("Ready to start training")
        controls_layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)

        # GPU info
        self.gpu_info_label = QLabel("Checking GPU...")
        controls_layout.addWidget(self.gpu_info_label)

        main_layout.addWidget(controls_group)

        # 5. Training Log Group
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)

        # Loss display
        loss_layout = QHBoxLayout()
        loss_layout.addWidget(QLabel("Current Loss:"))
        self.current_loss_label = QLabel("---")
        self.current_loss_label.setStyleSheet("font-weight: bold; color: blue;")
        loss_layout.addWidget(self.current_loss_label)
        loss_layout.addStretch()

        loss_layout.addWidget(QLabel("Best Loss:"))
        self.best_loss_label = QLabel("---")
        self.best_loss_label.setStyleSheet("font-weight: bold; color: green;")
        loss_layout.addWidget(self.best_loss_label)
        loss_layout.addStretch()

        log_layout.addLayout(loss_layout)

        main_layout.addWidget(log_group)

        # 6. Loss Plot Group
        plot_group = QGroupBox("Loss Graph")
        plot_layout = QVBoxLayout(plot_group)

        # Create loss plot widget
        self.loss_plot = LossPlotWidget(width=8, height=4, dpi=80)
        plot_layout.addWidget(self.loss_plot)

        # Clear plot button
        clear_plot_layout = QHBoxLayout()
        self.clear_plot_button = QPushButton("Clear Graph")
        self.clear_plot_button.clicked.connect(self._clear_loss_plot)
        clear_plot_layout.addWidget(self.clear_plot_button)
        clear_plot_layout.addStretch()
        plot_layout.addLayout(clear_plot_layout)

        main_layout.addWidget(plot_group)

        self.setLayout(main_layout)

        # Check GPU info
        self._check_gpu_info()

    def _browse_dataset(self):
        dataset_dir = QFileDialog.getExistingDirectory(
            self, "Select Dataset Directory", self.dataset_path_edit.text()
        )
        if dataset_dir:
            self.dataset_path_edit.setText(dataset_dir)
            self._check_dataset()

    def _browse_resume(self):
        checkpoint_file, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint File", "checkpoints/",
            "PyTorch Checkpoint Files (*.pth *.pt);;All Files (*)"
        )
        if checkpoint_file:
            self.resume_path_edit.setText(checkpoint_file)

    def _check_dataset(self):
        dataset_path = self.dataset_path_edit.text()
        
        try:
            if not os.path.exists(dataset_path):
                self.dataset_info_label.setText("❌ Dataset path does not exist")
                self.start_training_button.setEnabled(False)
                return

            # Check dataset
            dataset = ImageDatasetV2(dataset_path)
            num_images = len(dataset)
            
            if num_images == 0:
                self.dataset_info_label.setText("❌ No images found in dataset")
                self.start_training_button.setEnabled(False)
            else:
                self.dataset_info_label.setText(f"✅ Found {num_images:,} images ready for training")
                self.start_training_button.setEnabled(True)
                
                # Calculate training info
                batch_size = self.batch_size_spinbox.value()
                accumulation = self.accumulation_spinbox.value()
                num_gpus = torch.cuda.device_count() if self.multi_gpu_checkbox.isChecked() else 1
                effective_batch = batch_size * num_gpus * accumulation
                
                steps_per_epoch = num_images // effective_batch
                self._log(f"Training setup: {effective_batch} effective batch size, {steps_per_epoch} steps/epoch")
                
        except Exception as e:
            self.dataset_info_label.setText(f"❌ Error checking dataset: {str(e)}")
            self.start_training_button.setEnabled(False)

    def _check_gpu_info(self):
        try:
            device = get_device()
            if device.type == 'cuda':
                num_gpus = torch.cuda.device_count()
                gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
                memory_info = []
                for i in range(num_gpus):
                    mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
                    memory_info.append(f"{mem_total:.1f}GB")
                
                gpu_info = f"✅ {num_gpus} GPU(s): {', '.join([f'{name} ({mem})' for name, mem in zip(gpu_names, memory_info)])}"
                self.gpu_info_label.setText(gpu_info)
            else:
                self.gpu_info_label.setText("⚠️ CUDA not available - using CPU (very slow)")
        except Exception as e:
            self.gpu_info_label.setText(f"❌ Error checking GPU: {str(e)}")

    def _start_training(self):
        if self.training_thread and self.training_thread.isRunning():
            QMessageBox.warning(self, "Training In Progress", "Training is already running.")
            return

        # Validate parameters
        if not self._validate_training_params():
            return

        # Prepare training parameters
        training_params = {
            'dataset_root': self.dataset_path_edit.text(),
            'epochs': self.epochs_spinbox.value(),
            'batch_size': self.batch_size_spinbox.value(),
            'accumulation_steps': self.accumulation_spinbox.value(),
            'learning_rate': self.learning_rate_spinbox.value(),
            'img_size': self.img_size_spinbox.value(),
            'context_size': self.context_size_spinbox.value(),
            'save_checkpoint_prefix': self.checkpoint_prefix_edit.text(),
            'save_interval': self.save_interval_spinbox.value(),
            'use_multi_gpu': self.multi_gpu_checkbox.isChecked()
        }

        # Add resume checkpoint if specified
        if self.resume_checkbox.isChecked() and self.resume_path_edit.text():
            training_params['model_checkpoint_path'] = self.resume_path_edit.text()

        # Start training thread
        self.training_thread = TrainingThread(training_params)
        self.training_thread.progress_updated.connect(self._on_progress_updated)
        self.training_thread.loss_updated.connect(self._on_loss_updated)
        self.training_thread.checkpoint_saved.connect(self._on_checkpoint_saved)
        self.training_thread.training_finished.connect(self._on_training_finished)
        self.training_thread.training_error.connect(self._on_training_error)

        # Update UI state
        self.start_training_button.setEnabled(False)
        self.stop_training_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)

        # Clear loss history and plot
        self.loss_history = []
        self.current_loss_label.setText("---")
        self.best_loss_label.setText("---")
        self.loss_plot.clear_plot()

        self._log("🚀 Starting training...")
        self.training_thread.start()

    def _stop_training(self):
        if self.training_thread and self.training_thread.isRunning():
            self._log("⏹️ Stopping training...")
            self.training_thread.stop()
            self.training_thread.wait(5000)  # Wait up to 5 seconds
            if self.training_thread.isRunning():
                self.training_thread.terminate()
            self._on_training_finished()

    def _validate_training_params(self):
        # Check dataset
        if not os.path.exists(self.dataset_path_edit.text()):
            QMessageBox.warning(self, "Invalid Parameters", "Dataset path does not exist.")
            return False

        # Check resume checkpoint if specified
        if self.resume_checkbox.isChecked():
            resume_path = self.resume_path_edit.text()
            if not resume_path or not os.path.exists(resume_path):
                QMessageBox.warning(self, "Invalid Parameters", "Resume checkpoint path is invalid.")
                return False

        return True

    def _on_progress_updated(self, message, percentage):
        self.progress_label.setText(message)
        self.progress_bar.setValue(percentage)
        self._log(f"Progress: {message}")

    def _on_loss_updated(self, epoch, loss):
        self.loss_history.append((epoch, loss))
        self.current_loss_label.setText(f"{loss:.6f}")
        
        # Update best loss
        best_loss = min([l for _, l in self.loss_history])
        self.best_loss_label.setText(f"{best_loss:.6f}")
        
        # Update loss plot
        self.loss_plot.update_plot(epoch, loss)
        
        self._log(f"Epoch {epoch}: Loss = {loss:.6f}")

    def _clear_loss_plot(self):
        """Clear the loss plot"""
        self.loss_plot.clear_plot()
        self._log("📊 Loss graph cleared")

    def _on_checkpoint_saved(self, checkpoint_path):
        self._log(f"💾 Checkpoint saved: {os.path.basename(checkpoint_path)}")

    def _on_training_finished(self):
        self.start_training_button.setEnabled(True)
        self.stop_training_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.progress_label.setText("Training completed!")
        self._log("✅ Training finished!")

    def _on_training_error(self, error_message):
        self.start_training_button.setEnabled(True)
        self.stop_training_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.progress_label.setText("Training failed!")
        self._log(f"❌ Training error: {error_message}")
        QMessageBox.critical(self, "Training Error", f"Training failed:\n{error_message}")

    def _log(self, message):
        """Add message to training log"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        # Auto-scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.End)
        self.log_text.setTextCursor(cursor)

# Example for running this tab standalone
if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication, QMainWindow
    
    app = QApplication(sys.argv)
    main_win = QMainWindow()
    training_tab_widget = TrainingTabV2()
    
    main_win.setCentralWidget(training_tab_widget)
    main_win.setWindowTitle("Training Tab V2 Test")
    main_win.resize(900, 800)
    main_win.show()
    sys.exit(app.exec_())
