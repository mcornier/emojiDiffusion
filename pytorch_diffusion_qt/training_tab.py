import sys
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QGroupBox, QProgressBar, QFileDialog, QMessageBox, QLineEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Matplotlib imports for plotting loss
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Ensure backend modules can be imported
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

try:
    from pytorch_diffusion.train import train_diffusion_model
    # We might need access to some constants or model class for parameter validation if not passed directly
    # For now, train_diffusion_model encapsulates most of it.
except ImportError as e:
    print(f"Error importing backend training module: {e}. GUI will have limited functionality.")
    def train_diffusion_model(*args, **kwargs):
        print("Backend 'train_diffusion_model' not available.")
        # Simulate some progress for UI testing
        if 'progress_callback' in kwargs and 'loss_callback' in kwargs:
            for i in range(kwargs.get('epochs', 10)):
                kwargs['progress_callback'].emit(f"Simulated Epoch {i+1}/{kwargs.get('epochs',10)}", int((i+1)/kwargs.get('epochs',10)*100) )
                kwargs['loss_callback'].emit(i+1, 1.0 / (i + 1)) # Simulated loss
                QThread.msleep(200) # Simulate work
        return None

# --- Worker Thread for Training ---
class TrainingThread(QThread):
    progress_updated = pyqtSignal(str, int)  # message, percentage
    epoch_loss_updated = pyqtSignal(int, float) # epoch_num, loss_value
    training_finished = pyqtSignal(str) # message (success or error)
    model_trained_path = pyqtSignal(str) # path to the last saved model checkpoint

    def __init__(self, epochs, batch_size, learning_rate, sample_emojis, font_path, model_checkpoint_path=None):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.sample_emojis = sample_emojis # This should come from DatasetTab or be configurable
        self.font_path = font_path         # This too
        self.model_checkpoint_path = model_checkpoint_path
        self._is_running = True

    def run(self):
        try:
            self.progress_updated.emit("Starting training...", 0)

            # Wrapper to pass Qt signals to the backend training function
            def qt_progress_callback(message, percentage):
                if not self._is_running:
                    raise InterruptedError("Training stopped by user.")
                self.progress_updated.emit(message, percentage)

            def qt_loss_callback(epoch, loss):
                if not self._is_running:
                    raise InterruptedError("Training stopped by user.")
                self.epoch_loss_updated.emit(epoch, loss)

            def qt_checkpoint_saved_callback(path):
                if not self._is_running:
                    raise InterruptedError("Training stopped by user.")
                self.model_trained_path.emit(path)


            # The train_diffusion_model function needs to be adapted or wrapped
            # to accept these callbacks if it doesn't already.
            # For now, assuming we might modify train_diffusion_model or use a wrapper.
            # Let's assume train_diffusion_model is modified to accept these:
            # (This is a conceptual adaptation; actual backend 'train_diffusion_model' would need these hooks)

            # Simplified: Directly call and poll status if not callback based
            # Or, the train_diffusion_model in the backend should be designed to be callable like this.
            # For the dummy function, we passed callbacks.
            # The real train_diffusion_model would need modification for live Qt updates.
            # For now, we assume the backend is structured to allow this.

            # This is a placeholder for how the actual training function would be called
            # with callbacks for GUI updates.
            # The actual `train_diffusion_model` from `pytorch_diffusion.train`
            # currently logs to console. It would need refactoring to support these callbacks.
            # For this subtask, we'll proceed as if it does, or use the dummy.

            # Let's assume we'll modify `train_diffusion_model` later if needed.
            # For now, this structure allows UI development.
            # The dummy function above already uses these.

            trained_model_object = train_diffusion_model(
                epochs=self.epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                sample_emojis=self.sample_emojis, # This needs to be set based on dataset tab
                font_path=self.font_path,         # This too
                model_checkpoint_path=self.model_checkpoint_path,
                save_checkpoint_prefix="gui_diffusion_ckpt",
                save_interval=1, # Save more frequently for GUI feedback on path
                # Conceptual additions for GUI feedback:
                progress_callback=qt_progress_callback,
                loss_callback=qt_loss_callback,
                checkpoint_saved_callback=qt_checkpoint_saved_callback
            )

            if self._is_running:
                self.training_finished.emit("Training completed successfully.")
            else:
                self.training_finished.emit("Training stopped by user.")

        except InterruptedError:
            self.training_finished.emit("Training stopped by user.")
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.training_finished.emit(f"Training failed: {str(e)}")

    def stop(self):
        self._is_running = False
        self.progress_updated.emit("Stopping training...", 100)


class TrainingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.training_thread = None
        self.current_dataset_emojis = None
        self.current_emoji_to_idx = None
        self.current_font_path = None
        self.current_vocab_size = 0
        self.epochs_history = []
        self.loss_history = []

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # 1. Parameters Group
        params_group = QGroupBox("Training Parameters")
        params_layout = QGridLayout(params_group)

        params_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 10000)
        self.epochs_spinbox.setValue(50)
        params_layout.addWidget(self.epochs_spinbox, 0, 1)

        params_layout.addWidget(QLabel("Learning Rate:"), 1, 0)
        self.lr_spinbox = QDoubleSpinBox()
        self.lr_spinbox.setRange(1e-6, 1e-1)
        self.lr_spinbox.setDecimals(6)
        self.lr_spinbox.setSingleStep(1e-5)
        self.lr_spinbox.setValue(1e-4)
        params_layout.addWidget(self.lr_spinbox, 1, 1)

        params_layout.addWidget(QLabel("Batch Size:"), 2, 0)
        self.batch_size_spinbox = QSpinBox()
        self.batch_size_spinbox.setRange(1, 512) # Adjust max based on typical memory
        self.batch_size_spinbox.setValue(32)
        params_layout.addWidget(self.batch_size_spinbox, 2, 1)

        params_layout.addWidget(QLabel("Load Checkpoint (Optional):"), 3, 0)
        self.checkpoint_path_edit = QLineEdit()
        self.checkpoint_path_edit.setPlaceholderText("Path to .pth model checkpoint")
        params_layout.addWidget(self.checkpoint_path_edit, 3, 1)
        self.browse_checkpoint_button = QPushButton("Browse...")
        self.browse_checkpoint_button.clicked.connect(self._browse_checkpoint)
        params_layout.addWidget(self.browse_checkpoint_button, 3, 2)

        main_layout.addWidget(params_group)

        # 2. Controls Group
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout(controls_group)
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self._start_training)
        controls_layout.addWidget(self.start_button)
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.clicked.connect(self._stop_training)
        self.stop_button.setEnabled(False)
        controls_layout.addWidget(self.stop_button)
        main_layout.addWidget(controls_group)

        # 3. Status Group
        status_group = QGroupBox("Training Status")
        status_layout = QVBoxLayout(status_group)
        self.status_label = QLabel("Status: Ready")
        status_layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)
        self.last_saved_model_label = QLabel("Last saved model: N/A")
        status_layout.addWidget(self.last_saved_model_label)
        main_layout.addWidget(status_group)

        # 4. Loss Plot Group
        plot_group = QGroupBox("Loss Curve")
        plot_layout = QVBoxLayout(plot_group)
        self.figure = Figure(figsize=(5,3)) # Smaller figure size for embedding
        self.loss_canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.grid(True)
        self.figure.tight_layout() # Adjust layout to prevent labels overlapping
        plot_layout.addWidget(self.loss_canvas)
        main_layout.addWidget(plot_group)

        self.setLayout(main_layout)

    def _browse_checkpoint(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        ckpt_file, _ = QFileDialog.getOpenFileName(self, "Select Model Checkpoint", "",
                                                  "PyTorch Checkpoint Files (*.pth *.pt);;All Files (*)", options=options)
        if ckpt_file:
            self.checkpoint_path_edit.setText(ckpt_file)

    @pyqtSlot(list, dict, str, int) # emojis, emoji_to_idx, font_path, vocab_size
    def update_active_dataset_info(self, emojis: list, emoji_to_idx: dict, font_path: str, vocab_size: int):
        self.current_dataset_emojis = emojis
        self.current_emoji_to_idx = emoji_to_idx
        self.current_font_path = font_path
        self.current_vocab_size = vocab_size

        if emojis:
            font_display_name = 'Default' if not font_path else os.path.basename(font_path)
            QMessageBox.information(self, "Dataset Updated",
                                  f"Training tab received dataset with {len(emojis)} emojis (Vocab: {vocab_size}). Font: {font_display_name}")
            self.status_label.setText(f"Status: Ready to train with new dataset (Vocab: {vocab_size}).")
            # Enable start button if it was disabled due to no dataset
            if not self.start_button.isEnabled() and not (self.training_thread and self.training_thread.isRunning()):
                 self.start_button.setEnabled(True)
        else:
            QMessageBox.warning(self, "Dataset Issue", "Received empty or invalid dataset information.")
            self.status_label.setText(f"Status: Waiting for valid dataset.")
            self.start_button.setEnabled(False) # Disable if dataset is not valid


    def _start_training(self):
        if self.training_thread and self.training_thread.isRunning():
            QMessageBox.warning(self, "Training In Progress", "A training session is already running.")
            return

        if not self.current_dataset_emojis or self.current_vocab_size == 0:
             QMessageBox.warning(self, "No Dataset", "Please generate a dataset in the 'Dataset' tab and ensure it's loaded here.")
             self.status_label.setText("Status: Cannot start training. Dataset not available or empty.")
             return


        epochs = self.epochs_spinbox.value()
        lr = self.lr_spinbox.value()
        batch_size = self.batch_size_spinbox.value()
        checkpoint_path = self.checkpoint_path_edit.text() if self.checkpoint_path_edit.text() else None

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Status: Starting training...")
        self.progress_bar.setValue(0)
        self.last_saved_model_label.setText("Last saved model: N/A")

        self.epochs_history.clear()
        self.loss_history.clear()
        self._update_loss_plot() # Clear plot

        self.training_thread = TrainingThread(
            epochs, batch_size, lr,
            sample_emojis=self.current_dataset_emojis,
            font_path=self.current_font_path, # This also needs to be obtained from DatasetTab
            model_checkpoint_path=checkpoint_path
        )
        self.training_thread.progress_updated.connect(self._update_training_progress)
        self.training_thread.epoch_loss_updated.connect(self._update_epoch_loss)
        self.training_thread.training_finished.connect(self._on_training_finished)
        self.training_thread.model_trained_path.connect(self._update_last_saved_model)
        self.training_thread.start()

    def _stop_training(self):
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop()
            # UI updates (button states, status) will be handled by _on_training_finished
        else:
            self.stop_button.setEnabled(False)

    def _update_training_progress(self, message, percentage):
        self.status_label.setText(f"Status: {message}")
        self.progress_bar.setValue(percentage)

    def _update_epoch_loss(self, epoch_num, loss_value):
        self.epochs_history.append(epoch_num)
        self.loss_history.append(loss_value)
        self._update_loss_plot()

    def _update_loss_plot(self):
        self.ax.clear()
        if self.epochs_history and self.loss_history:
            self.ax.plot(self.epochs_history, self.loss_history, marker='o', linestyle='-')
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training Loss Over Epochs")
        self.ax.grid(True)
        self.figure.tight_layout()
        self.loss_canvas.draw()

    def _update_last_saved_model(self, path):
        self.last_saved_model_label.setText(f"Last saved model: {os.path.basename(path)}")


    def _on_training_finished(self, message):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText(f"Status: {message}")
        self.progress_bar.setValue(100 if "completed" in message.lower() or "stopped" in message.lower() else self.progress_bar.value())
        if "failed" in message.lower() or "error" in message.lower():
            QMessageBox.critical(self, "Training Error", message)
        else:
            QMessageBox.information(self, "Training Status", message)
        self.training_thread = None # Clear thread reference


# Example for running this tab standalone
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = QMainWindow()
    training_tab_widget = TrainingTab()

    # Simulate receiving dataset info (normally connected to DatasetTab's signal)
    # training_tab_widget.update_active_dataset(['😀', '🚀'], 2)

    main_win.setCentralWidget(training_tab_widget)
    main_win.setWindowTitle("Training Tab Test")
    main_win.resize(700, 800)
    main_win.show()
    sys.exit(app.exec_())
