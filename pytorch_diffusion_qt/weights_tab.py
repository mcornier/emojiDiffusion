import sys
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QGroupBox, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot

class WeightsTab(QWidget):
    # Signals to be caught by the main window (or a controller)
    request_model_export = pyqtSignal(str) # Argument will be the suggested/chosen path
    request_model_import = pyqtSignal(str) # Argument will be the chosen path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Controls Group ---
        controls_group = QGroupBox("Model Weights Management")
        controls_layout = QVBoxLayout()

        # Export Button
        self.export_button = QPushButton("Export Weights...")
        self.export_button.clicked.connect(self._handle_export_request)
        controls_layout.addWidget(self.export_button)

        # Import Button
        self.import_button = QPushButton("Import Weights...")
        self.import_button.clicked.connect(self._handle_import_request)
        controls_layout.addWidget(self.import_button)

        controls_group.setLayout(controls_layout)
        main_layout.addWidget(controls_group)

        # --- Status Group ---
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()

        self.export_status_label = QLabel("Export Status: Ready")
        status_layout.addWidget(self.export_status_label)

        self.import_status_label = QLabel("Import Status: Ready")
        status_layout.addWidget(self.import_status_label)

        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)

        main_layout.addStretch(1) # Pushes everything to the top
        self.setLayout(main_layout)

    def _handle_export_request(self):
        # For now, we let the main window handle the file dialog for export path
        # This signal indicates the user's intent to export.
        # The main window can then pop up a QFileDialog.getSaveFileName
        # and then call the actual export logic.
        # If a path were chosen here, it would be passed in the signal.
        self.request_model_export.emit("") # Emit with empty path, main_window handles dialog

    def _handle_import_request(self):
        # Similar to export, main window will handle file dialog for import
        # self.request_model_import.emit("") # Emit with empty path, main_window handles dialog
        # OR, we can get the path here and emit it. The issue description implies
        # the tab emits the signal and the main window intercepts to run actions.
        # Let's try getting the path here as it's a common pattern for import.

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Model Weights",
            "", # Start directory
            "PyTorch Model Files (*.pth *.pt);;All Files (*)",
            options=options
        )
        if file_path:
            self.request_model_import.emit(file_path)
        else:
            self.update_import_status("Import cancelled by user.")


    # --- Slots to update status ---
    @pyqtSlot(str)
    def update_export_status(self, message):
        self.export_status_label.setText(f"Export Status: {message}")

    @pyqtSlot(str)
    def update_import_status(self, message):
        self.import_status_label.setText(f"Import Status: {message}")


# Example for running this tab standalone (for testing UI)
if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication, QMainWindow
    app = QApplication(sys.argv)
    main_win = QMainWindow()
    weights_tab_widget = WeightsTab()

    # --- Test signal connections (optional, for standalone testing) ---
    def test_export_handler(path_suggestion):
        QMessageBox.information(main_win, "Signal Test", f"Export requested (path suggestion: '{path_suggestion}'). Main window would show save dialog.")
        # Simulate main window getting path and then updating status
        save_path, _ = QFileDialog.getSaveFileName(main_win, "Save Model Weights As...", "", "PyTorch Model Files (*.pth *.pt)")
        if save_path:
            weights_tab_widget.update_export_status(f"Simulated export to {save_path}")
        else:
            weights_tab_widget.update_export_status("Export cancelled.")

    def test_import_handler(path):
        QMessageBox.information(main_win, "Signal Test", f"Import requested for path: {path}")
        # Simulate successful import
        weights_tab_widget.update_import_status(f"Simulated import from {path}")


    weights_tab_widget.request_model_export.connect(test_export_handler)
    weights_tab_widget.request_model_import.connect(test_import_handler)
    # --- End Test signal connections ---

    main_win.setCentralWidget(weights_tab_widget)
    main_win.setWindowTitle("Weights Tab Test")
    main_win.resize(400, 300)
    main_win.show()
    sys.exit(app.exec_())
