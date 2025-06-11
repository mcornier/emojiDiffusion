import sys
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGridLayout, QColorDialog, QGroupBox, QApplication, QMainWindow
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPalette
import torch
import numpy as np
from PIL import Image

class ColorButton(QPushButton):
    """Button representing one pixel in the 8x8 grid"""
    colorChanged = pyqtSignal(int, int, QColor)  # row, col, color
    
    def __init__(self, row, col, size=40):
        super().__init__()
        self.row = row
        self.col = col
        self.current_color = QColor(128, 128, 128)  # Default gray
        
        self.setFixedSize(size, size)
        self.setStyleSheet(f"background-color: {self.current_color.name()}; border: 1px solid black;")
        self.clicked.connect(self.on_clicked)
    
    def on_clicked(self):
        # Get the current brush color from parent widget
        parent_widget = self.parent()
        while parent_widget and not hasattr(parent_widget, 'current_brush_color'):
            parent_widget = parent_widget.parent()
        
        if parent_widget and hasattr(parent_widget, 'current_brush_color'):
            self.set_color(parent_widget.current_brush_color)
    
    def set_color(self, color):
        self.current_color = color
        self.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black;")
        self.colorChanged.emit(self.row, self.col, color)
    
    def get_color(self):
        return self.current_color

class Paint8x8Widget(QWidget):
    """8x8 Paint widget with color picker and tools"""
    gridChanged = pyqtSignal()  # Emitted when grid changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_brush_color = QColor(255, 0, 0)  # Default red
        self.grid_buttons = []
        self._setup_ui()
        self._initialize_grid()
    
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("8x8 Visual Prompt Editor")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        main_layout.addWidget(title_label)
        
        # Tools section
        tools_group = QGroupBox("Tools")
        tools_layout = QHBoxLayout(tools_group)
        
        # Current color display
        self.color_display = QPushButton()
        self.color_display.setFixedSize(40, 40)
        self.color_display.setStyleSheet(f"background-color: {self.current_brush_color.name()}; border: 2px solid black;")
        self.color_display.clicked.connect(self._pick_color)
        tools_layout.addWidget(QLabel("Brush:"))
        tools_layout.addWidget(self.color_display)
        
        # Color picker button
        pick_color_btn = QPushButton("Pick Color")
        pick_color_btn.clicked.connect(self._pick_color)
        tools_layout.addWidget(pick_color_btn)
        
        # Preset colors
        preset_colors = [
            QColor(255, 0, 0),    # Red
            QColor(0, 255, 0),    # Green  
            QColor(0, 0, 255),    # Blue
            QColor(255, 255, 0),  # Yellow
            QColor(255, 0, 255),  # Magenta
            QColor(0, 255, 255),  # Cyan
            QColor(255, 255, 255), # White
            QColor(0, 0, 0),      # Black
        ]
        
        for color in preset_colors:
            preset_btn = QPushButton()
            preset_btn.setFixedSize(25, 25)
            preset_btn.setStyleSheet(f"background-color: {color.name()}; border: 1px solid gray;")
            preset_btn.clicked.connect(lambda checked, c=color: self._set_brush_color(c))
            tools_layout.addWidget(preset_btn)
        
        tools_layout.addStretch()
        
        # Action buttons
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_grid)
        tools_layout.addWidget(clear_btn)
        
        random_btn = QPushButton("Random")
        random_btn.clicked.connect(self._randomize_grid)
        tools_layout.addWidget(random_btn)
        
        main_layout.addWidget(tools_group)
        
        # 8x8 Grid
        grid_group = QGroupBox("8x8 Grid")
        self.grid_layout = QGridLayout(grid_group)
        self.grid_layout.setSpacing(1)
        
        # Create 8x8 grid of color buttons
        for row in range(8):
            button_row = []
            for col in range(8):
                btn = ColorButton(row, col, size=35)
                btn.colorChanged.connect(self._on_pixel_changed)
                self.grid_layout.addWidget(btn, row, col)
                button_row.append(btn)
            self.grid_buttons.append(button_row)
        
        main_layout.addWidget(grid_group)
        
        # Info section
        info_group = QGroupBox("Info")
        info_layout = QVBoxLayout(info_group)
        self.tensor_info_label = QLabel("Tensor: (3, 8, 8) - RGB values in [-1, 1]")
        self.tensor_info_label.setStyleSheet("font-family: monospace; font-size: 10px;")
        info_layout.addWidget(self.tensor_info_label)
        main_layout.addWidget(info_group)
    
    def _initialize_grid(self):
        """Initialize grid with gray values"""
        gray_color = QColor(128, 128, 128)
        for row in range(8):
            for col in range(8):
                self.grid_buttons[row][col].set_color(gray_color)
    
    def _pick_color(self):
        """Open color picker dialog"""
        color = QColorDialog.getColor(self.current_brush_color, self, "Pick Brush Color")
        if color.isValid():
            self._set_brush_color(color)
    
    def _set_brush_color(self, color):
        """Set the current brush color"""
        self.current_brush_color = color
        self.color_display.setStyleSheet(f"background-color: {color.name()}; border: 2px solid black;")
    
    def _clear_grid(self):
        """Clear grid to gray"""
        gray_color = QColor(128, 128, 128)
        for row in range(8):
            for col in range(8):
                self.grid_buttons[row][col].set_color(gray_color)
    
    def _randomize_grid(self):
        """Fill grid with random colors"""
        import random
        for row in range(8):
            for col in range(8):
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                color = QColor(r, g, b)
                self.grid_buttons[row][col].set_color(color)
    
    def _on_pixel_changed(self, row, col, color):
        """Handle pixel color change"""
        self.gridChanged.emit()
        self._update_tensor_info()
    
    def _update_tensor_info(self):
        """Update tensor info display"""
        tensor = self.get_tensor()
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        mean_val = tensor.mean().item()
        self.tensor_info_label.setText(
            f"Tensor: (3, 8, 8) | Range: [{min_val:.3f}, {max_val:.3f}] | Mean: {mean_val:.3f}"
        )
    
    def get_tensor(self):
        """Get the current grid as a PyTorch tensor (3, 8, 8) normalized to [-1, 1]"""
        # Create RGB array
        rgb_array = np.zeros((8, 8, 3), dtype=np.float32)
        
        for row in range(8):
            for col in range(8):
                color = self.grid_buttons[row][col].get_color()
                rgb_array[row, col, 0] = color.red() / 255.0
                rgb_array[row, col, 1] = color.green() / 255.0
                rgb_array[row, col, 2] = color.blue() / 255.0
        
        # Convert to tensor and normalize to [-1, 1]
        tensor = torch.from_numpy(rgb_array).permute(2, 0, 1)  # (3, 8, 8)
        tensor = tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        
        return tensor
    
    def set_tensor(self, tensor):
        """Set grid from a PyTorch tensor (3, 8, 8) in range [-1, 1]"""
        # Denormalize from [-1, 1] to [0, 1]
        tensor = (tensor + 1.0) / 2.0
        tensor = tensor.clamp(0, 1)
        
        # Convert to numpy and set colors
        rgb_array = tensor.permute(1, 2, 0).numpy()  # (8, 8, 3)
        
        for row in range(8):
            for col in range(8):
                r = int(rgb_array[row, col, 0] * 255)
                g = int(rgb_array[row, col, 1] * 255)
                b = int(rgb_array[row, col, 2] * 255)
                color = QColor(r, g, b)
                self.grid_buttons[row][col].set_color(color)
    
    def get_pil_image(self):
        """Get the current grid as a PIL Image (8x8)"""
        tensor = self.get_tensor()
        # Denormalize to [0, 1]
        tensor = (tensor + 1.0) / 2.0
        tensor = tensor.clamp(0, 1)
        
        # Convert to PIL
        rgb_array = tensor.permute(1, 2, 0).numpy()  # (8, 8, 3)
        rgb_array = (rgb_array * 255).astype(np.uint8)
        
        return Image.fromarray(rgb_array, 'RGB')

# Example usage
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Test the widget
    main_win = QMainWindow()
    paint_widget = Paint8x8Widget()
    
    def on_grid_changed():
        tensor = paint_widget.get_tensor()
        print(f"Grid changed! Tensor shape: {tensor.shape}, range: [{tensor.min():.3f}, {tensor.max():.3f}]")
    
    paint_widget.gridChanged.connect(on_grid_changed)
    
    main_win.setCentralWidget(paint_widget)
    main_win.setWindowTitle("8x8 Paint Widget Test")
    main_win.resize(500, 600)
    main_win.show()
    
    sys.exit(app.exec_())
