import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QSlider, 
                            QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox, 
                            QComboBox, QFileDialog, QGroupBox, QGridLayout, QStatusBar)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon

class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # App settings
        self.setWindowTitle("Canny Edge Detector")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize variables
        self.camera = cv2.VideoCapture(0)
        self.frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.camera.get(cv2.CAP_PROP_FPS)
        
        # Image processing variables
        self.drawing = False
        self.roi_selected = False
        self.roi_start = (0, 0)
        self.roi_end = (0, 0)
        self.roi = None
        
        # Canny parameters
        self.low_threshold = 50
        self.high_threshold = 150
        self.canny_active = False
        
        # Video recording variables
        self.is_recording = False
        self.video_writer = None
        
        # Selected filter
        self.current_filter = "None"
        
        # Setup UI
        self.setup_ui()
        
        # Start timer for video feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~30 FPS
        
        # Performance tracking
        self.frame_count = 0
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(1000)  # Update FPS every second
        self.current_fps = 0

    def setup_ui(self):
        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Video display area
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setMinimumSize(800, 600)
        self.display_label.setStyleSheet("border: 2px solid #3498db; background-color: #2c3e50;")
        self.display_label.mousePressEvent = self.mouse_press_event
        self.display_label.mouseMoveEvent = self.mouse_move_event
        self.display_label.mouseReleaseEvent = self.mouse_release_event
        
        # Right panel - Controls
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignTop)
        
        # App title
        title_label = QLabel("Canny Edge Detector")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #3498db; margin-bottom: 20px;")
        control_layout.addWidget(title_label)
        
        # Canny Edge Detection Controls
        canny_group = QGroupBox("Canny Edge Detection")
        canny_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        canny_layout = QVBoxLayout()
        
        # Toggle Canny
        self.canny_checkbox = QCheckBox("Enable Canny Edge Detection")
        self.canny_checkbox.toggled.connect(self.toggle_canny)
        canny_layout.addWidget(self.canny_checkbox)
        
        # Threshold sliders
        threshold_layout = QGridLayout()
        
        # Low threshold
        threshold_layout.addWidget(QLabel("Low Threshold:"), 0, 0)
        self.low_threshold_slider = QSlider(Qt.Horizontal)
        self.low_threshold_slider.setRange(0, 255)
        self.low_threshold_slider.setValue(self.low_threshold)
        self.low_threshold_slider.valueChanged.connect(self.update_low_threshold)
        threshold_layout.addWidget(self.low_threshold_slider, 0, 1)
        self.low_threshold_value = QLabel(f"{self.low_threshold}")
        threshold_layout.addWidget(self.low_threshold_value, 0, 2)
        
        # High threshold
        threshold_layout.addWidget(QLabel("High Threshold:"), 1, 0)
        self.high_threshold_slider = QSlider(Qt.Horizontal)
        self.high_threshold_slider.setRange(0, 255)
        self.high_threshold_slider.setValue(self.high_threshold)
        self.high_threshold_slider.valueChanged.connect(self.update_high_threshold)
        threshold_layout.addWidget(self.high_threshold_slider, 1, 1)
        self.high_threshold_value = QLabel(f"{self.high_threshold}")
        threshold_layout.addWidget(self.high_threshold_value, 1, 2)
        
        canny_layout.addLayout(threshold_layout)
        canny_group.setLayout(canny_layout)
        control_layout.addWidget(canny_group)
        
        # ROI Controls
        roi_group = QGroupBox("Region of Interest (ROI)")
        roi_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        roi_layout = QVBoxLayout()
        
        self.roi_checkbox = QCheckBox("Enable ROI Selection")
        self.roi_checkbox.toggled.connect(self.toggle_roi_selection)
        roi_layout.addWidget(self.roi_checkbox)
        
        self.reset_roi_button = QPushButton("Reset ROI")
        self.reset_roi_button.clicked.connect(self.reset_roi)
        roi_layout.addWidget(self.reset_roi_button)
        
        roi_group.setLayout(roi_layout)
        control_layout.addWidget(roi_group)
        
        # Additional Filters
        filters_group = QGroupBox("Image Filters")
        filters_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        filters_layout = QVBoxLayout()
        
        filters_layout.addWidget(QLabel("Select Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["None", "Grayscale", "Sepia", "Blur", "Sharpen", "Invert"])
        self.filter_combo.currentTextChanged.connect(self.change_filter)
        filters_layout.addWidget(self.filter_combo)
        
        filters_group.setLayout(filters_layout)
        control_layout.addWidget(filters_group)
        
        # Capture Controls
        capture_group = QGroupBox("Capture Options")
        capture_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        capture_layout = QVBoxLayout()
        
        self.snapshot_button = QPushButton("Take Snapshot")
        self.snapshot_button.clicked.connect(self.take_snapshot)
        capture_layout.addWidget(self.snapshot_button)
        
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        capture_layout.addWidget(self.record_button)
        
        capture_group.setLayout(capture_layout)
        control_layout.addWidget(capture_group)
        
        # Information Group
        info_group = QGroupBox("App Statistics")
        info_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        info_layout = QVBoxLayout()
        
        self.fps_label = QLabel("FPS: 0")
        info_layout.addWidget(self.fps_label)
        
        self.resolution_label = QLabel(f"Camera Resolution: {self.frame_width}x{self.frame_height}")
        info_layout.addWidget(self.resolution_label)
        
        info_group.setLayout(info_layout)
        control_layout.addWidget(info_group)
        
        # Add control panel to main layout
        main_layout.addWidget(self.display_label, 3)  # 3:1 ratio
        main_layout.addWidget(control_panel, 1)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def update_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            return
        
        self.frame_count += 1
        
        # Flip horizontally for selfie view
        frame = cv2.flip(frame, 1)
        
        # Store original frame
        original_frame = frame.copy()
        
        # Process ROI if selected
        if self.roi_selected and self.roi is not None:
            # Extract ROI coordinates
            x1, y1, x2, y2 = self.roi
            # Ensure coordinates are within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            # Create a mask for ROI
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            
            # Process only ROI area
            if self.canny_active:
                # Apply Canny to ROI
                gray_roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
                canny_roi = cv2.Canny(blurred_roi, self.low_threshold, self.high_threshold)
                
                # Convert back to BGR for merging
                canny_colored = cv2.cvtColor(canny_roi, cv2.COLOR_GRAY2BGR)
                
                # Create masked regions
                roi_area = cv2.bitwise_and(frame, frame, mask=mask)
                
                # Create inverse mask
                mask_inv = cv2.bitwise_not(mask)
                background = cv2.bitwise_and(frame, frame, mask=mask_inv)
                
                # Place canny ROI into position
                frame[y1:y2, x1:x2] = canny_colored
                
                # Combine with background
                frame = cv2.add(background, frame)
            
            # Apply selected filter to ROI if not Canny
            if not self.canny_active and self.current_filter != "None":
                filtered_roi = self.apply_filter(frame[y1:y2, x1:x2], self.current_filter)
                
                # Create masked regions
                roi_area = cv2.bitwise_and(frame, frame, mask=mask)
                
                # Create inverse mask
                mask_inv = cv2.bitwise_not(mask)
                background = cv2.bitwise_and(frame, frame, mask=mask_inv)
                
                # Place filtered ROI into position
                frame[y1:y2, x1:x2] = filtered_roi
                
                # Combine with background
                frame = cv2.add(background, frame)
            
            # Draw rectangle around ROI
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            # Process full frame
            if self.canny_active:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)
                frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            elif self.current_filter != "None":
                frame = self.apply_filter(frame, self.current_filter)
        
        # Draw ROI selection in progress
        if self.drawing:
            cv2.rectangle(frame, self.roi_start, self.roi_end, (255, 0, 0), 2)
        
        # Record video if active
        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(frame)
        
        # Convert to QImage and display
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.display_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.display_label.width(), self.display_label.height(), 
            Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def apply_filter(self, image, filter_name):
        if filter_name == "Grayscale":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif filter_name == "Sepia":
            kernel = np.array([[0.272, 0.534, 0.131],
                              [0.349, 0.686, 0.168],
                              [0.393, 0.769, 0.189]])
            sepia = cv2.transform(image, kernel)
            sepia = np.clip(sepia, 0, 255).astype(np.uint8)
            return sepia
        elif filter_name == "Blur":
            return cv2.GaussianBlur(image, (15, 15), 0)
        elif filter_name == "Sharpen":
            kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
            return cv2.filter2D(image, -1, kernel)
        elif filter_name == "Invert":
            return cv2.bitwise_not(image)
        else:
            return image
    
    def update_fps(self):
        self.current_fps = self.frame_count
        self.frame_count = 0
        self.fps_label.setText(f"FPS: {self.current_fps}")
    
    def mouse_press_event(self, event):
        if self.roi_checkbox.isChecked():
            self.drawing = True
            self.roi_start = (int(event.x() * self.frame_width / self.display_label.width()),
                             int(event.y() * self.frame_height / self.display_label.height()))
            self.roi_end = self.roi_start
    
    def mouse_move_event(self, event):
        if self.drawing:
            self.roi_end = (int(event.x() * self.frame_width / self.display_label.width()),
                           int(event.y() * self.frame_height / self.display_label.height()))
    
    def mouse_release_event(self, event):
        if self.drawing:
            self.drawing = False
            self.roi_selected = True
            
            # Ensure start point is upper left, end point is lower right
            x1, y1 = self.roi_start
            x2, y2 = self.roi_end
            
            # Swap if needed to ensure x1,y1 is top-left and x2,y2 is bottom-right
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
                
            self.roi = (x1, y1, x2, y2)
            self.status_bar.showMessage(f"ROI Selected: ({x1},{y1}) to ({x2},{y2})")
    
    @pyqtSlot(bool)
    def toggle_canny(self, checked):
        self.canny_active = checked
    
    @pyqtSlot(bool)
    def toggle_roi_selection(self, checked):
        if not checked:
            self.roi_selected = False
            self.roi = None
    
    @pyqtSlot()
    def reset_roi(self):
        self.roi_selected = False
        self.roi = None
        self.status_bar.showMessage("ROI Reset")
    
    @pyqtSlot(int)
    def update_low_threshold(self, value):
        self.low_threshold = value
        self.low_threshold_value.setText(str(value))
    
    @pyqtSlot(int)
    def update_high_threshold(self, value):
        self.high_threshold = value
        self.high_threshold_value.setText(str(value))
    
    @pyqtSlot(str)
    def change_filter(self, filter_name):
        self.current_filter = filter_name
    
    @pyqtSlot()
    def take_snapshot(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Snapshot", "", "Images (*.png *.jpg *.jpeg)")
        if filename:
            ret, frame = self.camera.read()
            if ret:
                frame = cv2.flip(frame, 1)  # Flip for selfie view
                
                # Apply processing if needed
                if self.roi_selected and self.roi is not None:
                    x1, y1, x2, y2 = self.roi
                    if self.canny_active:
                        gray_roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                        blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
                        canny_roi = cv2.Canny(blurred_roi, self.low_threshold, self.high_threshold)
                        canny_colored = cv2.cvtColor(canny_roi, cv2.COLOR_GRAY2BGR)
                        frame[y1:y2, x1:x2] = canny_colored
                    elif self.current_filter != "None":
                        filtered_roi = self.apply_filter(frame[y1:y2, x1:x2], self.current_filter)
                        frame[y1:y2, x1:x2] = filtered_roi
                else:
                    if self.canny_active:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                        edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)
                        frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                    elif self.current_filter != "None":
                        frame = self.apply_filter(frame, self.current_filter)
                
                cv2.imwrite(filename, frame)
                self.status_bar.showMessage(f"Snapshot saved to {filename}")
    
    @pyqtSlot()
    def toggle_recording(self):
        if not self.is_recording:
            filename, _ = QFileDialog.getSaveFileName(self, "Save Video", "", "Videos (*.avi)")
            if filename:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(filename, fourcc, self.fps, 
                                                  (self.frame_width, self.frame_height))
                self.is_recording = True
                self.record_button.setText("⏹️ Stop Recording")
                self.status_bar.showMessage("Recording started...")
        else:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.is_recording = False
            self.record_button.setText("🔴 Start Recording")
            self.status_bar.showMessage("Recording stopped")
    
    def closeEvent(self, event):
        # Clean up
        if self.video_writer is not None:
            self.video_writer.release()
        
        if self.camera is not None:
            self.camera.release()
        
        self.timer.stop()
        self.fps_timer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Apply Fusion style for a modern look
    
    # Set application stylesheet for a more modern appearance
    app.setStyleSheet("""
        QMainWindow {
            background-color: #2c3e50;
            color: #ecf0f1;
        }
        QLabel {
            color: #ecf0f1;
        }
        QPushButton {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 8px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #2980b9;
        }
        QPushButton:pressed {
            background-color: #1c6ea4;
        }
        QSlider::groove:horizontal {
            border: 1px solid #999999;
            height: 8px;
            background: #3c4c5c;
            margin: 2px 0;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #3498db;
            border: 1px solid #5c5c5c;
            width: 18px;
            margin: -2px 0;
            border-radius: 9px;
        }
        QGroupBox {
            border: 2px solid #3498db;
            border-radius: 5px;
            margin-top: 12px;
            padding-top: 15px;
            color: #ecf0f1;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 5px;
            color: #3498db;
        }
        QComboBox {
            background-color: #34495e;
            color: #ecf0f1;
            border: 1px solid #3498db;
            padding: 5px;
            border-radius: 3px;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox QAbstractItemView {
            background-color: #34495e;
            color: #ecf0f1;
            selection-background-color: #3498db;
        }
        QCheckBox {
            color: #ecf0f1;
        }
        QCheckBox::indicator {
            width: 15px;
            height: 15px;
        }
        QStatusBar {
            background-color: #34495e;
            color: #ecf0f1;
        }
    """)
    
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())