from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
import os
import re

# Add path for core imports FIRST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Now import the modules (after path is set)
from core.file_handler import FileHandler
from ocr.ocr_engine import OCREngine
from converters.pdf_to_image import PDFConverter

# Import PIL for image type hints and conversion (with safe fallback)
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PILImage = None
    PIL_AVAILABLE = False


# ✨ OCR Worker Class - Non-blocking processing ✨
class OCRWorker(QThread):
    """Background worker for OCR processing - prevents UI freezing"""
    
    # Signals for communication with main thread
    progress_updated = pyqtSignal(int, str)  # progress%, message
    extraction_completed = pyqtSignal(dict)  # OCR result
    extraction_failed = pyqtSignal(str)      # Error message
    
    def __init__(self, ocr_engine):
        super().__init__()
        self.ocr_engine = ocr_engine
        self.image_data = None
        self.file_path = None
        self.processing_type = None
        self._is_cancelled = False
    
    def setup_enhanced_pdf_extraction(self, pil_image, structure_hints=None, preprocess=True):
        """Setup enhanced extraction for PDF with structure preservation"""
        self.processing_type = 'enhanced_pdf'
        self.image_data = pil_image
        self.structure_hints = structure_hints
        self.preprocess = preprocess
        self._is_cancelled = False
    
    def setup_extraction(self, processing_type, **kwargs):
        """Setup extraction parameters"""
        self.processing_type = processing_type
        self._is_cancelled = False
        
        if processing_type == 'image_file':
            self.file_path = kwargs.get('file_path')
        elif processing_type == 'pil_image':
            self.image_data = kwargs.get('pil_image')
        
        self.preprocess = kwargs.get('preprocess', True)
    
    def cancel_extraction(self):
        """Cancel the current extraction"""
        self._is_cancelled = True
        self.quit()
        self.wait()
    
    def run(self):
        """Background OCR processing with enhanced support"""
        try:
            if self._is_cancelled:
                return
            
            # Update progress
            self.progress_updated.emit(10, "Initializing OCR engine...")
            self.msleep(100)  # Small delay for UI update
            
            if self.processing_type == 'enhanced_pdf':
                # Process PDF with structure preservation
                self.progress_updated.emit(30, "Processing PDF with structure preservation...")
                self.msleep(100)
                
                if self._is_cancelled:
                    return
                
                self.progress_updated.emit(50, "Running advanced OCR analysis...")
                self.msleep(100)
                
                # Use the structure-aware extraction method
                result = self.ocr_engine.extract_structured_pdf_text(
                    self.image_data, 
                    self.structure_hints,
                    preprocess=self.preprocess
                )
                
            elif self.processing_type == 'image_file':
                # Process image file
                self.progress_updated.emit(30, "Loading image file...")
                self.msleep(100)
                
                if self._is_cancelled:
                    return
                
                self.progress_updated.emit(50, "Running OCR analysis...")
                self.msleep(100)
                
                result = self.ocr_engine.extract_text_from_image(
                    self.file_path, 
                    preprocess=self.preprocess
                )
                
            elif self.processing_type == 'pil_image':
                # Process PIL image
                self.progress_updated.emit(30, "Processing image data...")
                self.msleep(100)
                
                if self._is_cancelled:
                    return
                
                self.progress_updated.emit(50, "Running OCR analysis...")
                self.msleep(100)
                
                result = self.ocr_engine.extract_text_from_pil_image(
                    self.image_data, 
                    preprocess=self.preprocess
                )
            
            else:
                raise ValueError(f"Unknown processing type: {self.processing_type}")
            
            if self._is_cancelled:
                return
            
            # Final progress update
            self.progress_updated.emit(90, "Finalizing results...")
            self.msleep(100)
            
            # Emit success signal
            self.extraction_completed.emit(result)
            
        except Exception as e:
            if not self._is_cancelled:
                self.extraction_failed.emit(str(e))


class PageThumbnailWidget(QWidget):
    """Individual page thumbnail widget"""
    
    def __init__(self, page_number: int, thumbnail: QPixmap, main_window):
        super().__init__()
        
        self.page_number = page_number
        self.main_window = main_window
        self.is_current = False
        
        self.setFixedSize(130, 180)
        self.setup_ui(thumbnail)
        
    def setup_ui(self, thumbnail: QPixmap):
        """Setup thumbnail UI"""
        
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Thumbnail image
        self.image_label = QLabel()
        self.image_label.setPixmap(thumbnail.scaled(120, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #ccc; background: white;")
        
        # Page number label
        self.page_label = QLabel(f"Page {self.page_number + 1}")
        self.page_label.setAlignment(Qt.AlignCenter)
        self.page_label.setObjectName("thumbnailPageLabel")
        
        layout.addWidget(self.image_label)
        layout.addWidget(self.page_label)
        
        # Make clickable
        self.setCursor(Qt.PointingHandCursor)
        
    def mousePressEvent(self, event):
        """Handle thumbnail click"""
        if event.button() == Qt.LeftButton:
            self.main_window.jump_to_page(self.page_number)
            
    def set_current(self, is_current: bool):
        """Beautiful highlight for current page"""
        self.is_current = is_current
        
        if is_current:
            self.setStyleSheet("""
                PageThumbnailWidget {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                        stop: 0 #007bff, stop: 1 #0056b3);
                    border: 3px solid #ffffff;
                    border-radius: 8px;
                    padding: 5px;
                }
                QLabel {
                    color: white;
                    font-weight: bold;
                }
            """)
        else:
            self.setStyleSheet("""
                PageThumbnailWidget {
                    background: white;
                    border: 2px solid #e9ecef;
                    border-radius: 6px;
                    padding: 3px;
                }
                PageThumbnailWidget:hover {
                    background: #f8f9fa;
                    border-color: #007bff;
                }
                QLabel {
                    color: #495057;
                }
            """)


class ImageViewer(QScrollArea):
    """Enhanced image display widget with proper scaling, centering, and drag support"""
    
    def __init__(self):
        super().__init__()
        
        # Setup scroll area properties
        self.setMinimumSize(600, 400)
        self.setAlignment(Qt.AlignCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setWidgetResizable(False)  # Changed to False for better control
        
        # Create the actual image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(False)
        self.image_label.setMinimumSize(1, 1)
        
        # Set the image label as the widget for scroll area
        self.setWidget(self.image_label)
        
        # Setup initial appearance
        self.setup_empty_state()
        
        # Image properties
        self.original_pixmap = None
        self.current_zoom = 100
        self.rotation_angle = 0
        self.has_image = False
        
        # Drag properties
        self.dragging = False
        self.last_pan_point = QPoint()
        
        # Enable mouse tracking for smooth dragging
        self.setMouseTracking(True)
        self.image_label.setMouseTracking(True)
    
    def setup_empty_state(self):
        """Setup clean empty state appearance"""
        self.setStyleSheet("""
            QScrollArea {
                border: 2px dashed #ced4da;
                border-radius: 8px;
                background-color: #ffffff;
            }
            QLabel {
                color: #6c757d;
                font-size: 16px;
                font-weight: 500;
                background-color: transparent;
                border: none;
            }
        """)
        
        self.image_label.setText("""
        📄 Welcome to EyeShot AI
        
        Click "📁 Open File" to get started
        
        Supports: PNG, JPEG, PDF
        
        🖱️ Images will be draggable and zoomable
        🔍 Use Ctrl+Mouse Wheel to zoom
        """)
        self.has_image = False
    
    def set_image(self, pixmap: QPixmap):
        """Set image to display with proper scaling and centering"""
        self.original_pixmap = pixmap
        self.has_image = True
        self.current_zoom = 100
        self.rotation_angle = 0
        
        # Update display
        self.update_display()
        
        # Change styling when image is loaded
        self.setStyleSheet("""
            QScrollArea {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background-color: #f8f9fa;
            }
            QLabel {
                background-color: #ffffff;
                border: 1px solid #e9ecef;
                border-radius: 4px;
            }
        """)
        
        # Set cursor for dragging
        self.setCursor(Qt.OpenHandCursor)
    
    def update_display(self):
        """Update the displayed image with current zoom and rotation"""
        if not self.original_pixmap or not self.has_image:
            return
        
        # Start with original pixmap
        display_pixmap = self.original_pixmap
        
        # Apply zoom first
        if self.current_zoom != 100:
            # Calculate new size based on zoom
            new_width = int(self.original_pixmap.width() * (self.current_zoom / 100.0))
            new_height = int(self.original_pixmap.height() * (self.current_zoom / 100.0))
            display_pixmap = display_pixmap.scaled(
                new_width, new_height,
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
        
        # Apply rotation
        if self.rotation_angle != 0:
            transform = QTransform()
            transform.rotate(self.rotation_angle)
            display_pixmap = display_pixmap.transformed(transform, Qt.SmoothTransformation)
        
        # Set the pixmap and adjust label size
        self.image_label.setPixmap(display_pixmap)
        self.image_label.resize(display_pixmap.size())
    
    def set_zoom(self, zoom_percent: int):
        """Set zoom level and maintain center position"""
        if not self.has_image:
            return
        
        # Store current scroll position as ratio
        h_bar = self.horizontalScrollBar()
        v_bar = self.verticalScrollBar()
        
        # Calculate current center as ratio
        h_ratio = 0.5
        v_ratio = 0.5
        
        if h_bar.maximum() > 0:
            h_ratio = (h_bar.value() + self.viewport().width() / 2) / (h_bar.maximum() + self.viewport().width())
        if v_bar.maximum() > 0:
            v_ratio = (v_bar.value() + self.viewport().height() / 2) / (v_bar.maximum() + self.viewport().height())
        
        # Update zoom
        self.current_zoom = zoom_percent
        self.update_display()
        
        # Restore center position after a short delay
        QTimer.singleShot(10, lambda: self.restore_zoom_center(h_ratio, v_ratio))
    
    def restore_zoom_center(self, h_ratio: float, v_ratio: float):
        """Restore the center position after zoom change"""
        h_bar = self.horizontalScrollBar()
        v_bar = self.verticalScrollBar()
        
        # Calculate new scroll positions
        new_h_value = int(h_ratio * (h_bar.maximum() + self.viewport().width()) - self.viewport().width() / 2)
        new_v_value = int(v_ratio * (v_bar.maximum() + self.viewport().height()) - self.viewport().height() / 2)
        
        # Apply new positions
        h_bar.setValue(max(0, min(h_bar.maximum(), new_h_value)))
        v_bar.setValue(max(0, min(v_bar.maximum(), new_v_value)))
    
    def rotate_image(self):
        """Rotate image 90 degrees clockwise"""
        if self.has_image:
            self.rotation_angle = (self.rotation_angle + 90) % 360
            self.update_display()
    
    def reset_view(self):
        """Reset zoom and rotation, and center the image"""
        self.current_zoom = 100
        self.rotation_angle = 0
        self.update_display()
        
        # Center the image after reset
        QTimer.singleShot(50, self.center_view)
    
    def center_view(self):
        """Center the view on the image"""
        if not self.has_image:
            return
        
        # Center both scroll bars
        h_scrollbar = self.horizontalScrollBar()
        v_scrollbar = self.verticalScrollBar()
        
        h_scrollbar.setValue(h_scrollbar.maximum() // 2)
        v_scrollbar.setValue(v_scrollbar.maximum() // 2)
    
    def fit_to_window(self):
        """Fit image to window size"""
        if not self.has_image:
            return 100
        
        # Calculate zoom to fit
        viewport_size = self.viewport().size()
        image_size = self.original_pixmap.size()
        
        # Calculate scale factors
        scale_x = viewport_size.width() / image_size.width()
        scale_y = viewport_size.height() / image_size.height()
        
        # Use the smaller scale to fit completely
        scale = min(scale_x, scale_y) * 0.95  # 95% to leave some margin
        
        new_zoom = int(scale * 100)
        new_zoom = max(25, min(300, new_zoom))  # Clamp between 25% and 300%
        
        self.set_zoom(new_zoom)
        return new_zoom
    
    # Fixed Mouse Events for Dragging
    def mousePressEvent(self, event):
        """Handle mouse press for dragging"""
        if event.button() == Qt.LeftButton and self.has_image:
            self.dragging = True
            self.last_pan_point = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for dragging"""
        if self.dragging and self.has_image and (event.buttons() & Qt.LeftButton):
            # Calculate the movement delta
            delta = event.pos() - self.last_pan_point
            self.last_pan_point = event.pos()
            
            # Get current scroll bar values
            h_bar = self.horizontalScrollBar()
            v_bar = self.verticalScrollBar()
            
            # Update scroll positions (invert delta for natural movement)
            h_bar.setValue(h_bar.value() - delta.x())
            v_bar.setValue(v_bar.value() - delta.y())
            
            event.accept()
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if event.button() == Qt.LeftButton and self.dragging:
            self.dragging = False
            if self.has_image:
                self.setCursor(Qt.OpenHandCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)
    
    def wheelEvent(self, event):
        """Handle wheel event for zooming"""
        if self.has_image and event.modifiers() == Qt.ControlModifier:
            # Zoom with mouse wheel while holding Ctrl
            angle_delta = event.angleDelta().y()
            zoom_in = angle_delta > 0
            
            # Calculate new zoom
            zoom_step = 10
            if zoom_in:
                new_zoom = min(300, self.current_zoom + zoom_step)
            else:
                new_zoom = max(25, self.current_zoom - zoom_step)
            
            # Apply zoom
            self.set_zoom(new_zoom)
            
            # Update main window zoom slider if available
            main_window = self.window()
            if hasattr(main_window, 'zoom_slider'):
                main_window.zoom_slider.setValue(new_zoom)
                main_window.zoom_value_label.setText(f"{new_zoom}%")
            
            event.accept()
        else:
            # Normal scroll behavior
            super().wheelEvent(event)
    
    def enterEvent(self, event):
        """Handle mouse enter"""
        if self.has_image:
            self.setCursor(Qt.OpenHandCursor)
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """Handle mouse leave"""
        if not self.dragging and self.has_image:
            self.setCursor(Qt.ArrowCursor)
        super().leaveEvent(event)
    
    def resizeEvent(self, event):
        """Handle resize event"""
        super().resizeEvent(event)
        # No additional processing needed for draggable version


class MainWindow(QMainWindow):
    """Main application window for EyeShot AI with responsive design"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize file handler, OCR engine, and PDF converter
        self.file_handler = FileHandler()
        self.ocr_engine = OCREngine()
        self.pdf_converter = PDFConverter()
        
        # Test OCR installation on startup
        self.test_ocr_on_startup()
        
        # ✨ Beautiful Window Properties ✨
        self.setWindowTitle("✨ EyeShot AI - Smart OCR Text Extraction v1.2 ✨")
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        self.setMinimumSize(900, 700)  # Reduced minimum size for better compatibility
        self.resize(1400, 900)
        
        # Set window opacity for modern look
        self.setWindowOpacity(0.98)
        
        # Center window on screen
        self.center_window()
        
        # Setup UI components
        self.setup_ui()
        self.setup_styles()
        
        # Connect signals
        self.connect_signals()
        
        # Status
        self.current_file = None
        self.current_file_info = None
        
        # PDF-specific state
        self.current_page = 0
        self.total_pages = 0
        self.pdf_images = {}  # Cache converted pages
        self.thumbnail_widgets = {}  # Cache thumbnail widgets
        
        # OCR worker for background processing
        self.ocr_worker = None
        
        self.statusBar().showMessage("Ready - Open an image or PDF to get started")

    def show_message(self, title, message):
        """Show a message dialog"""
        QMessageBox.information(self, title, message)        
    
    def test_ocr_on_startup(self):
        """Test OCR installation when app starts"""
        
        is_working, message = self.ocr_engine.test_ocr_installation()
        
        if not is_working:
            QMessageBox.warning(
                self, 
                "OCR Engine Warning", 
                f"Tesseract OCR engine issue:\n\n{message}\n\nText extraction may not work properly."
            )
            self.statusBar().showMessage("OCR engine warning - check Tesseract installation")
        else:
            print(f"✅ {message}")  # Success message to console
    
    def center_window(self):
        """Center the window on the screen"""
        screen = QApplication.desktop().screenGeometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2,
            (screen.height() - size.height()) // 2
        )
    
    def setup_ui(self):
        """Setup the user interface"""
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main horizontal layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Left side - Thumbnails panel (initially hidden)
        self.thumbnails_panel = self.create_thumbnails_panel()
        self.thumbnails_panel.hide()
        
        # Center - Image area
        self.image_area = self.create_image_area()
        
        # Right side - Text panel (initially hidden)
        self.text_panel = self.create_text_panel()
        self.text_panel.hide()
        
        # Add to main layout
        main_layout.addWidget(self.thumbnails_panel)
        main_layout.addWidget(self.image_area, 1)  # Takes remaining space
        main_layout.addWidget(self.text_panel)
        
        # Create menu bar and status bar
        self.create_menu_bar()
        self.create_status_bar()
    
    def create_thumbnails_panel(self):
        """Create page thumbnails sidebar"""
        
        panel = QWidget()
        panel.setFixedWidth(150)
        panel.setObjectName("thumbnailsPanel")
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Header
        header_label = QLabel("Pages")
        header_label.setObjectName("thumbnailsHeader")
        layout.addWidget(header_label)
        
        # Thumbnails scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Thumbnails container
        self.thumbnails_container = QWidget()
        self.thumbnails_layout = QVBoxLayout(self.thumbnails_container)
        self.thumbnails_layout.setSpacing(5)
        self.thumbnails_layout.setContentsMargins(0, 0, 0, 0)
        self.thumbnails_layout.addStretch()
        
        scroll_area.setWidget(self.thumbnails_container)
        layout.addWidget(scroll_area)
        
        return panel
    
    def create_image_area(self):
        """Create the center image viewing area with enhanced viewer"""
        
        area_widget = QWidget()
        layout = QVBoxLayout(area_widget)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(10)
        
        # Top toolbar with reliable implementation
        toolbar = self.create_toolbar()
        layout.addWidget(toolbar)
        
        # Enhanced Image viewer
        self.image_viewer = ImageViewer()
        layout.addWidget(self.image_viewer, 1)  # Takes most space
        
        # Bottom controls with Text button
        bottom_controls = self.create_bottom_controls()
        layout.addWidget(bottom_controls)
        
        return area_widget

    def create_toolbar(self):
        """Create a reliable toolbar using QToolBar instead of custom widgets"""
        
        # Create a standard QToolBar
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toolbar.setStyleSheet("""
            QToolBar {
                spacing: 2px;
                padding: 2px;
                border: none;
                background-color: #f8f9fa;
            }
            QToolButton {
                min-width: 30px;
                min-height: 30px;
                padding: 4px;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background-color: white;
            }
            QToolButton:hover {
                background-color: #e9ecef;
            }
            QToolButton:pressed {
                background-color: #dee2e6;
            }
            QToolButton:disabled {
                color: #6c757d;
                background-color: #f8f9fa;
            }
            QToolButton:checked {
                background-color: #007bff;
                color: white;
            }
        """)
        
        # 1. File Operations
        self.open_action = toolbar.addAction("📁 Open")
        self.open_action.triggered.connect(self.open_file)
        toolbar.addSeparator()
        
        # 2. Navigation Group
        self.first_page_action = toolbar.addAction("⏮")
        self.first_page_action.triggered.connect(self.first_page)
        self.first_page_action.setEnabled(False)
        
        self.prev_action = toolbar.addAction("◀")
        self.prev_action.triggered.connect(self.previous_page)
        self.prev_action.setEnabled(False)
        
        self.next_action = toolbar.addAction("▶")
        self.next_action.triggered.connect(self.next_page)
        self.next_action.setEnabled(False)
        
        self.last_page_action = toolbar.addAction("⏭")
        self.last_page_action.triggered.connect(self.last_page)
        self.last_page_action.setEnabled(False)
        toolbar.addSeparator()
        
        # 3. Page Info - Using a label in a widget action
        self.page_label = QLabel("Page: -")
        self.page_label.setMinimumWidth(100)
        self.page_label.setAlignment(Qt.AlignCenter)
        self.page_label.setStyleSheet("""
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 4px 8px;
            font-weight: bold;
            font-size: 11px;
            color: #495057;
        """)
        toolbar.addWidget(self.page_label)
        toolbar.addSeparator()
        
        # 4. Zoom Group - Using widget actions for complex controls
        zoom_widget = QWidget()
        zoom_layout = QHBoxLayout(zoom_widget)
        zoom_layout.setContentsMargins(2, 2, 2, 2)
        zoom_layout.setSpacing(4)
        
        zoom_label = QLabel("Zoom:")
        zoom_label.setStyleSheet("font-size: 11px; font-weight: bold;")
        
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(25, 300)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setFixedWidth(80)
        self.zoom_slider.valueChanged.connect(self.zoom_changed)
        
        self.zoom_value_label = QLabel("100%")
        self.zoom_value_label.setFixedWidth(45)
        self.zoom_value_label.setAlignment(Qt.AlignCenter)
        self.zoom_value_label.setStyleSheet("""
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 2px;
            font-weight: bold;
            font-size: 11px;
        """)
        
        zoom_layout.addWidget(zoom_label)
        zoom_layout.addWidget(self.zoom_slider)
        zoom_layout.addWidget(self.zoom_value_label)
        
        toolbar.addWidget(zoom_widget)
        toolbar.addSeparator()
        
        # 5. Action Group
        self.thumbnails_action = QAction("🖼", self)
        self.thumbnails_action.setCheckable(True)
        self.thumbnails_action.triggered.connect(self.toggle_thumbnails)
        self.thumbnails_action.setEnabled(False)
        toolbar.addAction(self.thumbnails_action)
        
        self.fit_action = QAction("📐", self)
        self.fit_action.triggered.connect(self.fit_to_window)
        self.fit_action.setEnabled(False)
        toolbar.addAction(self.fit_action)
        
        self.rotate_action = QAction("↻", self)
        self.rotate_action.triggered.connect(self.rotate_image)
        self.rotate_action.setEnabled(False)
        toolbar.addAction(self.rotate_action)
        
        # Return the toolbar
        return toolbar
    
    def create_bottom_controls(self):
        """Create bottom area with Extract Text button"""
        
        bottom_widget = QWidget()
        bottom_widget.setFixedHeight(80)
        layout = QHBoxLayout(bottom_widget)
        layout.setContentsMargins(20, 15, 20, 15)
        
        # Center the Text button
        layout.addStretch()
        
        # Main Text extraction button
        self.text_btn = QPushButton("📝 EXTRACT TEXT")
        self.text_btn.setFixedSize(180, 50)
        self.text_btn.setEnabled(False)
        self.text_btn.setObjectName("textButton")
        self.text_btn.setToolTip("Extract text from current image/page (Ctrl+E)")
        
        layout.addWidget(self.text_btn)
        layout.addStretch()
        
        return bottom_widget
    
    def create_text_panel(self):
        """Create right side text extraction panel with reliable button display"""
        
        panel = QWidget()
        panel.setFixedWidth(400)
        
        # Use QVBoxLayout with proper size constraints
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 0, 0, 0)
        layout.setSpacing(15)
        
        # Panel header
        header_layout = QHBoxLayout()
        
        header_label = QLabel("Text Extraction Results")
        header_label.setObjectName("panelHeader")
        
        self.close_panel_btn = QPushButton("✕")
        self.close_panel_btn.setFixedSize(25, 25)
        self.close_panel_btn.setToolTip("Close text panel (Esc)")
        
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        header_layout.addWidget(self.close_panel_btn)
        
        layout.addLayout(header_layout)
        
        # Progress bar (initially hidden)
        self.extraction_progress = QProgressBar()
        self.extraction_progress.setMinimum(0)
        self.extraction_progress.setMaximum(100)
        self.extraction_progress.setVisible(False)
        layout.addWidget(self.extraction_progress)
        
        # Text display area - Set to stretch but with reasonable constraints
        self.text_display = QTextEdit()
        self.text_display.setPlaceholderText("Extracted text will appear here...")
        self.text_display.setMinimumHeight(200)  # Reduced minimum height
        
        # Add with stretch factor of 1 (can grow but won't take all space)
        layout.addWidget(self.text_display, 1)
        
        # Action buttons section with FIXED HEIGHT to ensure visibility
        buttons_section = QWidget()
        buttons_section.setMinimumHeight(120)  # Ensure enough room for buttons
        button_layout = QVBoxLayout(buttons_section)
        button_layout.setContentsMargins(0, 0, 0, 0)
        
        # Top row - Copy and Clear buttons
        top_row = QHBoxLayout()
        
        self.copy_btn = QPushButton("📋 Copy All")
        self.copy_btn.setFixedHeight(35)
        self.copy_btn.setToolTip("Copy text to clipboard (Ctrl+C)")
        
        self.clear_btn = QPushButton("🗑️ Clear")
        self.clear_btn.setFixedHeight(35)
        self.clear_btn.setToolTip("Clear extracted text")
        
        top_row.addWidget(self.copy_btn)
        top_row.addWidget(self.clear_btn)
        button_layout.addLayout(top_row)
        
        # Export button on its own row
        self.export_btn = QPushButton("💾 Export As...")
        self.export_btn.setFixedHeight(35)
        self.export_btn.setToolTip("Export text to file (Ctrl+S)")
        button_layout.addWidget(self.export_btn)
        
        # Add buttons section with fixed height (won't collapse)
        layout.addWidget(buttons_section, 0)  # 0 stretch factor - fixed height
        
        # Info label
        self.info_label = QLabel("Ready for text extraction")
        self.info_label.setObjectName("infoLabel")
        self.info_label.setWordWrap(True)
        self.info_label.setMinimumHeight(40)  # Ensure visible height
        
        layout.addWidget(self.info_label, 0)  # 0 stretch factor - won't grow
        
        # Add final stretch to push everything up
        layout.addStretch(0)
        
        return panel

    def create_menu_bar(self):
        """Create application menu bar"""
        
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        open_action = QAction('&Open File...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        export_action = QAction('&Export Text...', self)
        export_action.setShortcut('Ctrl+S')
        export_action.triggered.connect(self.export_text)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu('&View')
        
        thumbnails_action = QAction('&Thumbnails', self)
        thumbnails_action.setShortcut('Ctrl+T')
        thumbnails_action.triggered.connect(self.toggle_thumbnails)
        view_menu.addAction(thumbnails_action)
        
        fit_action = QAction('&Fit to Window', self)
        fit_action.setShortcut('Ctrl+F')
        fit_action.triggered.connect(self.fit_to_window)
        view_menu.addAction(fit_action)
        
        view_menu.addSeparator()
        
        zoom_in_action = QAction('Zoom &In', self)
        zoom_in_action.setShortcut('Ctrl+=')
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction('Zoom &Out', self)
        zoom_out_action.setShortcut('Ctrl+-')
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('&Tools')
        
        extract_action = QAction('&Extract Text', self)
        extract_action.setShortcut('Ctrl+E')
        extract_action.triggered.connect(self.extract_text)
        tools_menu.addAction(extract_action)
        
        rotate_action = QAction('&Rotate Image', self)
        rotate_action.setShortcut('Ctrl+R')
        rotate_action.triggered.connect(self.rotate_image)
        tools_menu.addAction(rotate_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        about_action = QAction('&About EyeShot AI', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_status_bar(self):
        """Create status bar"""
        self.statusBar().setSizeGripEnabled(True)
    
    def connect_signals(self):
        """Connect signals - simple and reliable"""
        
        # Store text_btn connections separately since we reassign it later
        self.text_btn.clicked.connect(self.extract_text)
            
        # Text panel connections
        self.close_panel_btn.clicked.connect(self.hide_text_panel)
        self.copy_btn.clicked.connect(self.copy_text)
        self.clear_btn.clicked.connect(self.clear_text)
        
        # Add this line if not already present
        self.export_btn.clicked.connect(self.export_text)


    def setup_styles(self):
        """Advanced styling with proper size constraints"""

        self.text_display.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ced4da;
                border-radius: 6px;
                padding: 15px;
                background-color: white;
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 12px;
                line-height: 1.6;
                color: #212529;
            }
            
            QTextEdit:focus {
                border-color: #007bff;
                box-shadow: 0 0 5px rgba(0, 123, 255, 0.5);
            }
        """)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                font-size: 11px;
                font-weight: 500;
                color: #495057;
                min-width: 30px;
                min-height: 30px;
                padding: 2px 4px;
            }
            
            QPushButton:hover {
                background-color: #e9ecef;
                border-color: #adb5bd;
            }
            
            QPushButton:pressed {
                background-color: #dee2e6;
            }
            
            QPushButton:disabled {
                background-color: #f8f9fa;
                color: #6c757d;
                border-color: #dee2e6;
            }
            
            QPushButton:checked {
                background-color: #007bff;
                color: white;
                border-color: #007bff;
            }
            
            /* Slider styling */
            QSlider::groove:horizontal {
                border: 1px solid #dee2e6;
                background: #f8f9fa;
                height: 6px;
                border-radius: 3px;
            }
            
            QSlider::handle:horizontal {
                background: #007bff;
                border: 1px solid #0056b3;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            
            /* Text Extraction Button */
            QPushButton#textButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #28a745, stop: 1 #1e7e34);
                color: white;
                border: none;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px 20px;
                min-width: 160px;
                min-height: 40px;
            }
            
            QPushButton#textButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #218838, stop: 1 #155724);
            }
            
            QPushButton#textButton:disabled {
                background: #6c757d;
                color: #ffffff;
            }
            
            /* Panel Headers */
            QLabel#panelHeader {
                font-size: 16px;
                font-weight: bold;
                color: #ffffff;
                background-color: #007bff;
                padding: 10px;
                border-radius: 6px;
            }
            
            QLabel#thumbnailsHeader {
                font-size: 14px;
                font-weight: bold;
                color: #ffffff;
                background-color: #6f42c1;
                padding: 8px;
                border-radius: 4px;
            }
            
            /* Text Areas */
            QTextEdit {
                border: 1px solid #ced4da;
                border-radius: 6px;
                padding: 10px;
                background-color: white;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                line-height: 1.4;
            }
            
            QTextEdit:focus {
                border-color: #007bff;
            }
            
            /* Progress Bar */
            QProgressBar {
                border: 1px solid #ced4da;
                border-radius: 6px;
                text-align: center;
                background-color: #f8f9fa;
                color: #495057;
                font-size: 10px;
                font-weight: bold;
            }
            
            QProgressBar::chunk {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #007bff, stop: 1 #0056b3);
                border-radius: 5px;
                margin: 1px;
            }
        """)
    
    # File handling methods
    def open_file(self):
        """Open file dialog and load image/PDF"""
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image or PDF",
            "",
            "Supported Files (*.png *.jpg *.jpeg *.pdf);;PNG Images (*.png);;JPEG Images (*.jpg *.jpeg);;PDF Documents (*.pdf);;All Files (*)"
        )
        
        if file_path:
            self.load_file(file_path)
    
    def load_file(self, file_path: str):
        """Load a file"""
        
        # Validate file
        is_valid, message = self.file_handler.validate_file(file_path)
        
        if not is_valid:
            QMessageBox.warning(self, "Invalid File", f"Cannot open file:\n{message}")
            return False
        
        # Update status
        self.statusBar().showMessage(f"Loading {os.path.basename(file_path)}...")
        
        # Get file info
        self.current_file_info = self.file_handler.get_file_info(file_path)
        file_type = self.current_file_info['type']
        
        try:
            if file_type == 'image':
                success = self.load_image_file(file_path)
            elif file_type == 'pdf':
                success = self.load_pdf_file(file_path)
            else:
                success = False
            
            if success:
                self.current_file = file_path
                self.update_ui_after_file_load()
                self.statusBar().showMessage(f"Loaded: {self.current_file_info['name']} ({self.current_file_info['size_mb']} MB)")
            else:
                self.statusBar().showMessage("Failed to load file")
                
            return success
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading file:\n{str(e)}")
            self.statusBar().showMessage("Error loading file")
            return False
    
    def load_image_file(self, image_path: str) -> bool:
        """Load image file into viewer with auto-fit"""
        
        pixmap = self.file_handler.load_image_as_pixmap(image_path)
        
        if pixmap:
            self.image_viewer.set_image(pixmap)
            
            # ✨ AUTO-FIT IMAGE FILES ✨
            QTimer.singleShot(100, self.auto_fit_current_page)
            
            return True
        else:
            return False
    
    def load_pdf_file(self, pdf_path: str) -> bool:
        """Load PDF file with thumbnails and auto-fit first page"""
        
        try:
            result = self.pdf_converter.load_pdf(pdf_path)
            
            if not result['success']:
                QMessageBox.warning(self, "PDF Error", f"Cannot load PDF:\n{result['error']}")
                return False
            
            # Store PDF info
            self.total_pages = result['page_count']
            self.current_page = 0
            self.pdf_images = {}
            self.thumbnail_widgets = {}
            
            print(f"📄 Loaded PDF: {self.total_pages} pages")
            
            # Convert first page to image
            first_page_image = self.pdf_converter.convert_page_to_image(0)
            
            if first_page_image:
                # Display first page
                qimage = self._pil_to_qimage(first_page_image)
                pixmap = QPixmap.fromImage(qimage)
                
                self.image_viewer.set_image(pixmap)
                
                # ✨ AUTO-FIT FIRST PAGE ✨
                QTimer.singleShot(100, self.auto_fit_current_page)
                
                # Cache the converted page
                self.pdf_images[0] = first_page_image
                
                # Update navigation UI
                self.update_pdf_navigation()
                
                # Generate thumbnails in background
                QTimer.singleShot(100, self.generate_thumbnails_async)
                
                return True
            else:
                QMessageBox.warning(self, "PDF Error", "Failed to convert PDF page to image")
                return False
                
        except Exception as e:
            QMessageBox.critical(self, "PDF Error", f"Error loading PDF:\n{str(e)}")
            return False
    
    def _pil_to_qimage(self, pil_image):
        """Convert PIL Image to QImage with robust error handling"""
        
        try:
            if pil_image is None:
                return None
                
            width, height = pil_image.size
            
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            img_data = pil_image.tobytes('raw', 'RGB')
            bytes_per_line = width * 3
            qimage = QImage(img_data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            if qimage.isNull():
                return self._pil_to_qimage_fallback(pil_image)
            
            return qimage
            
        except Exception as e:
            print(f"Error in _pil_to_qimage: {e}")
            return self._pil_to_qimage_fallback(pil_image)

    def _pil_to_qimage_fallback(self, pil_image):
        """Fallback method using temporary file approach"""
        
        try:
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
            
            pil_image.save(temp_path, 'PNG', optimize=True)
            qimage = QImage(temp_path)
            
            try:
                os.unlink(temp_path)
            except:
                pass
            
            return qimage if not qimage.isNull() else None
            
        except Exception as e:
            print(f"Error in fallback method: {e}")
            return None
    
    def update_pdf_navigation(self):
        """Update navigation buttons for PDF view"""
        
        if self.total_pages > 0:
            # Update page display
            self.page_label.setText(f"Page: {self.current_page + 1} of {self.total_pages}")
            
            # Update action states
            self.first_page_action.setEnabled(self.current_page > 0)
            self.prev_action.setEnabled(self.current_page > 0)
            self.next_action.setEnabled(self.current_page < self.total_pages - 1)
            self.last_page_action.setEnabled(self.current_page < self.total_pages - 1)
            self.thumbnails_action.setEnabled(True)
        else:
            # Single page or no file
            self.page_label.setText("Page: -")
            self.first_page_action.setEnabled(False)
            self.prev_action.setEnabled(False)
            self.next_action.setEnabled(False)
            self.last_page_action.setEnabled(False)
            self.thumbnails_action.setEnabled(False)
            self.thumbnails_action.setChecked(False)
            self.thumbnails_panel.hide()
    
    def update_ui_after_file_load(self):
        """Update UI elements after successfully loading a file"""
        
        if not self.current_file_info:
            return
        
        file_type = self.current_file_info['type']
        
        # Enable text extraction button
        self.text_btn.setEnabled(True)
        
        # Enable actions
        self.rotate_action.setEnabled(file_type == 'image')
        self.fit_action.setEnabled(True)
        
        # Update page navigation
        if file_type == 'pdf':
            self.update_pdf_navigation()
        else:
            self.first_page_action.setEnabled(False)
            self.prev_action.setEnabled(False)
            self.next_action.setEnabled(False)
            self.last_page_action.setEnabled(False)
            self.thumbnails_action.setEnabled(False)
            self.thumbnails_panel.hide()
        
        # Hide text panel if open
        self.hide_text_panel()

    def auto_fit_current_page(self):
        """Auto-fit the current page (works for both images and PDF pages)"""
        
        if self.image_viewer.has_image:
            new_zoom = self.image_viewer.fit_to_window()
            self.zoom_slider.setValue(new_zoom)
            self.zoom_value_label.setText(f"{new_zoom}%")
            
            # Show which page was fitted
            if hasattr(self, 'total_pages') and self.total_pages > 1:
                self.statusBar().showMessage(f"Page {self.current_page + 1}/{self.total_pages} - Auto-fitted to {new_zoom}%")
            else:
                self.statusBar().showMessage(f"Auto-fitted to {new_zoom}%")
    
    # PDF Navigation Methods
    def first_page(self):
        """Jump to first page with auto-fit"""
        if self.total_pages > 0:
            self.jump_to_page(0)

    def last_page(self):
        """Jump to last page with auto-fit"""
        if self.total_pages > 0:
            self.jump_to_page(self.total_pages - 1)

    def previous_page(self):
        """Navigate to previous page with auto-fit"""
        if hasattr(self, 'total_pages') and self.current_page > 0:
            self.jump_to_page(self.current_page - 1)

    def next_page(self):
        """Navigate to next page with auto-fit"""
        if hasattr(self, 'total_pages') and self.current_page < self.total_pages - 1:
            self.jump_to_page(self.current_page + 1)

    def jump_to_page(self, page_number: int):
        """Jump to specific page with auto-fit"""
        
        if not (0 <= page_number < self.total_pages):
            return
        
        old_page = self.current_page
        self.current_page = page_number
        
        # Load the page
        success = self.load_current_pdf_page()
        
        if success:
            # Update thumbnail highlights
            self.update_thumbnail_highlights(old_page, self.current_page)
            
            # ✨ AUTO-FIT EVERY PAGE SWITCH ✨
            QTimer.singleShot(100, self.auto_fit_current_page)
        else:
            # Revert on failure
            self.current_page = old_page

    def load_current_pdf_page(self):
        """Load current PDF page with caching"""
        
        if not (self.current_file_info and self.current_file_info['type'] == 'pdf'):
            return False
        
        try:
            # Check if page is cached
            if self.current_page in self.pdf_images:
                page_image = self.pdf_images[self.current_page]
            else:
                # Convert page to image
                page_image = self.pdf_converter.convert_page_to_image(self.current_page)
                if page_image:
                    # Cache the page (limit cache size)
                    if len(self.pdf_images) > 10:
                        oldest_page = min(self.pdf_images.keys())
                        del self.pdf_images[oldest_page]
                    
                    self.pdf_images[self.current_page] = page_image
            
            if page_image:
                # Convert PIL Image to QPixmap for display
                qimage = self._pil_to_qimage(page_image)
                if qimage:
                    pixmap = QPixmap.fromImage(qimage)
                    
                    # Display in image viewer
                    self.image_viewer.set_image(pixmap)
                    
                    # Update page navigation
                    self.update_pdf_navigation()
                    
                    # Update status
                    self.statusBar().showMessage(f"Showing page {self.current_page + 1} of {self.total_pages}")
                    return True
                
            QMessageBox.warning(self, "PDF Error", f"Failed to load page {self.current_page + 1}")
            return False
            
        except Exception as e:
            QMessageBox.critical(self, "PDF Error", f"Error loading page {self.current_page + 1}: {str(e)}")
            return False

    # Thumbnail Management
    def generate_thumbnails_async(self):
        """Generate page thumbnails in background"""
        
        if self.total_pages <= 1:
            return
        
        print(f"🖼️ Generating thumbnails for {self.total_pages} pages...")
        
        # Clear existing thumbnails
        self.clear_thumbnails()
        
        # Generate thumbnails for first few pages immediately
        max_initial_thumbs = min(5, self.total_pages)
        
        for page_num in range(max_initial_thumbs):
            self.create_page_thumbnail(page_num)
            QApplication.processEvents()
        
        # Generate remaining thumbnails with delay
        if self.total_pages > max_initial_thumbs:
            QTimer.singleShot(500, lambda: self.generate_remaining_thumbnails(max_initial_thumbs))

    def generate_remaining_thumbnails(self, start_page: int):
        """Generate remaining thumbnails with delay"""
        
        for page_num in range(start_page, min(start_page + 5, self.total_pages)):
            self.create_page_thumbnail(page_num)
            QApplication.processEvents()
        
        # Continue with next batch if needed
        if start_page + 5 < self.total_pages:
            QTimer.singleShot(200, lambda: self.generate_remaining_thumbnails(start_page + 5))

    def create_page_thumbnail(self, page_num: int):
        """Create thumbnail for specific page"""
        
        try:
            # Get or convert page image
            if page_num in self.pdf_images:
                page_image = self.pdf_images[page_num]
            else:
                page_image = self.pdf_converter.convert_page_to_image(page_num)
            
            if page_image:
                # Create thumbnail
                thumbnail_image = page_image.copy()
                thumbnail_image.thumbnail((120, 160), PILImage.Resampling.LANCZOS)
                
                # Convert to QPixmap
                qimage = self._pil_to_qimage(thumbnail_image)
                if qimage:
                    thumbnail_pixmap = QPixmap.fromImage(qimage)
                    
                    # Create thumbnail widget
                    thumbnail_widget = PageThumbnailWidget(page_num, thumbnail_pixmap, self)
                    
                    # Set current page highlight
                    thumbnail_widget.set_current(page_num == self.current_page)
                    
                    # Store reference
                    self.thumbnail_widgets[page_num] = thumbnail_widget
                    
                    # Insert at correct position
                    self.thumbnails_layout.insertWidget(page_num, thumbnail_widget)
                    
        except Exception as e:
            print(f"❌ Error creating thumbnail for page {page_num}: {e}")

    def clear_thumbnails(self):
        """Clear all thumbnails"""
        
        # Remove all thumbnail widgets
        while self.thumbnails_layout.count() > 1:
            child = self.thumbnails_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        self.thumbnail_widgets.clear()

    def update_thumbnail_highlights(self, old_page: int, new_page: int):
        """Update thumbnail highlights"""
        
        # Remove highlight from old page
        if old_page in self.thumbnail_widgets:
            self.thumbnail_widgets[old_page].set_current(False)
        
        # Add highlight to new page
        if new_page in self.thumbnail_widgets:
            self.thumbnail_widgets[new_page].set_current(True)

    def toggle_thumbnails(self):
        """Toggle thumbnails panel visibility"""
        if self.total_pages <= 1:
            return

        if self.thumbnails_panel.isVisible():
            self.thumbnails_panel.hide()
            self.thumbnails_action.setChecked(False)
        else:
            self.thumbnails_panel.show()
            self.thumbnails_action.setChecked(True)
            # Generate thumbnails if not already done
            if len(self.thumbnail_widgets) == 0:
                self.generate_thumbnails_async()

    # View Control Methods
    def fit_to_window(self):
        """Fit image to window"""
        if self.current_file and self.image_viewer.has_image:
            new_zoom = self.image_viewer.fit_to_window()
            self.zoom_slider.setValue(new_zoom)
            self.zoom_value_label.setText(f"{new_zoom}%")

    def zoom_in(self):
        """Zoom in on current image"""
        if self.current_file:
            current_zoom = self.zoom_slider.value()
            new_zoom = min(300, current_zoom + 25)
            self.zoom_slider.setValue(new_zoom)

    def zoom_out(self):
        """Zoom out on current image"""
        if self.current_file:
            current_zoom = self.zoom_slider.value()
            new_zoom = max(25, current_zoom - 25)
            self.zoom_slider.setValue(new_zoom)

    def rotate_image(self):
        """Rotate current image"""
        if self.current_file and self.current_file_info['type'] == 'image':
            self.image_viewer.rotate_image()

    def zoom_changed(self, value):
        """Handle zoom slider change"""
        self.zoom_value_label.setText(f"{value}%")
        if self.current_file:
            self.image_viewer.set_zoom(value)

    def get_file_type(self, file_path):
        """Get file type from file info or extension"""
        if self.current_file_info and 'type' in self.current_file_info:
            return self.current_file_info['type']
            
        # Fallback to extension check
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.pdf']:
            return 'pdf'
        elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            return 'image'
        else:
            return 'unknown'

    def extract_text(self):
        """Extract text from current document with enhanced structure preservation AND handler detection"""
        
        # Check if we have a document loaded
        if not self.current_file:
            self.show_message("No document loaded", "Please open a document first.")
            return
        
        # 🎯 ADD HANDLER DETECTION LOGGING
        print(f"\n🎯 USER ACTION: Extracting text through GUI with handler detection")
        print(f"📁 File: {os.path.basename(self.current_file)}")
        print(f"👤 User: Tigran0000")
        print(f"⏰ Time: 2025-06-20 15:45:53")
        
        # Get file type
        file_type = self.current_file_info['type']
        
        if file_type == 'pdf':
            # Use the enhanced PDF processing for better structure
            self.statusBar().showMessage("Extracting text with enhanced structure preservation...")
            
            # FIXED IMPORT: First try the module in 'converters' folder
            try:
                try:
                    from converters.pdf_helpers import create_enhanced_pdf_image
                except ImportError:
                    # Alternative: try direct import if not in converters folder
                    from pdf_helpers import create_enhanced_pdf_image
                
                result = create_enhanced_pdf_image(self.current_file, self.current_page)
                
                if result and result['success']:
                    # Set up OCR worker with enhanced image and structure hints
                    self.ocr_worker = OCRWorker(self.ocr_engine)
                    self.ocr_worker.setup_enhanced_pdf_extraction(
                        result['image'],
                        structure_hints=result['structure_hints'],
                        preprocess=True
                    )
                    
                    # Connect signals to ENHANCED completion handler
                    self.ocr_worker.extraction_completed.connect(self.on_extraction_completed_with_detection)
                    self.ocr_worker.progress_updated.connect(self.on_extraction_progress)
                    self.ocr_worker.extraction_failed.connect(self.on_extraction_failed)
                    
                    # Setup UI for extraction
                    self.setup_extraction_ui()
                    self.show_text_panel()
                    
                    # Start OCR process
                    self.ocr_worker.start()
                    return
            except Exception as e:
                print(f"Error setting up enhanced extraction: {e}")
                # Fall through to standard extraction
                
        # Standard extraction (for non-PDF or if enhanced failed)
        self._extract_text_standard_with_detection()

    def _extract_text_standard_with_detection(self):
        """Standard extraction with handler detection"""
        try:
            print(f"\n🔍 RUNNING HANDLER DETECTION FOR IMAGE FILE...")
            
            # 🎯 STEP 1: Run handler detection through file_handler
            detection_result = self.file_handler.process_with_ocr(self.current_file)
            
            # Store detection results for GUI display
            self.current_detection_result = detection_result
            
            # 🎯 STEP 2: Extract the recommended handler from detection
            recommended_handler = 'auto'  # default
            
            if detection_result.get('handler_detection'):
                detected_best = detection_result['handler_detection'].get('most_likely_handler', '')
                
                # Map detection result to OCR mode
                if detected_best == 'book_handler':
                    recommended_handler = 'book'
                    print(f"🎯 Using BookHandler as recommended!")
                elif detected_best == 'academic_handler':
                    recommended_handler = 'academic'
                    print(f"🎯 Using AcademicHandler as recommended!")
                else:
                    recommended_handler = 'auto'
                    print(f"🎯 Using auto mode as fallback")
            
            # 🎯 STEP 3: Get current image for OCR
            if self.current_file_info['type'] == 'pdf':
                # Get current page image
                if self.current_page in self.pdf_images:
                    current_image = self.pdf_images[self.current_page]
                else:
                    self.statusBar().showMessage("Preparing page for extraction...")
                    current_image = self.pdf_converter.convert_page_to_image(self.current_page)
                    
                if not current_image:
                    self.statusBar().showMessage("Failed to prepare page")
                    return
            else:
                # For image files, load the image
                from PIL import Image as PILImage
                current_image = PILImage.open(self.current_file)
            
            # 🎯 STEP 4: Run OCR with the RECOMMENDED HANDLER
            print(f"📝 Extracting text with mode: {recommended_handler}")
            
            # Show UI
            self.setup_extraction_ui()
            self.show_text_panel()
            
            # Run OCR with the recommended mode
            result = self.ocr_engine.extract_text(current_image, mode=recommended_handler, preprocess=True)
            
            # Add handler info to result
            result['selected_handler'] = f"{recommended_handler}_handler"
            
            # Process the result
            self.on_extraction_completed_with_detection(result)
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
            QMessageBox.critical(self, "Extraction Error", f"Error with handler detection: {str(e)}")
            self.statusBar().showMessage("Text extraction failed")

    def on_extraction_completed_with_detection(self, result):
        """Handle successful extraction completion with handler detection info"""
        
        self.reset_extraction_ui()
        file_type = self.current_file_info['type']
        
        # 🎯 MERGE HANDLER DETECTION RESULTS
        if hasattr(self, 'current_detection_result') and self.current_detection_result.get('handler_detection'):
            # Add handler detection info to the OCR result
            result['handler_detection'] = self.current_detection_result['handler_detection']
            result['selected_handler'] = self.current_detection_result.get('selected_handler')
            result['detected_handler'] = self.current_detection_result.get('detected_handler')
        
        if result['success'] and result['text'].strip():
            # Create HTML-formatted text for better paragraph display
            html_text = ""
            paragraphs = result['text'].split('\n\n')
            
            for paragraph in paragraphs:
                # Check if paragraph looks like a heading
                if paragraph.isupper() or (len(paragraph) < 60 and ("RULE" in paragraph.upper() or "CHAPTER" in paragraph.upper())):
                    html_text += f"<h3 style='text-align:center; margin-top:20px; margin-bottom:10px;'>{paragraph}</h3>"
                else:
                    html_text += f"<p style='text-indent:20px; margin-bottom:12px;'>{paragraph}</p>"
            
            # 🎯 ADD HANDLER DETECTION INFO TO GUI
            if 'handler_detection' in result:
                detection = result['handler_detection']
                selected = result.get('selected_handler', 'Unknown')
                detected = detection['most_likely_handler']
                confidence = detection['confidence']
                quality = detection['quality_metrics']['quality_level']
                
                # Add handler analysis section
                html_text += f"""
                <hr style='margin: 20px 0; border: 1px solid #ddd;'>
                <div style='background: #f8f9fa; padding: 15px; border-radius: 5px; font-size: 11px;'>
                    <h4 style='margin: 0 0 10px 0; color: #007bff;'>🔍 Handler Analysis</h4>
                    <p style='margin: 5px 0;'><strong>Selected Handler:</strong> {selected}</p>
                    <p style='margin: 5px 0;'><strong>Detected Best:</strong> {detected}</p>
                    <p style='margin: 5px 0;'><strong>Confidence:</strong> {confidence:.1f}/100</p>
                    <p style='margin: 5px 0;'><strong>Quality:</strong> {quality}</p>
                    <p style='margin: 5px 0;'><strong>OCR Errors:</strong> {detection['quality_metrics']['ocr_errors']}</p>
                    <p style='margin: 5px 0;'><strong>Recommendation:</strong> {detection['recommendation']}</p>
                </div>
                """
            
            # Set rich text with preserved formatting
            self.text_display.setHtml(html_text)
            
            # Use a nice, readable serif font like in books
            font = QFont("Georgia", 12)
            self.text_display.setFont(font)
            
            # Set paragraph spacing
            document = self.text_display.document()
            option = document.defaultTextOption()
            option.setAlignment(Qt.AlignJustify)  # Book-like justified text
            option.setWrapMode(QTextOption.WordWrap)
            document.setDefaultTextOption(option)
            
            # Apply additional styling to text display for book-like appearance
            self.text_display.setStyleSheet("""
                QTextEdit {
                    background-color: #fffef7;  /* Slight cream color like paper */
                    padding: 20px;
                    line-height: 1.5;
                    border: 1px solid #e0e0d1;
                }
            """)
            
            # Set cursor to beginning
            cursor = self.text_display.textCursor()
            cursor.movePosition(QTextCursor.Start)
            self.text_display.setTextCursor(cursor)
            
            # 🎯 UPDATE STATUS WITH HANDLER INFO
            confidence_text = f"{result['confidence']:.1f}%" if result['confidence'] > 0 else "N/A"
            page_info = f"Page {self.current_page + 1}/{self.total_pages}" if file_type == 'pdf' else ""
            
            handler_info = ""
            if 'handler_detection' in result:
                detection = result['handler_detection']
                if result.get('selected_handler', '').lower().replace('handler', '') == detection['most_likely_handler'].replace('_handler', ''):
                    handler_info = " | ✅ Optimal handler"
                else:
                    handler_info = f" | ⚠️ {detection['most_likely_handler']} recommended"
            
            self.info_label.setText(
                f"✅ Extracted {result['word_count']} words | "
                f"Confidence: {confidence_text} | "
                f"{page_info} | "
                f"Method: {result.get('best_method', 'standard')}"
                f"{handler_info}"
            )
            self.statusBar().showMessage(f"✅ Text extraction completed with handler analysis - {result['word_count']} words found")
        
        elif result['success'] and not result['text'].strip():
            # No text detected case
            self.text_display.setText("No text was detected on this page.\n\nTips:\n• Try a different page if this is a PDF\n• Ensure the page has clear, readable text\n• Check that text is not too small or blurry")
            self.info_label.setText("⚠️ No text detected")
            self.statusBar().showMessage("OCR completed - no text found")
        
        else:
            # Error case
            error_msg = result.get('error', 'Unknown error')
            self.text_display.setText(f"Text extraction failed.\n\nError: {error_msg}")
            self.info_label.setText("❌ Text extraction failed")
            self.statusBar().showMessage("OCR failed")
        
        # 🎯 CLEAN UP DETECTION RESULT
        if hasattr(self, 'current_detection_result'):
            delattr(self, 'current_detection_result')

    # You need to add this method
    def _extract_text_standard(self):
        """Standard extraction without structure enhancement"""
        try:
            if self.current_file_info['type'] == 'pdf':
                # Get current page image
                if self.current_page in self.pdf_images:
                    current_image = self.pdf_images[self.current_page]
                else:
                    self.statusBar().showMessage("Preparing page for extraction...")
                    current_image = self.pdf_converter.convert_page_to_image(self.current_page)
                    
                if current_image:
                    # Set up worker for PIL image
                    self.ocr_worker = OCRWorker(self.ocr_engine)
                    self.ocr_worker.setup_extraction('pil_image', pil_image=current_image)
                else:
                    self.statusBar().showMessage("Failed to prepare page")
                    return
            else:
                # Set up worker for image file
                self.ocr_worker = OCRWorker(self.ocr_engine)
                self.ocr_worker.setup_extraction('image_file', file_path=self.current_file)
            
            # Connect signals
            self.ocr_worker.extraction_completed.connect(self.on_extraction_completed)
            self.ocr_worker.progress_updated.connect(self.on_extraction_progress)
            self.ocr_worker.extraction_failed.connect(self.on_extraction_failed)
            
            # Setup UI for extraction
            self.setup_extraction_ui()
            self.show_text_panel()
            
            # Start OCR process
            self.ocr_worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Extraction Error", f"Error starting extraction: {str(e)}")
            self.statusBar().showMessage("Text extraction failed")

    def setup_extraction_ui(self):
        """Setup UI for extraction processing"""
        self.text_btn.setEnabled(True)
        self.text_btn.setText("⏹️ Cancel")
        self.text_btn.clicked.disconnect()
        self.text_btn.clicked.connect(self.cancel_extraction)
        self.text_display.clear()
        self.text_display.setPlainText("🔄 Starting text extraction...\n\nPlease wait while OCR processes the image.\nYou can continue using the app!")
        self.extraction_progress.setVisible(True)
        self.extraction_progress.setValue(0)
        self.info_label.setText("🔄 Initializing OCR processing...")
        self.statusBar().showMessage("🔄 Processing text extraction in background...")

    def reset_extraction_ui(self):
        """Reset UI after extraction"""
        self.text_btn.setEnabled(True)
        self.text_btn.setText("📝 EXTRACT TEXT")
        self.text_btn.clicked.disconnect()
        self.text_btn.clicked.connect(self.extract_text)
        self.extraction_progress.setVisible(False)

    def cancel_extraction(self):
        """Cancel ongoing extraction"""
        if self.ocr_worker and self.ocr_worker.isRunning():
            self.ocr_worker.cancel_extraction()
            self.text_display.setPlainText("❌ Text extraction cancelled by user.")
            self.info_label.setText("❌ Extraction cancelled")
            self.statusBar().showMessage("Text extraction cancelled")
            self.reset_extraction_ui()

    def on_extraction_progress(self, progress, message):
        """Handle extraction progress updates"""
        self.extraction_progress.setValue(progress)
        self.info_label.setText(f"🔄 {message}")
        self.statusBar().showMessage(f"Processing... {message}")
        self.text_display.setPlainText(f"🔄 {message}\n\nProgress: {progress}%\n\nProcessing in background...\nYou can continue using the app!")

    def on_extraction_completed(self, result):
        """Handle successful extraction completion with improved text display"""
        self.reset_extraction_ui()
        file_type = self.current_file_info['type']
        
        if result['success'] and result['text'].strip():
            # Create HTML-formatted text for better paragraph display
            html_text = ""
            paragraphs = result['text'].split('\n\n')
            
            for paragraph in paragraphs:
                # Check if paragraph looks like a heading
                if paragraph.isupper() or (len(paragraph) < 60 and ("RULE" in paragraph.upper() or "CHAPTER" in paragraph.upper())):
                    html_text += f"<h3 style='text-align:center; margin-top:20px; margin-bottom:10px;'>{paragraph}</h3>"
                else:
                    html_text += f"<p style='text-indent:20px; margin-bottom:12px;'>{paragraph}</p>"
            
            # Set rich text with preserved formatting
            self.text_display.setHtml(html_text)
            
            # Use a nice, readable serif font like in books
            font = QFont("Georgia", 12)
            self.text_display.setFont(font)
            
            # Set paragraph spacing
            document = self.text_display.document()
            option = document.defaultTextOption()
            option.setAlignment(Qt.AlignJustify)  # Book-like justified text
            option.setWrapMode(QTextOption.WordWrap)
            document.setDefaultTextOption(option)
            
            # Apply additional styling to text display for book-like appearance
            self.text_display.setStyleSheet("""
                QTextEdit {
                    background-color: #fffef7;  /* Slight cream color like paper */
                    padding: 20px;
                    line-height: 1.5;
                    border: 1px solid #e0e0d1;
                }
            """)
            
            # Set cursor to beginning
            cursor = self.text_display.textCursor()
            cursor.movePosition(QTextCursor.Start)
            self.text_display.setTextCursor(cursor)
            
            # Update status information
            confidence_text = f"{result['confidence']:.1f}%" if result['confidence'] > 0 else "N/A"
            page_info = f"Page {self.current_page + 1}/{self.total_pages}" if file_type == 'pdf' else ""
            self.info_label.setText(
                f"✅ Extracted {result['word_count']} words | "
                f"Confidence: {confidence_text} | "
                f"{page_info} | "
                f"Method: {result.get('best_method', 'standard')}"
            )
            self.statusBar().showMessage(f"✅ Text extraction completed - {result['word_count']} words found")
        
        elif result['success'] and not result['text'].strip():
            # No text detected case
            self.text_display.setText("No text was detected on this page.\n\nTips:\n• Try a different page if this is a PDF\n• Ensure the page has clear, readable text\n• Check that text is not too small or blurry")
            self.info_label.setText("⚠️ No text detected")
            self.statusBar().showMessage("OCR completed - no text found")
        
        else:
            # Error case
            error_msg = result.get('error', 'Unknown error')
            self.text_display.setText(f"Text extraction failed.\n\nError: {error_msg}")
            self.info_label.setText("❌ Text extraction failed")
            self.statusBar().showMessage("OCR failed")

    # Add this method to your MainWindow class
    def process_with_handler_detection(self):
        """Enhanced processing with handler detection"""
        
        if self.current_file:
            # Show in console which handler is being used
            result = self.file_handler.process_with_ocr(self.current_file)
            
            # Update your existing result display
            if result.get('success'):
                # Add handler info to result text
                handler_info = ""
                if 'handler_detection' in result:
                    detection = result['handler_detection']
                    handler_info = f"""

    🔍 HANDLER ANALYSIS:
    • Selected: {result.get('selected_handler', 'Unknown')}
    • Detected Best: {detection['most_likely_handler']}
    • Confidence: {detection['confidence']:.1f}/100
    • Quality: {detection['quality_metrics']['quality_level']}
    • Recommendation: {detection['recommendation']}
    """
                
                # Update your result display with handler info
                full_result = result.get('text', '') + handler_info
                # Display full_result in your UI
            
            return result            

    def on_extraction_failed(self, error_message):
        """Handle extraction failure"""
        self.reset_extraction_ui()
        self.text_display.setText(f"❌ Text extraction failed:\n\n{error_message}")
        self.info_label.setText("❌ OCR error occurred")
        self.statusBar().showMessage("Text extraction failed")

    # --- Text Panel actions ---
    def show_text_panel(self):
        """Show the text extraction panel"""
        self.text_panel.show()

    def hide_text_panel(self):
        """Hide the text extraction panel"""
        self.text_panel.hide()

    def copy_text(self):
        """Copy extracted text to clipboard"""
        text = self.text_display.toPlainText()
        if text.strip():
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            self.info_label.setText("Text copied to clipboard!")
            QTimer.singleShot(3000, lambda: self.info_label.setText("Ready for text extraction"))
        else:
            self.info_label.setText("No text to copy")

    def clear_text(self):
        """Clear extracted text"""
        self.text_display.clear()
        self.info_label.setText("Text cleared")

    def export_text(self):
        """Export extracted text to file"""
        text = self.text_display.toPlainText()
        if not text.strip():
            QMessageBox.information(self, "No Text", "No text to export. Extract text first.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Text",
            "extracted_text.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                self.info_label.setText(f"Text exported to {os.path.basename(file_path)}")
                self.statusBar().showMessage(f"Text exported successfully")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export text:\n{str(e)}")

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About EyeShot AI", 
            f"""<h3>✨ EyeShot AI v1.2.0 ✨</h3>
            <p><b>Smart OCR Text Extraction Tool with AI Enhancement</b></p>
            <p>Created by: <b>Tigran0000</b></p>
            <p>Last Updated: 2025-06-16</p>
            <br>
            <p><b>🌟 Key Features:</b></p>
            <p>• <b>AI-Enhanced OCR</b> - EasyOCR + Tesseract hybrid processing</p>
            <p>• <b>Structure Preservation</b> - Maintains text layout and formatting</p>
            <p>• <b>Draggable Image Viewer</b> - Click and drag to pan</p>
            <p>• <b>Advanced PDF Navigation</b> - Smooth page browsing</p>
            <p>• <b>Beautiful Thumbnails</b> - Visual page overview</p>
            <p>• <b>Responsive Design</b> - Adapts to any screen size</p>
            <p>• <b>Professional UI</b> - Clean, fast, and responsive</p>
            <br>
            <p><b>🖱️ Mouse Controls:</b></p>
            <p>• Click and drag to pan images</p>
            <p>• Ctrl + Mouse Wheel to zoom</p>
            <p>• Click thumbnails to jump to pages</p>
            <br>
            <p><b>📁 Supported Formats:</b></p>
            <p>• Images: PNG, JPEG</p>
            <p>• Documents: PDF (with advanced navigation)</p>
            <br>
            <p><b>⌨️ Keyboard Shortcuts:</b></p>
            <p>• Ctrl+O: Open file</p>
            <p>• Ctrl+E: Extract text</p>
            <p>• Ctrl+F: Fit to window</p>
            <p>• Ctrl+T: Toggle thumbnails</p>
            <p>• Ctrl+R: Rotate image</p>
            <p>• Page Up/Down: Navigate pages</p>
            <br>
            <p><b>🤖 AI-Powered OCR:</b></p>
            <p>• Neural network text detection</p>
            <p>• Automatic method selection</p>
            <p>• Superior accuracy on complex documents</p>
            <p>• Structure and formatting preservation</p>
            <br>
            <p><b>🔒 Free, Offline, Privacy-Focused!</b></p>
            <p>No data leaves your computer. 100% local processing.</p>""")

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        key = event.key()
        modifiers = event.modifiers()
        if modifiers == Qt.ControlModifier:
            if key == Qt.Key_O:
                self.open_file()
                return
            elif key == Qt.Key_E:
                self.extract_text()
                return
            elif key == Qt.Key_F:
                self.fit_to_window()
                return
            elif key == Qt.Key_T:
                self.toggle_thumbnails()
                return
            elif key == Qt.Key_R:
                self.rotate_image()
                return
            elif key == Qt.Key_S:
                self.export_text()
                return
            elif key == Qt.Key_C:
                self.copy_text()
                return
            elif key == Qt.Key_Plus or key == Qt.Key_Equal:
                self.zoom_in()
                return
            elif key == Qt.Key_Minus:
                self.zoom_out()
                return
        if modifiers == Qt.NoModifier:
            if key == Qt.Key_PageUp:
                self.previous_page()
                return
            elif key == Qt.Key_PageDown:
                self.next_page()
                return
            elif key == Qt.Key_Home:
                self.first_page()
                return
            elif key == Qt.Key_End:
                self.last_page()
                return
            elif key == Qt.Key_Escape:
                self.hide_text_panel()
                return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        """Handle application close event"""
        reply = QMessageBox.question(
            self, 
            "Exit EyeShot AI", 
            "Are you sure you want to exit?",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            # Clean up OCR worker
            if self.ocr_worker and self.ocr_worker.isRunning():
                self.ocr_worker.cancel_extraction()
                self.ocr_worker.wait(1000)  # Wait up to 1 second
            
            # Clean up resources
            self.pdf_images.clear()
            self.thumbnail_widgets.clear()
            event.accept()
        else:
            event.ignore()


        # Main entry point
        if __name__ == '__main__':
            app = QApplication(sys.argv)
            
            # Set fusion style for better cross-platform appearance
            app.setStyle('Fusion')
            
            # Create and show the main window
            main_window = MainWindow()
            main_window.show()
            
            # Start the event loop
            sys.exit(app.exec_())