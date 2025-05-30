"""
Splash Screen Module
-----------------

Provides an enhanced splash screen for the CEA Analyzer application.
"""

import os
import time
from PyQt6.QtWidgets import QSplashScreen, QApplication, QProgressBar, QVBoxLayout, QWidget
from PyQt6.QtGui import QPixmap, QPainter, QColor, QFont, QLinearGradient, QRadialGradient
from PyQt6.QtCore import Qt, QSize, QTimer, pyqtSignal

class EnhancedSplashScreen(QSplashScreen):
    """
    Enhanced splash screen with progress indicator and customizable styling.
    """
    
    def __init__(self, pixmap=None, parent=None, progress_bar=True):
        """Initialize the splash screen."""
        if pixmap is None:
            pixmap = self._create_default_splash()
            
        super().__init__(pixmap, parent)
        
        # Set a fixed size
        self.setFixedSize(pixmap.size())
        
        # Add progress bar if requested
        self.progress_bar = None
        if progress_bar:
            self.progress_bar = QProgressBar(self)
            self.progress_bar.setGeometry(50, pixmap.height() - 30, pixmap.width() - 100, 20)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #444;
                    border-radius: 5px;
                    background: rgba(30, 30, 30, 100);
                    color: white;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0078d7, stop:1 #5ea3e6);
                    border-radius: 5px;
                }
            """)
            
        # Start animation timer
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._animate_progress)
        self.animation_timer.start(50)  # Update every 50ms
        
        # Set progress to 0
        self.progress = 0
        
    def _animate_progress(self):
        """Animate the progress bar."""
        if self.progress_bar is not None:
            self.progress += 1
            if self.progress > 100:
                self.progress = 0
            self.progress_bar.setValue(self.progress)
            
    def showMessage(self, message, alignment=Qt.AlignBottom | Qt.AlignHCenter, color=Qt.white):
        """Show a message on the splash screen."""
        super().showMessage(message, alignment, color)
        QApplication.processEvents()
        
    def finish(self, main_window):
        """Finish the splash screen."""
        self.animation_timer.stop()
        super().finish(main_window)
        
    @staticmethod
    def _create_default_splash():
        """Create a default splash pixmap."""
        # Create a pixmap with a nice gradient background
        pixmap = QPixmap(600, 400)
        
        # Create a gradient background
        gradient = QLinearGradient(0, 0, 0, 400)
        gradient.setColorAt(0, QColor(0, 40, 80))   # Dark blue at top
        gradient.setColorAt(1, QColor(0, 15, 30))   # Darker blue at bottom
        
        # Paint the background
        painter = QPainter(pixmap)
        painter.fillRect(pixmap.rect(), gradient)
        
        # Add a radial gradient for a subtle highlight
        radial = QRadialGradient(300, 150, 300)
        radial.setColorAt(0, QColor(100, 150, 200, 100))  # Light blue center
        radial.setColorAt(1, QColor(0, 40, 80, 0))        # Transparent outer
        painter.setOpacity(0.7)
        painter.fillRect(pixmap.rect(), radial)
        painter.setOpacity(1.0)
        
        # Draw application name
        font = QFont("Arial", 40)
        font.setWeight(QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "CEA Analyzer")
        
        # Draw subtitle
        painter.setFont(QFont("Arial", 20))
        painter.setPen(QColor(220, 220, 220))
        painter.drawText(pixmap.rect().adjusted(0, 80, 0, 0), 
                        Qt.AlignmentFlag.AlignCenter, "Modern Rocket Design Tools")
        
        # Draw version
        try:
            from cea_analyzer import __version__
        except ImportError:
            __version__ = "2.0.0"  # Fallback version
            
        painter.setFont(QFont("Arial", 12))
        painter.setPen(QColor(180, 180, 180))
        painter.drawText(pixmap.rect().adjusted(0, 0, -20, -20), 
                        Qt.AlignRight | Qt.AlignBottom, 
                        f"Version {__version__}")
        
        # End painting
        painter.end()
        
        return pixmap
        
    @staticmethod
    def create_splash(progress_bar=True):
        """Create and return a splash screen instance."""
        # Create splash screen
        splash = EnhancedSplashScreen(progress_bar=progress_bar)
        return splash
