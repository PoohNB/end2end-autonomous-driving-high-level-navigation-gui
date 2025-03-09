from PyQt5.QtWidgets import QGridLayout,QSlider,QTextEdit, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import  QFont, QColor,QPainter,QLinearGradient,QPen


class GameButton(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        self.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 15px;
                padding: 15px;
                font-size: 18px;
                font-weight: bold;
                text-transform: uppercase;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1c4966;
            }
        """)
        self.setFont(QFont("Arial", 14))
        self.setFixedSize(120, 120)  # Set a fixed size for the button



class CarControlButton(QPushButton):
    def __init__(self, text, color, parent=None):
        super().__init__(text, parent)
        self.base_color = QColor(color)
        self.is_highlighted = False
        self.setFixedSize(120, 120)
        self.setCursor(Qt.PointingHandCursor)
        self.setFont(QFont('Arial', 14, QFont.Bold))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Button shape
        rect = self.rect().adjusted(5, 5, -5, -5)
        
        # Gradient
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        if self.is_highlighted:
            gradient.setColorAt(0, self.base_color.lighter(150))
            gradient.setColorAt(1, self.base_color.lighter(120))
        else:
            gradient.setColorAt(0, self.base_color.lighter(120))
            gradient.setColorAt(1, self.base_color)

        # Fill
        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(rect)

        # Border
        if self.is_highlighted:
            painter.setPen(QPen(Qt.white, 2))
        else:
            painter.setPen(QPen(self.base_color.darker(120), 2))
        painter.drawEllipse(rect)

        # Text
        painter.setPen(Qt.white)
        painter.drawText(rect, Qt.AlignCenter, self.text())

    def highlight(self):
        self.is_highlighted = True
        self.update()

    def unhighlight(self):
        self.is_highlighted = False
        self.update()

    def enterEvent(self, event):
        self.setCursor(Qt.PointingHandCursor)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        super().leaveEvent(event)
class StartButton(CarControlButton):
    def __init__(self, parent=None):
        super().__init__("Autopilot", "#28a745", parent)

class StopButton(CarControlButton):
    def __init__(self, parent=None):
        super().__init__("Brake", "#dc3545", parent)



# class DisplayLabel(QLabel):
#     def __init__(self, text):
#         super().__init__(text)
#         self.setStyleSheet("""
#             background-color: #2c3e50;
#             color: #ecf0f1;
#             border: 2px solid #34495e;
#             border-radius: 5px;
#             padding: 10px;
#             font-family: 'Courier New', monospace;
#             font-size: 24px;
#         """)
#         self.setAlignment(Qt.AlignCenter)
#         self.setMinimumHeight(50)


class DetailedDisplayLabel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setSpacing(10)

        # Left layout
        self.left_layout = QVBoxLayout()
        self.left_layout.setSpacing(1)
        self.left_layout.setContentsMargins(5, 5, 5, 5)

        self.labels = {
            'cmd': QLabel("Command: [0]"),
            'steer': QLabel("Steer: 0.0"),
            'throttle': QLabel("Throttle: 0.0"),
            'proc_time': QLabel("Process time: 0.0"),
            'ctrl_time': QLabel("Controlled time: 0.0")
        }

        for label in self.labels.values():
            self.left_layout.addWidget(label)

        # Right layout
        self.right_layout = QVBoxLayout()
        self.right_layout.setContentsMargins(5, 5, 5, 5)

        self.cmd_box = QTextEdit()
        self.cmd_box.setMinimumWidth(200)  
        self.cmd_box.setReadOnly(True)

        self.right_layout.addWidget(QLabel("Command Output:"))
        self.right_layout.addWidget(self.cmd_box)

        # Add layouts to main layout
        self.main_layout.addLayout(self.left_layout, 1)
        self.main_layout.addLayout(self.right_layout, 1)

    def update_display(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.labels and value is not None:
                if isinstance(value,float) or isinstance(value,int):
                    self.labels[key].setText(f"{key.capitalize()}: {value:.2f}")
                else:
                    self.labels[key].setText(f"{key.capitalize()}: {value}")

    def append_cmd_output(self, text):
        self.cmd_box.append(text)
        self.cmd_box.moveCursor(self.cmd_box.textCursor().End)

    def set_height(self, height):
        self.setFixedHeight(height)
        left_height = height - 10  # Subtract margins
        right_height = height - 30  # Subtract margins and label height

        # Adjust left layout
        for label in self.labels.values():
            label.setFixedHeight(left_height // 5)  # Divide equally among 5 labels

        # Adjust right layout
        self.cmd_box.setFixedHeight(right_height)

        self.updateGeometry()

class CalibrationPanel(QWidget):
    def __init__(self):
        super().__init__()

        # Throttle Limit Slider
        self.throttle_label = QLabel("Throttle Limit: 0.00", self)
        self.throttle_slider = QSlider(Qt.Horizontal)
        self.throttle_slider.valueChanged.connect(lambda value: self.update_label(self.throttle_label, value, "Throttle Limit"))
        self.throttle_slider.setMinimum(0)
        self.throttle_slider.setMaximum(100)
        self.throttle_slider.setValue(40)
        self.throttle_slider.setTickPosition(QSlider.TicksBelow)
        self.throttle_slider.setTickInterval(10)

        # Steer Limit Slider
        self.steer_limit_label = QLabel("Steer Limit: 0.00", self)
        self.steer_limit_slider = QSlider(Qt.Horizontal)        
        self.steer_limit_slider.valueChanged.connect(lambda value: self.update_label(self.steer_limit_label, value, "Steer Limit"))
        self.steer_limit_slider.setMinimum(20)
        self.steer_limit_slider.setMaximum(100)
        self.steer_limit_slider.setValue(60)
        self.steer_limit_slider.setTickPosition(QSlider.TicksBelow)
        self.steer_limit_slider.setTickInterval(10)

        # Steer Center Slider with range from -0.2 to 0.2
        self.steer_center_label = QLabel("Steer Center: 0.00", self)
        self.steer_center_slider = QSlider(Qt.Horizontal)
        self.steer_center_slider.valueChanged.connect(lambda value: self.update_label(self.steer_center_label, value, "Steer Center"))
        self.steer_center_slider.setMinimum(-60)
        self.steer_center_slider.setMaximum(60)
        self.steer_center_slider.setValue(0)
        self.steer_center_slider.setTickPosition(QSlider.TicksBelow)
        self.steer_center_slider.setTickInterval(5)

        # Delay time slider (0.2 to 1)
        self.delay_label = QLabel("Ctrl Delay: 0.00", self)
        self.delay_slider = QSlider(Qt.Horizontal)
        self.delay_slider.valueChanged.connect(lambda value: self.update_label(self.delay_label, value, "Ctrl Delay", scale=0.01))
        self.delay_slider.setMinimum(20)
        self.delay_slider.setMaximum(120)
        self.delay_slider.setValue(40)
        self.delay_slider.setTickPosition(QSlider.TicksBelow)
        self.delay_slider.setTickInterval(10)

        # Action Slider
        self.action_label = QLabel("Action: Agent", self)
        self.action_slider = QSlider(Qt.Horizontal)
        self.action_slider.valueChanged.connect(self.update_action_label)
        self.action_slider.setMinimum(0)
        self.action_slider.setMaximum(3)
        self.action_slider.setValue(0)
        self.action_slider.setTickPosition(QSlider.TicksBelow)
        self.action_slider.setTickInterval(1)

        # Layout to organize sliders and labels horizontally to reduce height
        layout = QGridLayout()
        layout.addWidget(self.throttle_label, 0, 0)
        layout.addWidget(self.throttle_slider, 1, 0)
        layout.addWidget(self.steer_limit_label, 0, 1)
        layout.addWidget(self.steer_limit_slider, 1, 1)
        layout.addWidget(self.steer_center_label, 2, 0)
        layout.addWidget(self.steer_center_slider, 3, 0)
        layout.addWidget(self.delay_label, 2, 1)
        layout.addWidget(self.delay_slider, 3, 1)
        layout.addWidget(self.action_label, 4, 0)
        layout.addWidget(self.action_slider, 5, 0, 1, 2)  # Span two columns
        
        self.setLayout(layout)

    def update_label(self, label, value, prefix, scale=0.01):
        # Update the label with the current slider value and apply the scale
        label.setText(f"{prefix}: {value * scale:.2f}")

    def update_action_label(self, value):
        actions = ["Agent", "Forward", "Left", "Right"]
        self.action_label.setText(f"Action: {actions[value]}")

    def set_value(self,
                  throttle_limit,
                  steer_limit,
                  steer_center,
                  delay):
        
        self.throttle_slider.setValue(throttle_limit)
        self.steer_limit_slider.setValue(steer_limit)
        self.steer_center_slider.setValue(steer_center)
        self.delay_slider.setValue(delay)

    def connect(self, function):
        self.throttle_slider.valueChanged.connect(lambda value: function('throttle', value))
        self.steer_limit_slider.valueChanged.connect(lambda value: function('steer_limit', value))
        self.steer_center_slider.valueChanged.connect(lambda value: function('steer_center', value))
        self.delay_slider.valueChanged.connect(lambda value: function('delay', value))

    def get_value(self):
        # Return every slider's value in a dictionary
        return {
            "throttle": self.throttle_slider.value() * 0.01,
            "steer_limit": self.steer_limit_slider.value() * 0.01,
            "steer_center": self.steer_center_slider.value() * 0.01,
            "delay": self.delay_slider.value() * 0.01
        }