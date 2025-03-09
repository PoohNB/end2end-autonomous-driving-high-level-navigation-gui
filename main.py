import sys
import os
from PyQt5.QtWidgets import QFileDialog, QSlider,QProgressBar,QDialog,QSizePolicy,QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QMessageBox, QScrollArea, QGridLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal,QTimer,QCoreApplication
from PyQt5.QtGui import QImage, QPixmap
from system.Robot import AVbot
import cv2
import serial.tools.list_ports
import traceback
import sys
from io import StringIO
import time
import shutil
import logging
logging.basicConfig(level=logging.INFO)

from qt_modules import GameButton,StartButton,StopButton,DetailedDisplayLabel,CalibrationPanel
from tools import camera_blankorcover,check_agent

class LoadAgentThread(QThread):
    finished = pyqtSignal(int, str)  # Signal to emit the number of cameras and any error message

    def __init__(self, loop_thread, zip_path):
        super().__init__()
        self.loop_thread = loop_thread
        self.zip_path = zip_path

    def run(self):
        try:
            n_cam = self.loop_thread.load_agent(self.zip_path)
            self.finished.emit(n_cam, "")
        except Exception as e:
            self.finished.emit(0, str(e))

class CheckValidAgentThread(QThread):
    finished = pyqtSignal(bool, str,str)  # Signal to emit the number of cameras and any error message

    def __init__(self, model_dir):
        super().__init__()
        self.model_dir = model_dir

    def run(self):
        try:
            valid = check_agent(self.model_dir)
            self.finished.emit(valid,self.model_dir, "")
        except Exception as e:
            self.finished.emit(False,self.model_dir, str(e))



class LoadingDialog(QDialog):
    def __init__(self, text,parent=None):
        super().__init__(parent)
        self.setWindowTitle(text)
        self.setModal(True)
        self.setFixedSize(300, 100)

        layout = QVBoxLayout()
        labe_text = f"{text}, please wait..."
        self.label = QLabel(labe_text)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

    def set_progress(self, value):
        self.progress_bar.setValue(value)


class PortCheckThread(QThread):
    port_disappeared = pyqtSignal()

    def __init__(self, port_name):
        super().__init__()
        self.port_name = port_name
        self.running = True

    def run(self):
        while self.running:
            if self.port_name not in [port.device for port in serial.tools.list_ports.comports()]:
                self.port_disappeared.emit()
                break
            time.sleep(1)  # Check every second

    def stop(self):
        self.running = False

class LoopThread(QThread):
    image_update = pyqtSignal(list)
    control_update = pyqtSignal(float, float,float,float)  # New signal for steer and throttle
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.cmd = [0]
        self.robot = AVbot()
    
    def load_agent(self, zip_path):
        self.robot.init_agent(zip_path)
        return self.robot.agent.n_cam

    def clear_data(self):
        self.images = []  # Clear the stored images
        self.image_update.emit([])  # Emit an empty list to clear the display
        self.control_update.emit(0, 0,0,0)

    def run(self):
        while self.running:
            try:
     
                steer, throttle,proc_time,ctrled_time = self.robot.step(self.cmd)            
                images = self.robot.get_vision()
                self.image_update.emit(images)
                self.control_update.emit(steer, throttle,proc_time,ctrled_time)  # Emit control values

            except Exception as e:
                tb = traceback.format_exc()
                print(f"Error in loop thread: {e}")
                print(f"Traceback details:\n{tb}")
        self.clear_data()

            
class CarControlGUI(QMainWindow):

    def __init__(self) -> None:
        super().__init__()


        self.loop_thread = LoopThread()
        self.loop_thread.image_update.connect(self.display_images)
        self.loop_thread.control_update.connect(self.update_control_display)
        self.camera_inputs=[]
        self.n_cam = 0
        self.cmd = [0]
        self.control_activated = False
        self.blank_cover_count = 0
        self.camera_blanking = True
        self.port_check_thread = None

        self.setWindowTitle("PathRover")
        self.setGeometry(100, 100, 1024, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.folder_combo = QComboBox()
        self.zip_combo = QComboBox()
        layout.addWidget(QLabel("Select Folder:"))
        layout.addWidget(self.folder_combo)
        layout.addWidget(QLabel("Select Zip File:"))
        layout.addWidget(self.zip_combo)
        self.folder_combo.addItems(["select"])
        self.zip_combo.addItems(["select"])
        self.folder_combo.currentTextChanged.connect(self.update_zip_files)
        self.populate_folder_combo()

        self.load_agent_button = QPushButton("Load agent")
        self.close_agent_button = QPushButton("Close agent")
        layout.addWidget(self.load_agent_button)
        layout.addWidget(self.close_agent_button)
        self.load_agent_button.clicked.connect(self.load_agent)
        self.close_agent_button.clicked.connect(self.close_agent)

        self.upload_agent_button = QPushButton("Upload Agent Model")
        layout.addWidget(self.upload_agent_button)
        self.upload_agent_button.clicked.connect(self.upload_agent)

        # create camera-selection related widget
        self.camera_input_layout = QHBoxLayout()
        layout.addLayout(self.camera_input_layout)

        self.refresh_cameras_button = QPushButton("Refresh Cameras")
        self.refresh_cameras_button.clicked.connect(self.refresh_cameras)
        self.refresh_cameras_button.setMaximumWidth(120)

        # Create port-selection widgets
        self.port_select_layout = QHBoxLayout()
        layout.addLayout(self.port_select_layout)

        self.port_container = QWidget()
        port_layout = QHBoxLayout(self.port_container)

        self.port_label = QLabel("Select Port:")
        self.port_combo = QComboBox()
        self.port_combo.currentIndexChanged.connect(self.update_port)

        self.refresh_ports_button = QPushButton("Refresh Ports")
        self.refresh_ports_button.clicked.connect(self.refresh_ports)
        self.refresh_ports_button.setMaximumWidth(110)

        port_layout.addWidget(self.port_label)
        port_layout.addWidget(self.port_combo,1)
        port_layout.addWidget(self.refresh_ports_button)

        # display 
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QGridLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area)

        # control
        button_layout = QGridLayout()
        self.button_left = GameButton("left")
        self.button_forward = GameButton("forward")
        self.button_right = GameButton("right")
        self.start_car_button = StartButton()
        self.stop_car_button = StopButton()

        # display agent info
        self.detailed_display = DetailedDisplayLabel()
        self.detailed_display.set_height(150)

        # calibration panel
        self.calibrate_panel = CalibrationPanel()
        self.calibrate_panel.setFixedSize(400, 200)
        self.calibrate_panel.connect(self.loop_thread.robot.calib.apply_setting)
        self.calibrate_panel.action_slider.valueChanged.connect(self.loop_thread.robot.apply_calibrating_mode)

        # add widget to button layout
        
        button_layout.addWidget(self.start_car_button, 0, 0)
        button_layout.addWidget(self.stop_car_button, 1, 0)        
        button_layout.addWidget(self.calibrate_panel,0,1)
        button_layout.addWidget(self.detailed_display, 0, 2,1,2)

        # high_level_cmd_panel = QHBoxLayout()
        # high_level_cmd_panel.addWidget(self.button_forward)
        # high_level_cmd_panel.addWidget(self.button_left)
        # high_level_cmd_panel.addWidget(self.button_right)
        button_layout.addWidget(self.button_forward, 1, 2)
        button_layout.addWidget(self.button_left, 1,1)
        button_layout.addWidget(self.button_right, 1, 3)

        # button_layout.addLayout(high_level_cmd_panel,1,1,1,3)
        layout.addLayout(button_layout)

        self.start_car_button.clicked.connect(self.activate_control)
        self.stop_car_button.clicked.connect(self.deactivate_control)
        self.button_forward.clicked.connect(lambda: self.set_cmd([0]))
        self.button_left.clicked.connect(lambda: self.set_cmd([1]))
        self.button_right.clicked.connect(lambda: self.set_cmd([2]))

        self.start_car_button.unhighlight()
        self.stop_car_button.highlight()
        

        # Redirect stdout
        self.old_stdout = sys.stdout
        self.stdout_capture = StringIO()
        sys.stdout = self.stdout_capture

        # Create a timer to update the command output display
        self.cmd_update_timer = QTimer(self)
        self.cmd_update_timer.timeout.connect(self.update_cmd_output)
        self.cmd_update_timer.start(100)  # Update every 100 ms

        self.showMaximized()

    def update_calibration_value(self):
        try:
            throttle_range = self.loop_thread.robot.agent.action_wrapper.throttle_range
            max_steer = self.loop_thread.robot.agent.action_wrapper.max_steer            
            print(f"limit throttle1 : {throttle_range}")
            print(f"max steer1 {max_steer}")
            self.loop_thread.robot.calib.apply_default_limit(max_steer,throttle_range[1])
            self.calibrate_panel.set_value(throttle_range[1]*100,max_steer*100,0,0)
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Not found default throttle limit and max steer info, calibration may have some limit and not acurate. \n Error: {e}")
        
        values = self.calibrate_panel.get_value()
        for k,v in values.items():
            self.loop_thread.robot.calib.apply_setting(k,v*100)


    def populate_folder_combo(self):
        self.folder_combo.clear()
        self.folder_combo.addItems(["select"])
        rlmodel_path = "RLmodel"  # Adjust this path if needed
        if os.path.exists(rlmodel_path) and os.path.isdir(rlmodel_path):
            folders = [f for f in os.listdir(rlmodel_path) if os.path.isdir(os.path.join(rlmodel_path, f))]
            self.folder_combo.addItems(folders)

    def update_zip_files(self, folder):
        self.zip_combo.clear()
        self.zip_combo.addItems(["select"])
        folder_path = os.path.join("RLmodel", folder)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            zip_files = [f for f in os.listdir(folder_path) if f.endswith('.zip')]
            self.zip_combo.clear()
            self.zip_combo.addItems(zip_files)

    def populate_port_combo(self):
        current_selection = self.port_combo.currentText()
        self.port_combo.clear()
        available_ports = [port.device for port in serial.tools.list_ports.comports()]
        if not available_ports:
            self.port_combo.addItem("No ports available")
        else:
            self.port_combo.addItems(["Select"] + available_ports)
        
        if current_selection in available_ports:
            self.port_combo.setCurrentText(current_selection)
        else:
            self.port_combo.setCurrentIndex(0)  # Set to "Select" if previous selection is no longer available

    def update_port(self):
        if self.control_activated:
            QMessageBox.warning(self, "Warning", "Cannot change port while control is activated.")
            return

        selected_port = self.port_combo.currentText()
        if selected_port != "Select" and selected_port != "No ports available":
            try:
                self.loop_thread.robot.set_control(selected_port)
                print(f"Set control port to: {selected_port}")
                
                # Provide visual feedback
                self.port_combo.setStyleSheet("background-color: #c8e6c9;")  # Light green background
                QTimer.singleShot(1000, lambda: self.port_combo.setStyleSheet(""))  # Reset after 1 second
            except Exception as e:
                print(f"Error setting control port: {str(e)}")
                
                # Provide visual feedback for error
                self.port_combo.setStyleSheet("background-color: #ffcdd2;")  # Light red background
                QTimer.singleShot(1000, lambda: self.port_combo.setStyleSheet(""))  # Reset after 1 second
        else:
            print("No valid port selected")

            selected_port = self.port_combo.currentText()
            if selected_port != "Select" and selected_port != "No ports available" and  selected_port != "":
                try:
                    self.loop_thread.robot.set_control(selected_port)
                    print(f"Set control port to: {selected_port}")
                    
                    # Provide visual feedback
                    self.port_combo.setStyleSheet("background-color: #c8e6c9;")  # Light green background
                    QTimer.singleShot(1000, lambda: self.port_combo.setStyleSheet(""))  # Reset after 1 second
                except Exception as e:
                    print(f"Error setting control port: {str(e)}")
                    
                    # Provide visual feedback for error
                    self.port_combo.setStyleSheet("background-color: #ffcdd2;")  # Light red background
                    QTimer.singleShot(1000, lambda: self.port_combo.setStyleSheet(""))  # Reset after 1 second
            else:
                print("No valid port selected")

    def set_cmd(self, value):
        self.cmd = value
        self.update_control_display()  # We'll update steer and throttle later
        if self.loop_thread.running:
            self.loop_thread.cmd = value
        print(f"Command set to: {value}")
    
    def update_control_display(self, steer=None, throttle=None,proc_time=None,ctrl_time=None):
        self.detailed_display.update_display(cmd=self.cmd, steer=steer, throttle=throttle,proc_time=proc_time,ctrl_time=ctrl_time)

    def update_cmd_output(self):
        try:
            captured_output = self.stdout_capture.getvalue()
            if captured_output:
                self.detailed_display.append_cmd_output(captured_output)
                self.stdout_capture.truncate(0)
                self.stdout_capture.seek(0)
        except ValueError:
            # The StringIO object is closed, so we should stop the timer
            self.cmd_update_timer.stop()
            print("Command output capture has been closed.")

    def activate_control(self):

        if not self.check_cameras_selected():
            QMessageBox.warning(self, "Warning", "Please select all cameras before activating control.")
            return
        
        if self.camera_blanking:
            QMessageBox.warning(self, "Warning", "Camera is blank or cover")
            return
        
        selected_port = self.port_combo.currentText()
        
        if selected_port == "Select" or selected_port == "No ports available" or selected_port == "":
            reply = QMessageBox.question(self, 'No port select', 'The car will runing in test mode, are you sure?',
                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                
                return

        else:
            # Start port checking thread
            self.port_check_thread = PortCheckThread(selected_port)
            self.port_check_thread.port_disappeared.connect(self.handle_port_disappearance)
            self.port_check_thread.start()
        
        if self.loop_thread.robot.agent:
            self.set_cmd([0])
            self.loop_thread.robot.start()
            self.control_activated = True
            self.disable_camera_selection()
            self.port_combo.setEnabled(False)
            self.start_car_button.highlight()
            self.stop_car_button.unhighlight()
        else:
            QMessageBox.warning(self, "Warning", "Agent not ready")

    def deactivate_control(self):
        if self.loop_thread.robot.agent:
            print("deactivating control..")
            self.loop_thread.robot.stop()
            self.control_activated = False
            self.enable_camera_selection()
            self.port_combo.setEnabled(True)
            self.blank_cover_count = 0
            self.camera_blanking = True
            self.start_car_button.unhighlight()
            self.stop_car_button.highlight()

            # Stop port checking thread
            if self.port_check_thread:
                self.port_check_thread.stop()
                self.port_check_thread.wait()
                self.port_check_thread = None

    def handle_port_disappearance(self):
        QMessageBox.warning(self, "Emergency", "Control port has disappeared. press the emergency button on your left or right and use manual control.")
        self.deactivate_control()

    def refresh_ports(self):
        if self.control_activated:
            QMessageBox.warning(self, "Warning", "Cannot refresh ports while control is activated.")
            return

        self.populate_port_combo()
        QMessageBox.information(self, "Success", "Port list has been refreshed.")

    def check_cameras_selected(self):
        return all(combo.currentText() != "Select" for combo in self.camera_inputs)

    def disable_camera_selection(self):
        for combo in self.camera_inputs:
            combo.setEnabled(False)

    def enable_camera_selection(self):
        for combo in self.camera_inputs:
            combo.setEnabled(True)

    def refresh_cameras(self):
        if not self.loop_thread.robot.agent:
            QMessageBox.warning(self, "Warning", "Please load an agent first.")
            return

        if self.control_activated:
            QMessageBox.warning(self, "Warning", "Cannot refresh cameras while control is activated.")
            return

        self.populate_camera_combos()
        QMessageBox.information(self, "Success", "Camera list has been refreshed.")


    def upload_agent(self):
            
        if self.loop_thread.robot.agent:
            QMessageBox.warning(self, "Warning", "Cannot upload agent please close the current agent first")
            return

        folder_path = QFileDialog.getExistingDirectory(self, "Select Agent Folder")
        print("selected the folder",folder_path)

        if folder_path:

            # Create and show loading dialog
            self.uploading_dialog = LoadingDialog("checking Agent",self)
            self.uploading_dialog.show()

            # Create and start loading thread
            self.checking_thread = CheckValidAgentThread(folder_path)
            self.checking_thread.finished.connect(self.on_agent_uploaded)
            self.checking_thread.start()

    def copy_folder(self,folder_path):
        folder_name = os.path.basename(folder_path)
        destination = os.path.join("RLmodel", folder_name)
        
        if os.path.exists(destination):
            reply = QMessageBox.question(self, 'Folder Exists', 
                                        f'A folder named "{folder_name}" already exists in RLmodel. Do you want to replace it?',
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return
            shutil.rmtree(destination)
        
        try:
            shutil.copytree(folder_path, destination)
            QMessageBox.information(self, "Success", f"Agent '{folder_name}' has been uploaded successfully.")
            self.populate_folder_combo() 
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to copy agent folder: {str(e)}")


    def on_agent_uploaded(self,valid,model_dir,error_message):
        self.uploading_dialog.close()

        if error_message:
            QMessageBox.critical(self, "Error", f"Failed to upload agent: {error_message}")
            return
        
        if not valid: 
            QMessageBox.warning(self, "Invalid Agent", "The selected folder does not contain a valid agent or use wrong config file.")
            return
        else:
            self.copy_folder(model_dir) 

    def load_agent(self):
        
        if self.loop_thread.robot.agent:
            QMessageBox.warning(self, "Warning", "Can't load agent before close agent.")
            return

        selected_folder = self.folder_combo.currentText()
        selected_zip = self.zip_combo.currentText()
        
        if selected_folder == "select" or selected_zip == "select":
            QMessageBox.warning(self, "Warning", "Please select both a folder and a zip file.")
            return

        zip_path = os.path.join("RLmodel", selected_folder, selected_zip)

        # Create and show loading dialog
        self.loading_dialog = LoadingDialog("Loading Agent",self)
        self.loading_dialog.show()

        # Create and start loading thread
        self.load_thread = LoadAgentThread(self.loop_thread, zip_path)
        self.load_thread.finished.connect(self.on_agent_loaded)
        
        self.load_thread.start()

    def on_agent_loaded(self, n_cam, error_message):
        self.loading_dialog.close()

        if error_message:
            QMessageBox.critical(self, "Error", f"Failed to load agent: {error_message}")
            return
        
        self.n_cam = n_cam
        self.setup_camera_inputs()
        self.setup_port_selection() 
        self.update_calibration_value()
        #
        self.loop_thread.robot.reset()

        QMessageBox.information(self, "Success", f"Agent loaded successfully")

        self.loop_thread.running = True
        self.loop_thread.start()

    def check_action_limit(self):

        low_st,low_th = self.loop_thread.robot.agent.action_wrapper([0,0])
        high_st,high_th = self.loop_thread.robot.agent.action_wrapper([1,1])
        # (steer,throttle)
        

    def setup_port_selection(self):

        # Add the port container widget to the main layout
        self.port_select_layout.addWidget(self.port_container)
        # Populate the port combo box
        self.populate_port_combo()

        # # Connect the port combo box signal
        # self.port_combo.currentIndexChanged.connect(self.update_port)

    def setup_camera_inputs(self):
        # Clear existing camera inputs
        for i in reversed(range(self.camera_input_layout.count())): 
            self.camera_input_layout.itemAt(i).widget().setParent(None)

        # Create new camera inputs
        self.camera_inputs = []
        for i in range(self.n_cam):
            container = QWidget()
            layout = QVBoxLayout(container)
            
            label = QLabel(f"Camera {i} index:")
            combo = QComboBox()
            combo.setObjectName(f"camera_{i}")
            combo.currentIndexChanged.connect(self.update_camera_indices)
            
            layout.addWidget(label)
            layout.addWidget(combo)
            
            self.camera_input_layout.addWidget(container)
            self.camera_inputs.append(combo)

        # Add the refresh button to the camera input layout
        self.camera_input_layout.addWidget(self.refresh_cameras_button)

        # Populate the combo boxes with available camera indices
        self.populate_camera_combos()

    def populate_camera_combos(self):
        available_cameras = self.get_available_cameras()
        for combo in self.camera_inputs:
            current_selection = combo.currentText()
            combo.clear()
            combo.addItems(["Select"] + [str(i) for i in available_cameras])
            if current_selection in [str(i) for i in available_cameras]:
                combo.setCurrentText(current_selection)
            else:
                combo.setCurrentIndex(0)  # Set to "Select" if previous selection is no longer available

    def get_available_cameras(self):
        print("checking for camera..")
        available_cameras = []
        for i in range(6):  # Check first 10 indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras

    def update_camera_indices(self):

        sender = self.sender()
        if isinstance(sender, QComboBox):
            cam_position = self.camera_inputs.index(sender)
            cam_index = sender.currentIndex() - 1  # Subtract 1 because "Select" is at index 0
            if cam_index >= 0 :
                self.loop_thread.robot.set_cameras(cam_index, cam_position)
                print(f"Set camera {cam_position} to index {cam_index}")
                
                # Provide visual feedback
                sender.setStyleSheet("background-color: #c8e6c9;")  # Light green background
                QTimer.singleShot(1000, lambda: sender.setStyleSheet(""))  # Reset after 1 second
            else:
                print(f"No camera selected for position {cam_position}")
                
                # Provide visual feedback for error
                sender.setStyleSheet("background-color: #ffcdd2;")  # Light red background
                QTimer.singleShot(1000, lambda: sender.setStyleSheet(""))  # Reset after 1 second
                
    def display_images(self, images):

        if len(images) > 0:
            if len(images[0]) > 0:
                if camera_blankorcover(images[0][0],th=45):
                    self.camera_blanking =True
                    if self.control_activated:
                        self.blank_cover_count+=1
                else:
                    self.camera_blanking =False
                    if self.control_activated:
                        self.blank_cover_count=0
                
  
        if self.control_activated and self.blank_cover_count > 1:
            print("camera is blank or cover")
            self.deactivate_control()

        # Clear the existing layout
        for i in reversed(range(self.scroll_layout.count())): 
            self.scroll_layout.itemAt(i).widget().setParent(None)
        
        # Calculate the number of columns and rows
        num_columns = len(images)
        num_rows = max(len(group) for group in images) if images else 0

        # Calculate the size for each image
        available_width = self.scroll_area.viewport().width() - (num_columns + 1) * self.scroll_layout.spacing()
        available_height = self.scroll_area.viewport().height() - (num_rows + 1) * self.scroll_layout.spacing()
        image_width = available_width // num_columns if num_columns > 0 else available_width
        image_height = available_height // num_rows if num_rows > 0 else available_height

        # Display new images
        for col, image_group in enumerate(images):
            for row, image in enumerate(image_group):
                # Check if the image is grayscale or color
                if len(image.shape) == 2:
                    height, width = image.shape
                    q_image = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
                elif len(image.shape) == 3:
                    height, width, channel = image.shape
                    bytes_per_line = 3 * width
                    q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                else:
                    print(f"Unexpected image shape: {image.shape}")
                    continue

                pixmap = QPixmap.fromImage(q_image)
                
                # Create a label and set the pixmap
                label = QLabel()
                label.setPixmap(pixmap.scaled(image_width, image_height, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                label.setAlignment(Qt.AlignCenter)
                label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                
                # Add the label to the layout
                self.scroll_layout.addWidget(label, row, col)
        
        # Update the scroll area
        self.scroll_content.adjustSize()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Recalculate image sizes when the window is resized
        if hasattr(self, 'loop_thread') and self.loop_thread.robot.agent:
            images = self.loop_thread.robot.get_vision()
            self.display_images(images)

    def closeEvent(self, event):
        """Handle the window close event."""

        if self.loop_thread.robot:
            reply = QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?',
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                self.close_agent()

                # Stop the command update timer
                self.cmd_update_timer.stop()

                # Restore stdout
                sys.stdout = self.old_stdout
                
                # Close the StringIO object
                if not self.stdout_capture.closed:
                    self.stdout_capture.close()

                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def close_agent(self):
        if self.loop_thread.robot.agent:
            if self.control_activated:
                self.deactivate_control()
            self.loop_thread.running = False
            self.loop_thread.wait()  # Wait for the thread to finish
            self.loop_thread.robot.close()
            # Clear camera inputs
            self.clear_layout(self.camera_input_layout)
            self.camera_inputs.clear()  # Clear the list of camera inputs  

            # clear port selection
            self.clear_layout(self.port_select_layout)
     
            QMessageBox.information(self,"Success", "Agent closed successfully")
        else:
            QMessageBox.information(self,"Fail", "No agent is currently loaded")

    def clear_layout(self,layout):

        for i in reversed(range(layout.count())): 
            widget = layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

    def force_quit(self):
        """Force quit the application."""
        self.close_agent()
        QCoreApplication.quit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    window = CarControlGUI()
    window.show()
    sys.exit(app.exec_())