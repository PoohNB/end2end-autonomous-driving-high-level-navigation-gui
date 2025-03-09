import tkinter as tk
from tkinter import ttk
import serial
import serial.tools.list_ports
import threading
import time
import json

COMMANDS = {
    'most_left': [36, 0, 60, 0, 0, 64],
    'left': [36, 55, 60, 0, 0, 64],
    'bit_left': [36, 109, 70, 0, 0, 64],
    'forward': [36, 128, 90, 0, 0, 64],
    'bit_right': [36, 145, 70, 0, 0, 64],
    'right': [36, 199, 60, 0, 0, 64],
    'most_right': [36, 255, 60, 0, 0, 64],
    'ready': [36, 127, 0, 0, 0, 64],
    'brake': [36, 127, 0, 255, 0, 64],
}

class CarControlApp:
    def __init__(self, master):
        self.master = master
        master.title("Car Control UI")
        master.geometry("1000x800")  # Increased size to accommodate new features

        self.serial_port = None
        self.is_connected = False
        self.current_command = 'ready'

        # Default calibration values
        self.default_middle_steer = 128
        self.default_steer_limit = 128
        self.default_throttle_limit = 255
        self.default_delay_time = 0.4

        # Load saved calibration values or use defaults
        self.load_calibration()

        self.create_widgets()
        self.configure_grid()

    def create_widgets(self):
        # Port selection and baudrate
        ttk.Label(self.master, text="Select Port:", font=('Arial', 14)).grid(row=0, column=0, padx=10, pady=10, sticky="e")
        self.port_combo = ttk.Combobox(self.master, values=self.get_serial_ports(), font=('Arial', 14), width=30)
        self.port_combo.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        ttk.Label(self.master, text="Baudrate:", font=('Arial', 14)).grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.baudrate_combo = ttk.Combobox(self.master, values=[9600, 19200, 38400, 57600, 115200], font=('Arial', 14))
        self.baudrate_combo.set(115200)  # Default baudrate
        self.baudrate_combo.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        # Connect/Disconnect button
        self.connect_button = ttk.Button(self.master, text="Connect", command=self.toggle_connection, style='Large.TButton')
        self.connect_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        # Control buttons
        self.create_control_buttons()

        # Calibration panel
        self.create_calibration_panel()

    def create_calibration_panel(self):
        calibration_frame = ttk.LabelFrame(self.master, text="Calibration", padding="10 10 10 10")
        calibration_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        # Configure grid for calibration frame
        calibration_frame.columnconfigure(1, weight=1)  # Make the slider column expandable

        # Middle Steer Value
        ttk.Label(calibration_frame, text="Middle Steer Value:").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.middle_steer_var = tk.IntVar(value=self.middle_steer)
        middle_steer_scale = ttk.Scale(calibration_frame, from_=51, to=203, variable=self.middle_steer_var, command=self.update_middle_steer)
        middle_steer_scale.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        ttk.Label(calibration_frame, textvariable=self.middle_steer_var, width=5).grid(row=0, column=2, sticky="w")

        # Steer Limit Value
        ttk.Label(calibration_frame, text="Steer Limit Value:").grid(row=1, column=0, sticky="w", padx=(0, 10))
        self.steer_limit_var = tk.IntVar(value=self.steer_limit)
        steer_limit_scale = ttk.Scale(calibration_frame, from_=0, to=127, variable=self.steer_limit_var, command=self.update_steer_limit)
        steer_limit_scale.grid(row=1, column=1, sticky="ew", padx=(0, 10))
        ttk.Label(calibration_frame, textvariable=self.steer_limit_var, width=5).grid(row=1, column=2, sticky="w")

        # Throttle Limit Value
        ttk.Label(calibration_frame, text="Throttle Limit Value:").grid(row=2, column=0, sticky="w", padx=(0, 10))
        self.throttle_limit_var = tk.IntVar(value=self.throttle_limit)
        throttle_limit_scale = ttk.Scale(calibration_frame, from_=0, to=255, variable=self.throttle_limit_var, command=self.update_throttle_limit)
        throttle_limit_scale.grid(row=2, column=1, sticky="ew", padx=(0, 10))
        ttk.Label(calibration_frame, textvariable=self.throttle_limit_var, width=5).grid(row=2, column=2, sticky="w")

        # Delay Time Value
        ttk.Label(calibration_frame, text="Delay Time (s):").grid(row=3, column=0, sticky="w", padx=(0, 10))
        self.delay_time_var = tk.DoubleVar(value=self.delay_time)
        delay_time_scale = ttk.Scale(calibration_frame, from_=0.4, to=1.0, variable=self.delay_time_var, command=self.update_delay_time)
        delay_time_scale.grid(row=3, column=1, sticky="ew", padx=(0, 10))
        ttk.Label(calibration_frame, textvariable=self.delay_time_var, width=5).grid(row=3, column=2, sticky="w")

        # Save button
        ttk.Button(calibration_frame, text="Save Calibration", command=self.save_calibration).grid(row=4, column=0, columnspan=3, pady=10)


    def create_control_buttons(self):
        button_frame = ttk.Frame(self.master)
        button_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        for i in range(4):
            button_frame.columnconfigure(i, weight=1)
        for i in range(4):
            button_frame.rowconfigure(i, weight=1)

        # Define button styles
        style = ttk.Style()
        style.configure('Large.TButton', font=('Arial', 14))
        style.configure('Active.TButton', font=('Arial', 14), background='#4CAF50')

        # Create buttons
        self.buttons = {}
        button_config = [
            ('forward', "Forward", 0, 1, 2),
            ('most_left', "Most Left", 1, 0, 1),
            ('left', "Left", 1, 1, 1),
            ('right', "Right", 1, 2, 1),
            ('most_right', "Most Right", 1, 3, 1),
            ('bit_left', "Bit Left", 2, 1, 1),
            ('bit_right', "Bit Right", 2, 2, 1),
            ('ready', "Ready", 3, 0, 2),
            ('brake', "Brake", 3, 2, 2),
        ]

        for cmd, text, row, col, colspan in button_config:
            button = ttk.Button(button_frame, text=text, command=lambda c=cmd: self.set_command(c), style='Large.TButton')
            button.grid(row=row, column=col, columnspan=colspan, padx=5, pady=5, sticky="nsew")
            self.buttons[cmd] = button

        # Set initial active button
        self.update_active_button('ready')

    def update_middle_steer(self, value):
        self.middle_steer = int(float(value))

    def update_steer_limit(self, value):
        self.steer_limit = int(float(value))

    def update_throttle_limit(self, value):
        self.throttle_limit = int(float(value))

    def update_delay_time(self, value):
        self.delay_time = round(float(value), 2)

    def load_calibration(self):
        try:
            with open('calibration.json', 'r') as f:
                calibration = json.load(f)
            self.middle_steer = calibration.get('middle_steer', self.default_middle_steer)
            self.steer_limit = calibration.get('steer_limit', self.default_steer_limit)
            self.throttle_limit = calibration.get('throttle_limit', self.default_throttle_limit)
            self.delay_time = calibration.get('delay_time', self.default_delay_time)
        except FileNotFoundError:
            self.middle_steer = self.default_middle_steer
            self.steer_limit = self.default_steer_limit
            self.throttle_limit = self.default_throttle_limit
            self.delay_time = self.default_delay_time

    def save_calibration(self):
        calibration = {
            'middle_steer': self.middle_steer,
            'steer_limit': self.steer_limit,
            'throttle_limit': self.throttle_limit,
            'delay_time': self.delay_time
        }
        with open('calibration.json', 'w') as f:
            json.dump(calibration, f)

    def configure_grid(self):
        self.master.columnconfigure(0, weight=1)
        self.master.columnconfigure(1, weight=3)
        for i in range(5):  # Rows 0-4
            self.master.rowconfigure(i, weight=1)
        self.master.rowconfigure(4, weight=2)  # Give more space to the calibration panel

    def get_serial_ports(self):
        return [port.device for port in serial.tools.list_ports.comports()]

    def toggle_connection(self):
        if self.is_connected:
            self.disconnect()
        else:
            self.connect()

    def connect(self):
        port = self.port_combo.get()
        baudrate = int(self.baudrate_combo.get())
        try:
            self.serial_port = serial.Serial(port, baudrate=baudrate, timeout=1)
            self.is_connected = True
            self.connect_button.config(text="Disconnect")
            self.port_combo.config(state="disabled")
            self.baudrate_combo.config(state="disabled")
            threading.Thread(target=self.send_commands, daemon=True).start()
        except serial.SerialException as e:
            print(f"Error connecting to {port}: {e}")

    def disconnect(self):
        if self.serial_port:
            self.is_connected = False
            self.send_brake_sequence()
            self.serial_port.close()
            self.serial_port = None
            self.connect_button.config(text="Connect")
            self.port_combo.config(state="normal")
            self.baudrate_combo.config(state="normal")

    def send_commands(self):
        while self.is_connected:
            cmd = self.calibrate_command(COMMANDS[self.current_command])
            self.serial_port.write(bytearray(cmd))
            # self.serial_port.flush()
            self.serial_port.reset_input_buffer()
            print(f"sending {cmd}")
            time.sleep(self.delay_time)

    def calibrate_command(self, cmd):
        calibrated_cmd = cmd.copy()
        
        # Calibrate steering
        # If not centered
        steer_value = cmd[1]
        centered_steer = steer_value - 128
        calibrated_steer = round(self.middle_steer + (centered_steer * self.steer_limit/self.default_steer_limit ))
        calibrated_cmd[1] = max(0, min(255, calibrated_steer))

        # Calibrate throttle
        throttle = cmd[2]
        calibrated_throttle = round(min(255, max(0, throttle * self.throttle_limit/self.default_throttle_limit )))
        calibrated_cmd[2] = calibrated_throttle

        return calibrated_cmd

    def set_command(self, command):
        self.current_command = command
        self.update_active_button(command)

    def update_active_button(self, active_command):
        for cmd, button in self.buttons.items():
            if cmd == active_command:
                button.config(style='Active.TButton')
            else:
                button.config(style='Large.TButton')

    def send_brake_sequence(self):
        if self.serial_port:
            cmd = self.calibrate_command(COMMANDS['brake'])
            self.serial_port.write(bytearray(cmd))
            # self.serial_port.flush()
            self.serial_port.reset_input_buffer()
            time.sleep(3)
            cmd = self.calibrate_command(COMMANDS['ready'])
            self.serial_port.write((bytearray(cmd)))
            # self.serial_port.flush()
            self.serial_port.reset_input_buffer()

    def on_closing(self):
        self.disconnect()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CarControlApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()