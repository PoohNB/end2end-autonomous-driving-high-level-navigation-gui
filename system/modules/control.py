import serial
import time


# self.throttle_slider.valueChanged.connect(lambda value: function('throttle', value))
# self.steer_limit_slider.valueChanged.connect(lambda value: function('steer_limit', value))
# self.steer_center_slider.valueChanged.connect(lambda value: function('steer_center', value))
# self.delay_slider.valueChanged.connect(lambda value: function('delay', value))


class CalibrationBox:

    def __init__(self):
        self.throttle_limit = 1
        self.max_steer = 1
        self.steer_center = 0
        self.delay = 0
        self.default_throttle_limit = 1
        self.default_max_steer = 1

    def apply_setting(self,param,value):
        print("apply value",param,value/100)
        if param == 'throttle':
            self.throttle_limit = value/100
        elif param == 'steer_limit':
            self.max_steer = value/100
        elif param == 'steer_center':
            self.steer_center = value/100
        elif param == 'delay':
            self.delay = value/100

    def apply_default_limit(self,default_max_steer,default_throttle_limit):
        self.default_throttle_limit = default_throttle_limit
        self.default_max_steer = default_max_steer

    def __call__(self,steer, throttle,proc_time=100):

        cal_steer= min(1,max(-1,self.steer_center+(steer * (self.max_steer/ self.default_max_steer))))
        cal_throttle=min(1,max(0,throttle *(self.throttle_limit/self.default_throttle_limit)))
        if self.delay > proc_time:
            time.sleep(self.delay-proc_time)
        return cal_steer,cal_throttle

class SerialControl:

    def __init__(self, port=None, baud=115200):
        self.ser = None
        self.set_port(port, baud)

    def set_port(self, port, baud=115200):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Previous serial port closed.")
        try:
            self.ser = serial.Serial(port, baud)
            print(f"Serial port {port} opened.")
        except serial.SerialException as e:
            print(f"Serial error: {e}")
            self.ser = None
        
    def send(self, steer, throttle, brake=False):
        
        command = self.mc_cmd(steer, throttle, brake)
        if self.ser and self.ser.is_open:
            self.ser.write(bytearray(command))
            self.ser.flush()  # Ensures the data is sent immediately
            self.ser.reset_input_buffer()  # Clears the buffer after sending the command
            print(f"Sent command: {command}")
        else:
            print(f"Send dummy command: {command}")
        return command

    @staticmethod
    def mc_cmd(steer, throttle, brake=False):
        if not (-1 <= steer <= 1):
            raise ValueError("Steer must be between -1 and 1")
        if not (0 <= throttle <= 1):
            raise ValueError("Throttle must be between 0 and 1")
        
        scaled_steer = round(((steer + 1) / 2) * 255)
        scaled_throttle = round(throttle * 255)

        if brake:
            scaled_throttle = 0
            brake_signal = 255
        else:
            brake_signal = 0

        return [36,int(scaled_steer), int(scaled_throttle), int(brake_signal),0,64]

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial port closed.")

