import time

from .modules.agent import Agent
from .modules.control import SerialControl,CalibrationBox
from .modules.perception import RGBCams

class AVbot:

    def __init__(self):
        self.cams = None
        self.ctrl = SerialControl(None)
        self.calib = CalibrationBox()
        self.agent = None
        self.activated_ctrl = False
        self.delta_frame=0.4
        self.calibrating = 0

    def apply_calibrating_mode(self,value):
        self.calibrating = value

    def init_agent(self,model_path):
        self.agent = Agent(model_path)
        self.cams = RGBCams(self.agent.n_cam,*self.agent.resolution)
        self.delta_frame = self.agent.delta_frame
        # self.delta_frame=0.4

    def set_cameras(self,cam_idx,pos):
        if self.cams is not None:
            self.cams.initialize_cameras(cam_idx,pos)

    def set_control(self,port):       
        self.ctrl.set_port(port)

    def _ensure_initialized(self):
        if self.cams is None:
            raise RuntimeError("Cameras not initialized. Call init_camera() before using this method.")
       
    def reset(self):
        self._ensure_initialized
        self.images = self.cams.read_images()
        self.agent.reset(self.images)
        steer ,throttle = self.calib(0,0)
        self.ctrl.send(steer=steer,throttle=throttle)
        time.sleep(0.4)
    
    def start(self):
        self.reset()
        self.activated_ctrl = True

    def step(self,maneuver):
        steer,throttle,proc_time,ctrled_time = 0,0,-1,-1
        if self.agent is not None and self.cams is not None:
            st_time = time.time()
            self.images = self.cams.read_images()
            if self.calibrating == 0:
                steer,throttle,brake = self.agent(list_images=self.images,maneuver=maneuver)
            elif self.calibrating ==1:
                steer,throttle,brake = 0,0.35,False
            elif self.calibrating ==2:
                steer,throttle,brake = 0.6,0.3,False
            elif self.calibrating ==3:
                steer,throttle,brake = -0.6,0.3,False
            else:
                steer,throttle,brake = 0,0.35,False

            proc_time = (time.time() - st_time)

            if proc_time < self.delta_frame:
                time.sleep(self.delta_frame - proc_time) 

            steer,throttle = self.calib(steer=steer,throttle=throttle,proc_time=max(self.delta_frame,proc_time))

            ctrled_time = (time.time() - st_time)

            
            if self.activated_ctrl:
                self.ctrl.send(steer=steer,throttle=throttle)
        else:
            print("agent or cams is None")
            
        return steer,throttle,proc_time,ctrled_time

    def get_vision(self):

        agent_vision = self.agent.render() 

        return [self.images]+agent_vision

    def stop(self):
        print("stoping..")
        self.activated_ctrl= False
        steer,throttle = self.calib(steer=0,throttle=0)
        self.ctrl.send(steer=steer,throttle=throttle,brake=True)
        time.sleep(2)
        self.ctrl.send(steer=steer,throttle=throttle)
        print("complete..")

    def close(self):
        # self.stop()
        self.ctrl.close()
        if self.cams is not None:
            self.cams.close()
        if self.agent is not None:
            self.agent.close()
        self.cams = None
        self.agent = None

