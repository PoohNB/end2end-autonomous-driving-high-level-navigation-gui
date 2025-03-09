import cv2
import numpy as np 
import threading
import queue
import logging

import cv2
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

class RGBCams:
    def __init__(self, n_cam, width=1280, height=720):
        self.n_cam = n_cam
        self.width = width
        self.height = height
        self.caps = [None]*n_cam
        self.initialized_index = [None]*n_cam

    def initialize_cameras(self, cam_idx,pos):
        if isinstance(cam_idx,int):
            if cam_idx in self.initialized_index:
                prev_pos = self.initialized_index.index(cam_idx)
                self.release_cameras(prev_pos)

            cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                logging.warning(f"Camera {cam_idx} failed to initialize")
                cap.release()
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                logging.info(f"Camera {cam_idx} initialized")
                if self.initialized_index[pos] is not None:
                    self.release_cameras(pos)
                self.caps[pos] = cap
                self.initialized_index[pos] = cam_idx


    
    def release_cameras(self, pos):
        if self.initialized_index[pos] is not None:
            self.caps[pos].release()
            self.caps[pos] = None
            self.initialized_index[pos] = None
            logging.info(f"removed cam position {pos}")
            

    def read_images(self):
        imgs = []
        for i,cap in enumerate(self.caps):
            if cap is not None:
                ret, frame = cap.read()
                if ret:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    imgs.append(rgb_frame)
                else:
                    logging.warning("Failed to read frame")
                    imgs.append(np.zeros((self.height, self.width, 3), dtype=np.uint8))  # Placeholder for failed capture
                    
            else:
                # print(f"cam index {i} are blanking")
                imgs.append(np.zeros((self.height, self.width, 3), dtype=np.uint8))  # Placeholder for failed capture
        self.rgb_image = imgs
        return self.rgb_image

    def close(self):
        for cap in self.caps:
            if cap is not None:
                cap.release()
        self.caps = [None]*self.n_cam
        self.initialized_index = [None]*self.n_cam
        logging.info("All cameras released and lists cleared")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()





class Recorder:

    def __init__(self,video_path):
        self.recorded = queue.Queue()
        self.video_path = video_path
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = None
        self.is_recording = False
        self.saving_thread = threading.Thread(target=self.save_video_real_time)
        self.saving_thread.daemon = True
        self.saving_thread.start()
        self.rgb_image = []


    def add_images(self):
        if self.rgb_image:
            self.recorded.put(self.rgb_image)
        else:
            print("Warning: No images to add")

    def start_recording(self, fps=20.0):
        self.is_recording = True
        first_image_set = self.recorded.get()
        height, width, _ = first_image_set[0].shape
        self.recorded.task_done()
        n_cams = len(self.caps)
        grid_size = int(np.ceil(np.sqrt(n_cams)))
        frame_height = height // grid_size
        frame_width = width // grid_size
        output_height = frame_height * grid_size
        output_width = frame_width * grid_size

        self.out = cv2.VideoWriter(self.video_path, self.fourcc, fps, (output_width, output_height))

    def stop_recording(self):
        self.is_recording = False
        self.saving_thread.join()
        if self.out:
            self.out.release()
        print(f"Video saved at {self.video_path}")

    def save_video_real_time(self):
        while self.is_recording or not self.recorded.empty():
            if not self.recorded.empty():
                frame_set = self.recorded.get()
                height, width, _ = frame_set[0].shape
                n_cams = len(frame_set)
                grid_size = int(np.ceil(np.sqrt(n_cams)))
                frame_height = height // grid_size
                frame_width = width // grid_size
                output_height = frame_height * grid_size
                output_width = frame_width * grid_size

                combined_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                for i, frame in enumerate(frame_set):
                    resized_frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (frame_width, frame_height))
                    row = i // grid_size
                    col = i % grid_size
                    combined_frame[row*frame_height:(row+1)*frame_height, col*frame_width:(col+1)*frame_width] = resized_frame

                self.out.write(combined_frame)
                self.recorded.task_done()
