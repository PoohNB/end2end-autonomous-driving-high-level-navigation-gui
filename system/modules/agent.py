import os
import json
from wrapper.utils import init_component
import stable_baselines3
import sb3_contrib
import torch
import gc
import traceback

class Agent:

    def __init__(self,model_path):
        self.observer = None
        self.action_wrapper = None
        self.model = None

        try:
            model_dir = os.path.dirname(model_path)

            # load component
            config_path = os.path.join(model_dir, "config.json")
            with open(config_path, 'rb') as file:
                loaded_config = json.load(file)
            loaded_env_config = loaded_config['env']
            self.n_cam = len(loaded_env_config['env_config']['cam_config_list'])
            self.delta_frame = loaded_env_config['env_config']['carla_setting']['delta_frame']
            cam_attribute = loaded_env_config['env_config']['cam_config_list'][0]['attribute']
            self.resolution = (cam_attribute['image_size_x'],cam_attribute['image_size_y'])
            if "Vae" in loaded_env_config['observer_config']['name']:
                if loaded_env_config.get('observer_config', {}).get('config').get('vae_decoder_config') is None:
                    vencoder_model_path = loaded_env_config['observer_config']['config']['vae_encoder_config']['model_path']
                    vencoder_latent_dims = loaded_env_config['observer_config']['config']['vae_encoder_config']['latent_dims']
                    decoder_model_path = os.path.join(os.path.dirname(vencoder_model_path), "decoder_model.pth")

                    if os.path.exists(decoder_model_path):

                        loaded_env_config['observer_config']['config']['vae_decoder_config'] = {
                            'model_path': decoder_model_path,
                            'latent_dims': vencoder_latent_dims
                        }

            # init deep learning model for observation and fuction for postprocessing action
            self.observer,self.action_wrapper=init_component(loaded_env_config)

            # load agent
            algo_name = loaded_config['algorithm']['method']

            try:
                algo = getattr(stable_baselines3, algo_name)
            except AttributeError:
                try:
                    algo = getattr(sb3_contrib, algo_name)
                except AttributeError:
                    raise ValueError(f"{algo_name} is not found in both stable_baselines3 and sb3_contrib")
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.model = algo.load(model_path, device=self.device)

        except Exception as e:
            tb = traceback.format_exc()
            print(f"Error in loop thread: {e}")
            print(f"Traceback details:\n{tb}")
            self.close()
            raise

    def reset(self,list_images):
        self.observer.reset(list_images)
        self.previous_action = [0,0]
    
    def render(self):
        displays = self.observer.get_renders()
        return displays

    def __call__(self,list_images,maneuver):

        """
        input: list_image:list of raw images
               action:agent predicted action
               maneuver: high level command 0,1,2 for guide the agent
        return: action (steer,throttle,brake)
        """

        obs = self.observer.step(
                            imgs = list_images,
                            act=self.previous_action,
                            maneuver=maneuver)
        
        self.previous_action,_ = self.model.predict(obs.reshape((1,self.observer.len_latent)) ,deterministic=True)

        action = self.action_wrapper(self.previous_action)

        return action
    
    def close(self):
        if self.observer is not None:
            del self.observer
            self.observer = None
        if self.action_wrapper is not None:
            del self.action_wrapper
            self.action_wrapper = None
        if self.model is not None:
            del self.model
            self.model = None
        torch.cuda.empty_cache()
        gc.collect() 


