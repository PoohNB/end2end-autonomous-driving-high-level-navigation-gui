# observer is for convert the input from environment to the state

from collections import deque
import numpy as np

# semantic segmentation
from segmentation import seg_hf

# variational autoencoder
from autoencoder.vae_wrapper import VencoderWrapper,DecoderWrapper

class RawObserver:
    """
    class for test only can't work with sb3

    """

    def __init__(self,image_size):
        
        self.image_size = image_size
    
    def get_state(self):
        return 
    
    def reset(self,imgs):

        
        return imgs
        
    def step(self,**arg):


        return arg["imgs"]
    
class SegVaeObserver:
    """
    class for convert image observation to state space

    seg_model_config = HFsegwrapper module
    vae_model = VencoderWrapper

    """


    def __init__(self,                
                seg_model_config,
                vae_encoder_config,
                vae_decoder_config=None):
        
        # select type of seg model and config
        seg_class = getattr(seg_hf,seg_model_config['name'])
        self.seg = seg_class(**seg_model_config['config'])
        self.vae_encoder = VencoderWrapper(**vae_encoder_config)
        self.vae_decoder = DecoderWrapper(**vae_decoder_config) if vae_decoder_config is not None else None
        self.len_latent = self.vae_encoder.latent_dims
        self.cam_adjust = {}
    def reset(self,imgs):

        raise NotImplementedError("Method 'reset' must be implemented in subclasses")
        
    def step(self,**arg):

        raise NotImplementedError("Method 'step' must be implemented in subclasses")
    
    def apply_adjust_params(self,cam_adjust):
        self.cam_adjust = cam_adjust

    
    def adjust_cam_angle(image, rows):
        height, width, channels = image.shape
        
        # Limit rows to half the image height
        rows = max(min(rows, height // 2), -height // 2)
        
        if rows > 0:
            # Add black rows at the top and remove from bottom
            new_image = np.zeros_like(image)
            new_image[rows:, :, :] = image[:-rows, :, :]
        elif rows < 0:
            # Remove rows from top and add black rows at bottom
            new_image = np.zeros_like(image)
            new_image[:height+rows, :, :] = image[-rows:, :, :]
        else:
            # No change if rows is 0
            new_image = image.copy()
        
        return new_image

    def get_latent(self,imgs):

        self.pred_segs = self.seg(imgs)
        for k,v in self.cam_adjust.items():
            if k < len(self.pred_segs):
                self.pred_segs[k] = self.adjust_cam_angle(self.pred_segs[k],v)
        self.latents = self.vae_encoder(self.pred_segs)
        cat_latent = self.latents.flatten().cpu().numpy()

        return cat_latent
    
    def get_renders(self):
        obsr = []
        obsr.append(self.seg.get_seg_images(self.pred_segs))
        if self.vae_decoder is not None:
            obsr.append(self.vae_decoder(self.latents))
        return obsr
    

#==

class SegVaeActManvHistObserver(SegVaeObserver):

    def __init__(self,                 
                 seg_model_config,
                 vae_encoder_config,
                 num_img_input,
                 act_num=2,
                 maneuver_num=1,
                 hist_len = 8,
                 skip_frame=0,
                 vae_decoder_config=None):

        
        super().__init__(seg_model_config=seg_model_config,
                         vae_encoder_config=vae_encoder_config,
                         vae_decoder_config=vae_decoder_config)

        self.latent_space = self.vae_encoder.latent_dims
        self.skip_frame = skip_frame
        self.hist_len = hist_len
        self.act_num = act_num
        self.maneuver_num = maneuver_num
        self.len_latent = (self.latent_space*num_img_input+self.act_num+self.maneuver_num)*self.hist_len
        self.history_state = deque(maxlen=self.hist_len*(self.skip_frame+1)-self.skip_frame)

    def get_state(self):
        state = np.concatenate([self.history_state[i] for i in range(len(self.history_state)) if i%(self.skip_frame+1)==0])
        return state
    
    def reset(self,imgs):

        cat_latent = self.get_latent(imgs)
        observation = np.concatenate((cat_latent, [0]*self.act_num,[0]*self.maneuver_num), axis=-1)
        self.history_state.extend([observation]*((self.hist_len*(self.skip_frame+1))-self.skip_frame))

        return self.get_state()
        
    def step(self,**arg):

        imgs = arg["imgs"]
        act = arg["act"]
        manv= arg['maneuver']
        
        cat_latent = self.get_latent(imgs)
        observation = np.concatenate((cat_latent,act,manv), axis=-1)
        self.history_state.append(observation)

        return self.get_state()

        

class ClipObserver:

    def __init__(self,clip_model,split=3):
        self.split = split
        self.model = clip_model

    def gym_obs(self):

        pass

    def get_state(self):

        pass

    def reset(self):

        pass

    def step(self):

        pass

    def get_renders(self):
        return []



