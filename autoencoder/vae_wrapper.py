from autoencoder.CNNVae import Encoder,Decoder
import torch
import cv2
import time
from torchvision import transforms
import numpy as np
from PIL import Image


model_path = '../autoencoder/model/vae32/best/var_encoder_model.pth'



class VencoderWrapper:

    """
    model_path: pth path
    """
    
    def __init__(self,model_path,latent_dims,custom_process=None):
        print('===== Encoder =====')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"using {self.device}")
        self.latent_dims = latent_dims 
        self.model = Encoder(latent_dims=self.latent_dims)
        self.model.load(model_path)
        self.model.eval()

        default_transforms = transforms.Compose([transforms.Resize((245, 245)),
                                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: ((x * 255.0) / 4.0))
                        ])
        self.processor = custom_process or default_transforms

        self.warmup()

    def warmup(self,image_shape=(512,1024)):
        # warmup
        dummy_images = np.random.randint(0, 4, image_shape, dtype=np.uint8)
        self(dummy_images)
        test_times = 20
        st = time.time()
        for _ in range(test_times):
            self(dummy_images)
        print(f"inference time :{(time.time()-st)/test_times:.6f}")
 
        
    def __call__(self,image):
        
        if isinstance(image,np.ndarray):
            image = Image.fromarray(image)
            preprocessed = self.processor(image).unsqueeze(0)
        elif isinstance(image,list):
            preprocessed = torch.stack([self.processor(Image.fromarray(img)) for img in image])

        with torch.no_grad():

            latent = self.model(preprocessed)

        return latent


class DecoderWrapper():

    """
    model_path: pth path
    """
    
    def __init__(self,model_path,latent_dims):
        print('===== Decoder =====')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"using {self.device}")

        self.model = Decoder(latent_dims=latent_dims)
        self.model.load(model_path)
        self.model.eval()

        self.warmup()

    def warmup(self,latent_size = (1,32)):
        # warmup
        dummy = torch.rand(latent_size).to(self.device)
        self(dummy)
        test_times = 20
        st = time.time()
        for _ in range(test_times):
            self(dummy)
        print(f"inference time :{(time.time()-st)/test_times:.6f}")
 
        
    def __call__(self,latents,post_process=True):
        
 
        with torch.no_grad():

            images = self.model(latents)
        
        if post_process:
            images = [cv2.cvtColor((img.squeeze(0).numpy()*255).astype(np.uint8),cv2.COLOR_GRAY2RGB) for img in images.cpu()]

        return images




        

