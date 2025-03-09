import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets

from CNNVae import VariationalAutoencoder
# Hyper-parameters

model_path = 'autoencoder/model/vae32_v2/best'
data_dir = 'autoencoder/dataset/'

variables = {}
with open(os.path.join(model_path.rsplit('/',1)[0],'vae.py'), 'r') as file:
    exec(file.read(), variables)
LATENT_SPACE = variables['LATENT_SPACE']

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
save_dir = os.path.join('autoencoder/reconstructed',model_path.rsplit('/',2)[-2])
os.makedirs(save_dir, exist_ok=True)
save_dir = os.path.join(save_dir,str(len(os.listdir(save_dir))))
os.makedirs(save_dir, exist_ok=True)
test_transforms = transforms.Compose([transforms.Resize((245, 245)),
                                        transforms.Grayscale(),
                                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: ((x * 255.0) / 4.0))
                        ])

post_transforms = transforms.Compose([
                        # transforms.Lambda(lambda x: (x *4) / 255.0),
                        transforms.ToPILImage()
                        ])

test_data = datasets.ImageFolder(data_dir+'test', transform=test_transforms)

testloader = torch.utils.data.DataLoader(test_data, batch_size=1)

model = VariationalAutoencoder(latent_dims=LATENT_SPACE).to(device)
model.load(model_path)
model.Reconstructor(dataloader=testloader,
                    post_transforms=post_transforms,
                    save_dir=save_dir)

