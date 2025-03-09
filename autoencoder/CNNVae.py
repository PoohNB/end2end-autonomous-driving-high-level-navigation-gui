# import os
import torch
import torch.nn as nn
# import numpy as np
# from tqdm import tqdm
# from torchvision import transforms
# from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



class Encoder(nn.Module):

    """
    config : [con(input chanel,kernel size,stride,pad),..] = [layer1,..]
    
    """

    def __init__(self, 
                 latent_dims=64,
                 ):  
        super().__init__()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.encoder_layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2),  # 79, 39
            nn.LeakyReLU())

        self.encoder_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 40, 20
            nn.BatchNorm2d(64),
            nn.LeakyReLU())


        self.encoder_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, stride=2),  # 19, 9
            nn.LeakyReLU())

        self.encoder_layer4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 9, 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU())

        self.linear = nn.Sequential(
            nn.Linear(15*15*256, 1024),
            nn.LeakyReLU())

        self.mu = nn.Linear(1024, latent_dims)
        self.sigma = nn.Linear(1024, latent_dims)
        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.to(device)
        # self.N.scale = self.N.scale.to(device)
        self.kl = 0

        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        x = self.encoder_layer4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        mu =  self.mu(x)
        sigma = torch.exp(self.sigma(x))
        z = mu + sigma*self.N.sample(mu.shape).to(self.device)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 0.5).sum()
        return z

    def save(self,path):
        torch.save(self.state_dict(), path)

    def load(self,path):
        self.load_state_dict(torch.load(path, map_location=torch.device(self.device), weights_only=True))

class Decoder(nn.Module):
    
    def __init__(self, 
                 latent_dims):
        super().__init__()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dims, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 15 * 15 * 256),
            nn.LeakyReLU()
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256,15,15))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2,padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 5,  stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2,padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, 5, stride=2),
            nn.Sigmoid())
        
        self.to(self.device)
        
    def forward(self, x):
        x = self.decoder_linear(x)
        x = self.unflatten(x)
        x = self.decoder(x)
        return x

    def save(self,path):
        torch.save(self.state_dict(), path)

    def load(self,path):
        self.load_state_dict(torch.load(path, map_location=torch.device(self.device), weights_only=True))


# class VariationalAutoencoder(nn.Module):
#     def __init__(self, latent_dims):
#         super(VariationalAutoencoder, self).__init__()
#         self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#         self.encoder = Encoder(latent_dims)
#         self.decoder = Decoder(latent_dims)
#         self.to(self.device)
#         print('using: ',self.device)

#     def forward(self, x):
#         x = x.to(self.device)
#         z = self.encoder(x)
#         return self.decoder(z)
    
#     def save(self,path):
#         os.makedirs(path, exist_ok=True)
#         torch.save(self.state_dict(),os.path.join(path,'var_autoencoder.pth'))
#         self.encoder.save(os.path.join(path,'var_encoder_model.pth'))
#         self.decoder.save(os.path.join(path,'decoder_model.pth'))
#         print(f"saved in {path}")
    
#     def load(self,path):
#         os.makedirs(path, exist_ok=True)
#         self.load_state_dict(torch.load(os.path.join(path,'var_autoencoder.pth'), map_location=torch.device(self.device), weights_only=True))
#         self.encoder.load(os.path.join(path,'var_encoder_model.pth'))
#         self.decoder.load(os.path.join(path,'decoder_model.pth'))
#         print(f"loaded from {path}")

#     def train_epoch(self, trainloader, optim):
#         self.train()
#         train_loss = 0.0
#         for(x, _) in tqdm(trainloader,desc='training'):
#             # Move tensor to the proper device
#             x = x.to(self.device)
#             x_hat = self(x)
#             loss = ((x - x_hat)**2).sum() + self.encoder.kl
#             optim.zero_grad(set_to_none=True)
#             # for param in self.parameters():
#             #     param.grad = None
#             loss.backward()
#             optim.step()
#             train_loss+=loss.item()
#         return train_loss / len(trainloader.dataset)


#     def val_epoch(self, valloader):
#         # Set evaluation mode for encoder and decoder
#         self.eval()
#         val_loss = 0.0
#         with torch.no_grad(): # No need to track the gradients
#             for x, _ in tqdm(valloader,desc='evaluating'):
#                 # Move tensor to the proper device
#                 x = x.to(self.device)
#                 # Encode data
#                 encoded_data = self.encoder(x)
#                 # Decode data
#                 x_hat = self(x)
#                 loss = ((x - x_hat)**2).sum() + self.encoder.kl
#                 val_loss += loss.item()

#         return val_loss / len(valloader.dataset)
    
#     def Trainer(self,
#                 trainloader,
#                 valloader,
#                 optim,
#                 num_epochs = 50,
#                 log_path = "autoencoder/runs/experiment0",
#                 checkpoint_path = 'autoencoder/model/check0',
#                 seed=1234):  
              

#         torch.manual_seed(seed)
#         if torch.cuda.is_available():
#             torch.backends.cudnn.benchmark = False
#             torch.backends.cudnn.deterministic = True

#         best_loss = np.inf

#         writer = SummaryWriter(log_path)

#         for epoch in range(num_epochs):
#             train_loss = self.train_epoch(trainloader, optim)
#             writer.add_scalar("Training Loss/epoch", train_loss, epoch+1)
#             val_loss = self.val_epoch(valloader)
#             writer.add_scalar("Validation Loss/epoch", val_loss, epoch+1)
#             if val_loss <= best_loss:
#                 best_loss = val_loss
#                 self.save(os.path.join(checkpoint_path,'best')) 
#             print('\nEPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))

#         self.save(os.path.join(checkpoint_path,'last'))


#     def Reconstructor(self,dataloader,post_transforms = transforms.ToPILImage(),save_dir='autoencoder/reconstructed'):
#         os.makedirs(save_dir, exist_ok=True)
#         self.eval()
#         count = 1
#         with torch.no_grad(): # No need to track the gradients
#             for x, _ in dataloader:
#                 # Move tensor to the proper device
#                 x = x.to(self.device)
#                 # Decode data
#                 x_hat = self(x)
#                 x_hat = x_hat.cpu()
#                 x_hat = x_hat.squeeze(0)

#                 # convert the tensor to PIL image using above transform
#                 img =  post_transforms(x_hat)

#                 image_filename = str(count) +'.png'
#                 img.save(os.path.join(save_dir,image_filename))
#                 count +=1

#         print(f"saved in {save_dir}")

        

        

        
