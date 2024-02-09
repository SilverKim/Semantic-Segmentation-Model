import torch
import torch.nn as nn
import torch.nn.functional as F
latent = 512
class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,

    def forward(self, x):
        return x.view(*self.shape)

class Colon(nn.Module):
    def __init__(self):

        super(Colon, self).__init__()

        self.encoder_layers = nn.Sequential(  ###### Colon
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(512),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(512),
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(1024),
                nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(1024),
                nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(2048),
                nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(2048),
                nn.Flatten(),
                nn.Linear(2048 * 2 * 2, latent * 2),
                nn.ReLU(),
                nn.Linear(latent * 2, latent * 2)
            )

        self.decoder_layers = nn.Sequential(  ###### Colon
                nn.Linear(latent, 2048 * 2 * 2),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(2048 * 2 * 2),
                View((-1, 2048, 2, 2)),

                nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(1024),
                nn.ConvTranspose2d(1024, 1024, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),

                nn.BatchNorm2d(1024),  ##16x16

                nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(512),
                nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(512),  ##32x32

                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),  ##64x64

                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),  ##8x8

                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),  ##16x16

                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),  ##32x32

                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),

                nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),

            )
    
    def forward(self, x):
        x = self.encoder_layers(x)
        mu, logvar = x[:, :latent], x[:, latent:]
        z = self.reparameterize(mu, logvar)
        return self.decoder_layers(z), mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create an instance of the model
model = Colon().to(device)

# create a random input tensor of shape 3x256x256
x = torch.randn(3, 3, 256, 256).to(device)

# forward pass
out, mu, logvar = model(x)

# print the output shape
print(out.shape)
