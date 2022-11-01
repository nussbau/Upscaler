import torch
import torch.nn as nn
from os import listdir
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np


class Discriminator(nn.Module):

    def __init__(self, conv_dim=32):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, conv_dim, 3, padding="same"),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.AvgPool2d((2, 2)),
   
            nn.Conv2d(conv_dim, conv_dim * 2, 3, padding="same"),
            nn.BatchNorm2d(conv_dim * 2, momentum=0.7),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.AvgPool2d((2, 2)),
            
            nn.Conv2d(conv_dim * 2, conv_dim * 4, 3, padding="same"),
            nn.BatchNorm2d(conv_dim * 4, momentum=0.7),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.AvgPool2d((2, 2)),
            
            nn.Conv2d(conv_dim * 4, conv_dim * 8, 3, padding="same"),
            nn.BatchNorm2d(conv_dim * 8, momentum=0.7),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.AvgPool2d((2, 2)),

            nn.Conv2d(conv_dim * 8, conv_dim * 16, 3, padding="same"),
            nn.BatchNorm2d(conv_dim * 16, momentum=0.7),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.AvgPool2d((2, 2)),

            nn.Conv2d(conv_dim * 16, conv_dim * 32, 3, padding="same"),
            nn.BatchNorm2d(conv_dim * 32, momentum=0.7),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.AvgPool2d((2, 2)),
          
            nn.Flatten(),
            nn.Linear(in_features=32*conv_dim*16, out_features=128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )


    def forward(self, x):
        return self.main(x)


class Generator(nn.Module):
    
    def __init__(self, conv_dim=64):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(3, conv_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 2),
            nn.ReLU(True),
           
            nn.ConvTranspose2d( conv_dim * 2, conv_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(True),
          
            nn.ConvTranspose2d(conv_dim, 3, 4, 1, 1, bias=False),
            nn.Tanh()
        )


    def forward(self, x):
        return self.main(x)

def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)



class MonetPictures(Dataset):
    def __init__(self) -> None:
        imagePaths = listdir("monet_jpg")
        self.length = len(imagePaths)
        self.images = np.zeros((self.length, 256, 256, 3))
        self.resized_images = np.zeros((self.length, 64, 64, 3))
        for i in range(self.length):
            self.images[i] = np.array(Image.open("monet_jpg/" + imagePaths[i]))
            self.resized_images[i] = np.array(Image.open("monet_jpg/" + imagePaths[i]).resize((64, 64)))
        self.mean = np.mean(self.images, (0, 1, 2))
        self.stDev = np.std(self.images, (0, 1, 2))
        self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize(
                        [self.mean[i] for i in range(3)], [self.stDev[i] for i in range(3)]
                    ),                  
            ])

    def __getitem__(self, index):
        return self.transform(self.images[index]), self.transform(self.resized_images[index])

    def __len__(self):
        return self.length

    def getNorm(self):
        return self.mean, self.stDev
