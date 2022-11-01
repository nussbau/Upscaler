import torch.optim as optim
import torch
import GAN
from torch.utils.data import DataLoader

#Finds if computer has cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Parameters
learningRate = 0.001
batchSize = 5
features_d = 128
features_g = 128
numEpochs = 100000
discIterations = 5
weightClip = 0.01


#Loads dataset
images = GAN.MonetPictures()
imageLoader = DataLoader(images, batch_size=batchSize, shuffle=True)
mean, std = images.getNorm()


#Loads generator and discriminator
gen = GAN.Generator(features_g).to(device)
disc = GAN.Discriminator(features_d).to(device)
optimizer_gen = optim.Adam(gen.parameters(), lr=learningRate)
optimizer_disc = optim.Adam(disc.parameters(), lr=learningRate)
GAN.initialize_weights(gen)
GAN.initialize_weights(disc)

try:
    genCP = torch.load("modelGen")
    discCP = torch.load("modelDisc")
except:
    genCP = None
    discCP = None

if genCP is not None:
    gen.load_state_dict(genCP["model_state_dict"])
    optimizer_gen.load_state_dict(genCP["optimizer_state_dict"])
    disc.load_state_dict(discCP["model_state_dict"])
    optimizer_disc.load_state_dict(discCP["optimizer_state_dict"])


#Training
for epoch in range(0, numEpochs):
    for large_pic, small_pic in iter(imageLoader):

        large_pic = large_pic.to(device)
        #Train Discriminator 
        for _ in range(discIterations):
            fake = gen(small_pic.to(device))
            disc_real = disc(large_pic).reshape(-1)
            disc_fake = disc(fake).reshape(-1)
            lossD = -(torch.mean(disc_real)-torch.mean(disc_fake))
            disc.zero_grad()
            lossD.backward(retain_graph=True)
            optimizer_disc.step()

        #Puts a cap on the weights
        for p in disc.parameters():
            p.data.clamp_(-weightClip, weightClip)

        #Trains the Generator
        out = disc(fake).reshape(-1)
        lossG = -torch.mean(out)
        gen.zero_grad()
        lossG.backward()
        optimizer_gen.step()


    

    #Saves the networks
    if epoch % 25 == 0:
        torch.save({
                'model_state_dict': gen.state_dict(),
                'optimizer_state_dict': optimizer_gen.state_dict(),
                }, "modelGen")
        torch.save({
                'model_state_dict': disc.state_dict(),
                'optimizer_state_dict': optimizer_disc.state_dict(),
                }, "modelDisc")
        torch.save(gen, "model")
    
    print(f"Current Epoch: {epoch}, Loss D: {lossD:.4f}, loss G: {lossG:.4f}\r", end="\n")

