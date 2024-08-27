import os

import torch
import torchvision
from torch import nn
from torch import optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt

import numpy as np

from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory



#Generator Class
class Generator(nn.Module):
    def __init__(self,Gen_Input):
        super(Generator, self).__init__()

        self.conv128 = nn.Sequential(
            nn.ConvTranspose2d(Gen_Input, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False)
            )

        self.conv64 = nn.Sequential(
            nn.ConvTranspose2d(Gen_Input, 512, 4, 1, 0, bias=False),
            )
                
        self.conv = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, image_size, input):
        if image_size == 64: output = self.conv64(input)
        else: output = self.conv128(input)
        output = self.conv(output)
        return output

#Discriminator Class
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.25),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.25),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.Dropout2d(0.25),
            nn.LeakyReLU(0.2, inplace=True),
      
        )

        self.conv128 = nn.Sequential(
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.Dropout2d(0.25),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            )

        self.conv64 = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()

            )

    def forward(self, image_size, input):
        output = self.conv(input)
        if image_size == 64: output = self.conv64(output)
        else: output = self.conv128(output)
        return output
        
    
    
class GAN():
    def __init__(self, Gen_Input, epoch_val, batch_size, image_size, device, dataloader):

        self.netG = Generator(Gen_Input).to(device)
        self.netG.apply(self.weights_init)
        self.netD = Discriminator().to(device)
        self.netD.apply(self.weights_init)
        self.epoch_val = epoch_val
        self.batch_size = batch_size
        self.image_size = image_size
        self.device = device
        self.dataloader = dataloader
        self.criterion = nn.BCELoss()
        self.Gen_Input = Gen_Input

        self.G_loss = []
        self.D_loss = []
        
        if device == 'cuda':
            self.netG.cuda()
            self.netG = nn.DataParallel(self.netG, list(range(1)))

            self.netD.cuda()
            self.netD = nn.DataParallel(self.netD, list(range(1)))

            self.criterion.cuda()

        self.optimizerD = optim.Adam(self.netD.parameters(),lr=0.0002, betas = (0.5, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(),lr=0.0002, betas = (0.5, 0.999))  

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)          

    def train_loop(self):

        
        for epoch in range(self.epoch_val):
            for batch, (real_samples, _) in enumerate(self.dataloader):
   
                #Discriminator
                real_samples = real_samples.to(self.device)
                self.batch_size = real_samples.shape[0]

                real = torch.ones((real_samples.shape[0],), device=self.device)
                fake = torch.zeros((real_samples.shape[0],), device=self.device)
                
                self.netD.zero_grad()
                pred_real = self.netD(self.image_size, real_samples).view(-1)
                loss_real = self.criterion(pred_real, real)

                
                fake_samples = self.image_gen()
                fake_samples.detach()
                pred_fake = self.netD(self.image_size, fake_samples).view(-1)
                loss_fake = self.criterion(pred_fake, fake)

                lossD = (loss_real + loss_fake) / 2
                lossD.backward()
                self.optimizerD.step()

                #Generator
                self.netG.zero_grad()
                fake_samples = self.image_gen()
                fake_samples.detach()
                pred_fake = self.netD(self.image_size, fake_samples).view(-1)
                lossG = self.criterion(pred_fake, real)
                lossG.backward()
                self.optimizerG.step()


            print('[Epoch: %d/%d] ~ [Generator Loss: %.4f] ~ [Discriminator Loss: %.4f]' % (epoch+1, self.epoch_val, lossG.item(), lossD.item()))  
            self.G_loss.append(lossG.item())
            self.D_loss.append(lossD.item())

    def image_gen(self):
        noise = torch.randn(self.batch_size, self.Gen_Input, 1, 1, device=self.device)
        fakes = self.netG(self.image_size, noise)
        return fakes

    def save_model(self):
        print("Please select directory to store model file")
        try:
            Tk().withdraw()
            file_direc = askdirectory()
            file_name = input("Enter file name: ")
            file_name = file_direc + "/" + file_name + ".pth"
            torch.save(self.netG.state_dict(), file_name)
        except:
                print("Error: Invalid File Name")
        
    def load_model(self):
        print("Please select .pth file from directory")
        while True:
            try:
                Tk().withdraw()
                file_name = askopenfilename()
                self.netG.load_state_dict(torch.load(file_name))
                break
            except:
                print("Error: Invalid File Please Select a Different File")

def select_input(args):
    while True:
        try:
            selectInp = int(input('--'))
            if selectInp <= args and selectInp > 0: break
            else: print("Invalid Input: Incorrect Option Value")
        except ValueError as e:
            print("Invalid Input: Integer Required")
    return selectInp

def value_input():    
    while True:
        try:
            valueInp = int(input('--'))
            break
        except ValueError as e:
            print("Invalid Input: Integer Required")

    return valueInp


def main():
    Gen_Input = 100
    epoch_val = 10
    batch_size = 64
    image_size = 64
    dataName = "GOAT"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    userInp = 0
    print('''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 ______  _______      ______ _______ __   _
 |     \ |       ___ |  ____ |_____| | \  |
 |_____/ |_____      |_____| |     | |  \_|

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ''')

    print('''Operate the program by inputting the number of the corresponding menu option.
When given no options, enter the desired value for the displayed parameter.
If your device is cuda compatible, it will automatically be used\n
Recommended Training Parameters:
Image Size [64] - Batch Size [] - Training Epochs []
Image Size [128] - Batch Size [] - Training Epochs []
''')

    print('''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Select an operation:
1: Train a model
2: Load a model
3: Check current Device
4: Exit''')
    while True:
        userInp = select_input(4)
        if userInp == 1 or userInp == 2: break
        if userInp == 3:
            print("Current device is: %s" % device)
        if userInp == 4:
            print('''Exit program?
1: Yes
2: No''')
            userInp = select_input(2)
            if userInp == 1: sys.exit()
        

    
    print('''Select model image size:
1: 64
2: 128''')
    image_size = select_input(2)
    if image_size == 1: image_size = 64
    else: image_size = 128
    
    if userInp == 1:
        print('''Select dataset for training:
1: Abstract
2: Best of All Time''')
        dataName = select_input(2)
        if dataName == 1: dataName = "Abstract"
        else: dataName = "GOAT"
        
        print('Input batch size: ')
        batch_size = value_input()
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

    dataset = ImageFolder(root=os.path.join("Data", dataName), transform=transform)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=2)

    if userInp == 1:
        print('Input number of training epochs:')
        epoch_val = value_input()
        
        gan = GAN(Gen_Input, epoch_val, batch_size, image_size, device, dataloader)
        print("Processing...")
        gan.train_loop()

        dataVals = [[dataName],[image_size],[batch_size],[epoch_val]]
        rows = ["Dataset","Image Size","Batch Size", "Training Epochs"]

        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Training Loss\nDataset = %s ~ Image Size = %d ~ Batch Size = %d ~ Training Epochs = %d" % (dataName, image_size, batch_size, epoch_val))
        plt.plot(gan.G_loss, label="G")
        plt.plot(gan.D_loss, label="D")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    else:
        gan = GAN(Gen_Input, epoch_val, batch_size, image_size, device, dataloader)
        gan.load_model()

    
    
    while True:
        print('''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[Current Model Image Size: [%d]
Select an option:
1: Save current model
2: Generate Image
3: Load a different model (!Will erase the current model if not saved!)
4: Exit''' % image_size)
    
        userInp = select_input(4)
        if userInp == 1:
            gan.save_model()
        if userInp == 2:
            print('Input number of images to generate: ')
            gan.batch_size = value_input()
            fakes = gan.image_gen()
            ims = torchvision.utils.make_grid(fakes, normalize = True)
            plt.imshow(ims.cpu().numpy().transpose((1,2,0)))
            plt.show()
        if userInp == 3:
            print('''Select image size of loaded model:
1: 64
2: 128''')
            image_size = select_input(2)
            if image_size == 1: image_size = 64
            else: image_size = 128
            gan = GAN(Gen_Input, epoch_val, batch_size, image_size, device, dataloader)
            gan.load_model()
        if userInp == 4:
            print('''Exit program?
1: Yes
2: No''')
            userInp = select_input(2)
            if userInp == 1: sys.exit()

    

if __name__ == '__main__':
    main()
