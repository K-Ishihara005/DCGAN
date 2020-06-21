import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import os
import pickle
import statistics
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse

def load_datasets():
    dataset = datasets.ImageFolder("data/",
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

    batch_size = 128

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0), # 4x4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, 2, 2, 0), #8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 2, 2, 0), #16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 2, 2, 0), #32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 2, 2, 0), #64x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, 2, 2, 0), #128x128
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(32, 32 * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(32 * 2, 32 * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(32 * 4, 32 * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(32 * 8, 32 * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32 * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(32 * 16, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid()


            #nn.Conv2d(3, 32, kernel_size=3, padding=1),
            #nn.BatchNorm2d(32),
            #nn.LeakyReLU(0.2, inplace=True),

            #nn.AvgPool2d(2),
            #nn.Conv2d(32, 64, kernel_size=3, padding=1),
            #nn.BatchNorm2d(64),
            #nn.LeakyReLU(0.2, inplace=True),

            #nn.AvgPool2d(2),
            #nn.Conv2d(64, 128, kernel_size=3, padding=1),
            #nn.BatchNorm2d(128),
            #nn.LeakyReLU(0.2, inplace=True),

            #nn.AvgPool2d(2),
            #nn.Conv2d(128, 256, kernel_size=3, padding=1),
            #nn.BatchNorm2d(256),
            #nn.LeakyReLU(0.2, inplace=True),

            #nn.AvgPool2d(2),
            #nn.Conv2d(256, 512, kernel_size=3, padding=1),
            #nn.BatchNorm2d(512),
            #nn.LeakyReLU(0.2, inplace=True),

            #nn.AvgPool2d(2),
            #nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            #nn.BatchNorm2d(1024),
            #nn.LeakyReLU(0.2, inplace=True),

            #nn.AvgPool2d(4),
            #nn.Conv2d(1024, 1, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze()

def test(args):
    model_G = Generator()
    device = "cuda"
    model_G = nn.DataParallel(model_G)
    model_G = model_G.to(device)
    model_G.load_state_dict(torch.load(args.savefile+"/model/modelG"+str(args.ep-1)+".pth"))
    z = torch.randn(args.batch, 100, 1, 1).to(device)
    if not os.path.exists(args.savefile+"/test"):
        os.mkdir(args.savefile+"/test")
    torchvision.utils.save_image(model_G(z), args.savefile+"/test/ep"+str(args.ep-1)+".png")
    
def train(args):
    device = "cuda"
    model_G, model_D = Generator(), Discriminator()
    model_G, model_D = nn.DataParallel(model_G), nn.DataParallel(model_D)
    model_G, model_D = model_G.to(device), model_D.to(device)

    params_G = torch.optim.Adam(model_G.parameters(),
                lr=0.0002, betas=(0.5, 0.999))
    params_D = torch.optim.Adam(model_D.parameters(),
                lr=0.0002, betas=(0.5, 0.999))

    ones = torch.ones(args.batch).to(device)
    zeros = torch.zeros(args.batch).to(device)
    #loss_f = nn.BCEWithLogitsLoss()
    loss_f = nn.BCELoss()

    result = {}
    result["log_loss_G"] = []
    result["log_loss_D"] = []

    dataset = load_datasets()
    sample_z = torch.randn(20, 100, 1, 1).to(device)
    for i in range(args.ep):
        log_loss_G, log_loss_D = [], []
        for real_img, _ in tqdm(dataset):
            batch_len = len(real_img)

            z = torch.randn(batch_len, 100, 1, 1).to(device)
            fake_img = model_G(z)

            fake_img_tensor = fake_img.detach()

            out = model_D(fake_img)
            loss_G = loss_f(out, ones[:batch_len])
            log_loss_G.append(loss_G.item())

            params_D.zero_grad()
            params_G.zero_grad()
            loss_G.backward()
            params_G.step()

            real_img = real_img.to(device)
            real_out = model_D(real_img)
            loss_D_real = loss_f(real_out, ones[:batch_len])

            fake_out = model_D(fake_img_tensor)
            loss_D_fake = loss_f(fake_out, zeros[:batch_len])

            loss_D = loss_D_real + loss_D_fake
            log_loss_D.append(loss_D.item())

            params_D.zero_grad()
            params_G.zero_grad()
            loss_D.backward()
            params_D.step()

        result["log_loss_G"].append(statistics.mean(log_loss_G))
        result["log_loss_D"].append(statistics.mean(log_loss_D))
        print("log_loss_G =", result["log_loss_G"][-1], ", log_loss_D =", result["log_loss_D"][-1])

        if not os.path.exists(args.savefile):
            os.mkdir(args.savefile)
        if not os.path.exists(args.savefile+"/fakeimg"):
            os.mkdir(args.savefile+"/fakeimg")
        torchvision.utils.save_image(model_G(sample_z), args.savefile+"/fakeimg/ep"+str(i)+".png")
        modelG_path = 'modelG'+str(i)+'.pth'
        modelD_path = 'modelD'+str(i)+'.pth'
        if not os.path.exists(args.savefile+"/model"):
            os.mkdir(args.savefile+"/model")
        torch.save(model_G.state_dict(), args.savefile+"/model/"+modelG_path)
        torch.save(model_D.state_dict(), args.savefile+"/model/"+modelD_path)
    with open(args.savefile+"/logs.pkl", "wb") as fp:
        pickle.dump(result, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ep", type=int,  default=30)
    parser.add_argument("savefile")
    parser.add_argument("batch", type=int, default=128)
    parser.add_argument("phase", default="train")
    args = parser.parse_args()
    if args.phase == "train":
        train(args)
    else:
        test(args)
