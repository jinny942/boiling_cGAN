import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import time
import matplotlib.pyplot as plt
from cGAN.generator import *
from cGAN.dcnator import Dcnator
from cGAN.facades1 import train, test, transform

import os
import datetime
from rich.progress import Progress

def print_header():
    # Clear the terminal screen
    os.system('cls' if os.name == 'nt' else 'clear')

    # Print the header
    print("=" * 80)
    print("THE SYSTEM MACHINE LEARNING FOR BOILING".center(80))
    print("=" * 80)
    print(f"File: {'train.py'}".center(80))
    print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
    print("=" * 80)
    print("\nThis script implements a UNet-based generator and discriminator for image-to-image translation using the Pix2Pix architecture. The generator generates fake images, while the discriminator differentiates between real and fake images. The script includes a custom dataset class, training loop, and evaluation code to generate and display fake-to-real images.\n")
    print("=" * 80)

print_header()


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset and DataLoader creation
path2img = 'dataset_1to2/train'
train_ds = train(path2img, transform=transform)
train_dl = DataLoader(train_ds, batch_size=1, shuffle=False)
path2img_test = 'dataset_1to2/test'
test_ds = test(path2img_test, transform=transform)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

# Generator and Discriminator initialization
model_gen = Generator().to(device)
model_dis = Dcnator().to(device)

# Weight initialization
def initialize_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)

model_gen.apply(initialize_weights)
model_dis.apply(initialize_weights)

# Loss functions
loss_func_gan = nn.BCELoss()
loss_func_pix = nn.L1Loss()

# Hyperparameters
lambda_pixel = 100
patch = (1, 8, 3)
lr = 2e-4
beta1 = 0.5
beta2 = 0.999
num_epochs = 100

# Optimizers
opt_dis = torch.optim.Adam(model_dis.parameters(), lr=lr, betas=(beta1, beta2))
opt_gen = torch.optim.Adam(model_gen.parameters(), lr=lr, betas=(beta1, beta2))

# Training loop
model_gen.train()
model_dis.train()
batch_count = 0
start_time = time.time()
loss_hist = {'gen': [], 'dis': []}



# Initialize progress bar
with Progress() as p:
    task1 = p.add_task("[cyan]Training...", total=1000)
    for epoch in range(num_epochs):
        for real_a, real_b in train_dl:
            ba_si = real_a.size(0)

            # Real and fake labels
            real_label = torch.ones((ba_si, *patch), requires_grad=False).to(device)
            fake_label = torch.zeros((ba_si, *patch), requires_grad=False).to(device)

            # Generator
            model_gen.zero_grad()
            fake_b = model_gen(real_a.to(device))
            out_dis = model_dis(fake_b, real_b.to(device))
            gen_loss = loss_func_gan(out_dis, real_label)
            pixel_loss = loss_func_pix(fake_b, real_b.to(device))
            g_loss = gen_loss + lambda_pixel * pixel_loss
            g_loss.backward()
            opt_gen.step()

            # Discriminator
            model_dis.zero_grad()
            out_dis = model_dis(real_b.to(device), real_a.to(device))
            real_loss = loss_func_gan(out_dis, real_label)
            out_dis = model_dis(fake_b.detach(), real_a.to(device))
            fake_loss = loss_func_gan(out_dis, fake_label)
            d_loss = (real_loss + fake_loss) / 2.
            d_loss.backward()
            opt_dis.step()

            loss_hist['gen'].append(g_loss.item())
            loss_hist['dis'].append(d_loss.item())

            #batch_count += 1

            batch_count += 1


            if batch_count % len(train_dl) == 0:
                p.console.print(
                    f'Epoch: {epoch}, '
                    f'G_Loss: {g_loss.item():.6f}, '
                    f'D_Loss: {d_loss.item():.6f}, '
                    f'time: {(time.time() - start_time) / 60:.2f} min'
                )
                p.remove_task(task1)
                task1 = p.add_task("[cyan]Training...", total=1000)  # Add a new progress bar
            else:
                p.update(task1, advance=1)  # Advance the progress bar
            

# Loss history plot
plt.figure(figsize=(10, 5))
plt.title('Loss Progress')
plt.plot(loss_hist['gen'], label='Gen. Loss')
plt.plot(loss_hist['dis'], label='Dis. Loss')
plt.xlabel('Batch count')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_progress.png', bbox_inches='tight')

# Saving weights
path2models = './models/'
os.makedirs(path2models, exist_ok=True)
path2weights_gen = os.path.join(path2models, 'weights_gen.pt')
path2weights_dis = os.path.join(path2models, 'weights_dis.pt')
torch.save(model_gen.state_dict(), path2weights_gen)
torch.save(model_dis.state_dict(), path2weights_dis)

# Loading weights
weights = torch.load(path2weights_gen)
model_gen.load_state_dict(weights)
