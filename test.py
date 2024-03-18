import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
from torch.utils.data import Dataset, DataLoader
from cGAN.generator import *
from cGAN.dcnator import Dcnator
from cGAN.facades1 import test, transform

def print_header():
    # Clear the terminal screen
    os.system('cls' if os.name == 'nt' else 'clear')

    # Print the header
    print("=" * 80)
    print("THE SYSTEM MACHINE LEARNING FOR BOILING".center(80))
    print("=" * 80)
    print(f"File: {'test.py'}".center(80))
    print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
    print("=" * 80)
    print("The provided code defines UNet and Discriminator modules for image translation using the Conditional GAN architecture. It includes a custom dataset class for loading and preprocessing images, as well as functions for generating images, and visualizing the results using t-SNE for clustering. The code also handles model initialization, weight loading, loss functions, optimization, and saving the generated images.")
    print("=" * 80)

print_header()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path2img_test = 'dataset_1to2/test'
test_ds = test(path2img_test, transform=transform)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

# Model initialization and weight loading

model_gen = Generator().to(device)
model_dis = Dcnator().to(device)

def initialize_weights(model):
    if isinstance(model, nn.Conv2d):
        nn.init.normal_(model.weight.data, 0.0, 0.02)

model_gen.apply(initialize_weights)
model_dis.apply(initialize_weights)

loss_func_gan = nn.BCELoss()
loss_func_pix = nn.L1Loss()
lambda_pixel = 100
patch = (1, 6, 6)

opt_dis = torch.optim.Adam(model_dis.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_gen = torch.optim.Adam(model_gen.parameters(), lr=2e-4, betas=(0.5, 0.999))

path2models = './models/'
os.makedirs(path2models, exist_ok=True)
path2weights_gen = os.path.join(path2models, 'weights_gen.pt')
path2weights_dis = os.path.join(path2models, 'weights_dis.pt')


def generate_images(test_ds, model_gen, device):
    count = 0
    global space
    space = []
    os.makedirs(os.path.join('results', 'test_1'), exist_ok=True)
    with torch.no_grad():
        for a, b in tqdm(test_dl):
            fake_imgs = model_gen(a.to(device)).detach().cpu().squeeze()
            count = count + 1
            fake = transforms.functional.to_pil_image(0.5 * fake_imgs + 0.5)
            fake.save(os.path.join('results', "test_1", "fake_" + str(count) + ".png"))
            space.append(torch.flatten(fake_imgs))
        space = torch.stack(space, dim=0).cpu().numpy().squeeze()
    return np.stack(space, axis=0).squeeze()

weights = torch.load(path2weights_gen)
model_gen.load_state_dict(weights)
model_gen.eval()

generate_images(test_dl, model_gen, device)

'''
test_cases = {
    'test': 'tester/test',
}

test_datasets = {}
test_dataloaders = {}

for test_case, path in test_cases.items():
    test_datasets[test_case] = FacadeDataset(path, transform=transform)
    test_dataloaders[test_case] = DataLoader(test_datasets[test_case], batch_size=1, shuffle=False)
'''

''' t-sne algorithem
spaces = {}

for test_case in test_cases.keys():
    spaces[test_case] = generate_images(test_dataloaders[test_case], model_gen, device)
    print("Waiting for 100 seconds...")
    time.sleep(100)

x2 = spaces['1533_CHF']
x3 = spaces['1613_CHF']
whole = np.concatenate([x2, x3], axis=0)
whole = whole.squeeze()
name = []
for i in range(1650):
    name.append("1533kW/$m^2$, CHF")
for i in range(1650):
    name.append("1613kW/$m^2$, CHF")

tsne = TSNE(n_components=2, perplexity=10)
cluster = np.array(tsne.fit_transform(np.array(whole)))

color_map = {
    "1533kW/$m^2$, CHF": "green",
    "1613kW/$m^2$, CHF": "blue",
}

colors = [color_map[name] for name in name]

start = 0
end = 1650

for label, color in color_map.items():
    plt.scatter(
        cluster[start:end, 0],
        cluster[start:end, 1],
        alpha=0.5,
        c=color,
        label=label,
    )
    start += 1650
    end += 1650

legend_elements = [
    mpatches.Patch(color=color, label=label) for label, color in color_map.items()
]

plt.legend(handles=legend_elements, loc='upper left')
plt.show()
'''