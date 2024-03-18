from os import listdir
from os.path import join
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random

class draw(Dataset):
    def __init__(self, path2img, direction='b2a', transform=False):
        super().__init__()
        self.direction = direction
        self.path2a = join(path2img, 'a')
        self.path2b = join(path2img, 'b')
        self.img_filenames = []

        # List all files in both directories
        all_files =  listdir(self.path2b)

        # Filter out only .jpg files
        all_jpg_files = [file for file in all_files if file.endswith('.jpg')]

        # Sort the list of filenames alphabetically
        all_jpg_files.sort()

        if path2img == 'facades1/train':
            self.img_filenames = random.shuffle(all_jpg_files) #train시에는 랜덤한 순서(연속성 배제)
        else :
            self.img_filenames = all_jpg_files #test시에는 순서대로

        self.transform = transform

    def __getitem__(self, index):
        a = Image.open(join(self.path2a, self.img_filenames[index])).convert('RGB')
        b = Image.open(join(self.path2b, self.img_filenames[index])).convert('RGB')
        
        if self.transform:
            a = self.transform(a)
            b = self.transform(b)

        if self.direction == 'b2a':
            return b,a
        else:
            return a,b

    def __len__(self):
        return len(self.img_filenames)

transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
                    transforms.Resize((540,76))
])

