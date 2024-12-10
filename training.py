import os
import pandas as pd
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Imports for neural net
import torch.nn as nn
import torch.nn.functional as Func

############################
# Convert images to tensors
############################

Transform = transforms.Compose([
    transforms.ToTensor()
])


class Captcha_Text_Dataset(Dataset):
    def __init__(self, labels_csv, captcha_dir, transform = None):
        self.data = pd.read_csv(labels_csv)
        self.captcha_dir = captcha_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __get_item__(self, idx):
        row = self.data.iloc[idx]
        cap_name = row['filename']
        label = row['label']

        cap_path = os.path.join(self.captcha_dir, cap_name)
        cap = Image.open(cap_path)

        if self.transform:
            cap = self.transform(cap)

        return cap, label


csv_file = "Training_Data_Mappings.csv"
cap_dir = "Training_Data"

dataset = Captcha_Text_Dataset(labels_csv=csv_file, captcha_dir=cap_dir, transform=Transform)

dataloader = DataLoader(dataset, shuffle=True)

############
# Neural Net
############


# Class represents a convolutions neural network
class Net(nn.module):
    def __init__(self):
        super().__init__()
        
