import os
import pandas as pd
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Imports for neural net
import torch.nn as nn
import torch.nn.functional as func

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
class Net(nn.Module):
    def __init__(self, num_classes, hidden_dim, num_lstm_layers):
        super().__init__()
        self.conv1 == nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, num_layers=num_lstm_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # Reshape to (batch, width, channels * height)
        x = x.view(batch_size, width, -1)
        
        x, _ = self.lstm(x)  # Output shape: (batch, seq_len, hidden_dim)
        
        x = self.fc(x)  # Output shape: (batch, seq_len, num_classes)
        return x
