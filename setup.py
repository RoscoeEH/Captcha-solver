import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as func
import string
from helpers import read_csv

# ====================
# Dataset Preparation
# ====================

# Define the transformation to convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor()
])


class Captcha_Text_Dataset(Dataset):
    def __init__(self, labels_csv, captcha_dir, transform=None):
        # Hashmap of filenames to represented string
        self.data = read_csv(labels_csv)
        self.captcha_dir = captcha_dir
        self.transform = transform

        # Store the keys in a list for indexing
        self.keys = list(self.data.keys())

        # hashmap for char to index of a given alphanumeric char
        # Helps in conversion to tensor
        self.char_map = {char: idx for idx, char in enumerate(string.ascii_letters + string.digits)}
        # This is used to pad captchas the max length
        self.max_seq_len = 8



    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        # Use the stored keys to get the correct key for the data dictionary
        cap_name = self.keys[idx]
        cap_code = self.data[cap_name]
        # Load and transform image

        cap_path = os.path.join(self.captcha_dir, f"{cap_name:0{len(str(len(self.data)))}}.png")
        cap = Image.open(cap_path).convert("L")  # Convert to grayscale

        # Convert image to tensor 
        if self.transform:
            cap = self.transform(cap)

        # Convert cap_code to integer sequence
        cap_code_seq = [self.char_map[char] for char in cap_code] 
        cap_code_seq += [-100] * (self.max_seq_len - len(cap_code_seq))
        
        return cap, torch.tensor(cap_code_seq, dtype=torch.long)

        
############
# Neural Net
############

class Net(nn.Module):
    def __init__(self, num_classes, hidden_dim, num_lstm_layers):
        super().__init__()
        # Deeper CNN architecture
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, 8))
        
        # Bidirectional LSTM for better sequence modeling
        self.lstm = nn.LSTM(
            input_size=2048,  # Adjusted for new conv architecture
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_lstm_layers > 1 else 0
        )
        
        # Account for bidirectional in final layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # Enhanced forward pass with regularization
        x = self.pool(func.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(func.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(func.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        
        x = self.adaptive_pool(x)
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, width, channels * height)
        
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x
