# setup.py
#
# Author: RoscoeEH
#
# Description:
# Contains model setup for CAPTCHA recognition, including dataset class, data
# transformations, and the convolutional-recurrent neural network architecture.
# Also defines constants related to model training and character encoding.

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
import string

# ====================
# Model Parameters

NUM_CLASSES = len(string.ascii_letters + string.digits + '_')
HIDDEN_DIM = 512
NUM_LSTM_LAYERS = 2
LEARNING_RATE = 0.0005
NUM_EPOCHS = 100
BATCH_SIZE = 32
EARLY_STOP_THRESHHOLD = 15
EPSILON = 1e-4



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
        self.char_map = {char: idx for idx, char in enumerate(string.ascii_letters + string.digits + '_')}
        # This is used to pad captchas the max length
        self.max_seq_len = 8



    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        # Use the stored keys to get the correct key for the data dictionary
        cap_name = self.keys[idx]
        cap_code = self.data[cap_name]
        
        cap_path = os.path.join(self.captcha_dir, f"captcha_{cap_name}.png")
        cap = Image.open(cap_path).convert("L")  # Convert to grayscale
        
        # Convert image to tensor 
        if self.transform:
            cap = self.transform(cap)

        # Convert cap_code to integer sequence
        cap_code_seq = [self.char_map[char] for char in cap_code] 
        cap_code_seq += [self.char_map['_']] * (self.max_seq_len - len(cap_code_seq))
        
        return cap, torch.tensor(cap_code_seq, dtype=torch.long)

        
############
# Neural Net
############

class Net(nn.Module):
    def __init__(self, num_classes, hidden_dim, num_lstm_layers):
        super().__init__()
        # Simplified CNN architecture with fewer channels
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # Reduced from 32
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # Reduced from 64
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)  # Reduced from 128
        self.bn3 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # Fixed size output
        
        # Calculate LSTM input size based on CNN output
        cnn_output_size = 64 * 8 * 8 // 8  # Channels * Height * Width / sequence length
        
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_lstm_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # CNN layers
        x = self.pool(func.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(func.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = func.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        # Fixed size output
        x = self.adaptive_pool(x)
        batch_size, channels, height, width = x.size()
        
        # Reshape for LSTM - more explicit reshaping
        x = x.permute(0, 3, 1, 2)  # [batch, width, channels, height]
        x = x.contiguous().view(batch_size, width, channels * height)
        
        # LSTM and final layers
        try:
            self.lstm.flatten_parameters()
            x, _ = self.lstm(x)
            x = self.fc(x)
            return x
        except RuntimeError as e:
            print(f"Error in LSTM forward pass: {e}")
            print(f"Input shape to LSTM: {x.shape}")
            raise
