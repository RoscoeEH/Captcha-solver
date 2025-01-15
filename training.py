import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import string

# ====================
# Dataset Preparation
# ====================

# Define the transformation to convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor()
])

# Used to read in the filenames and the string encoded in the captcha
# Reads a 2-col csv into a hashmap where the first value is the key and the second is the value
def read_csv(path):
    hashMap = {}

    with open(path, "r") as file:
        for row in file.readlines():
            items = row.split(",")
            
            key = int(items[0][:-4])
            val = items[1][:-1]
                
            hashMap[key] = val

    return hashMap


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
        # Applies 2D convolution layer 
        self.conv1 = nn.Conv2d(1, 32, 3) # (1 for grayscale, 32 filters applied, 3x3 filters)
        # applies max pooling to the layers 
        self.pool = nn.MaxPool2d(2, 2)
        # applies second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, 3)

        # Adaptive pooling layer to fix output width
        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, 8))

        # Applies LSTM layer 
        self.lstm = nn.LSTM(input_size=1344, hidden_size=hidden_dim, num_layers=num_lstm_layers, batch_first=True)

        # Applies Fully Connected layer 
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # pool the results of conv1 -> pool the results of conv2
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        
        batch_size, channels, height, width = x.size()

        x = x.view(batch_size, width, channels * height)

        x, _ = self.lstm(x)
        
        x = self.fc(x)
        
        return x


    
# ====================
# Training Setup
# ====================
# Hyperparameters

# Total number of characters
num_classes = len(string.ascii_letters + string.digits)
hidden_dim = 128
num_lstm_layers = 2
learning_rate = 0.001
num_epochs = 10
batch_size = 16

# File paths
csv_file = "Training_Data_Mappings.csv"
cap_dir = "Training_Data"

# DataLoader
dataset = Captcha_Text_Dataset(labels_csv=csv_file, captcha_dir=cap_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, Loss, and Optimizer
device = torch.device("cpu")
model = Net(num_classes, hidden_dim, num_lstm_layers).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# ====================
# Training Loop
# ====================
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        batch_size, seq_len, num_classes = outputs.shape
        
        # Reshape outputs and labels
        outputs = outputs.reshape(-1, num_classes)
        labels = labels.reshape(-1)
        
        # Only compute loss on non-padding elements
        loss = criterion(outputs, labels)
                
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save the model's parameters (state_dict)
torch.save(model.state_dict(), "captcha_recognition_model.pth")
print("Model saved")

#  LocalWords:  captchas


