import os
import pandas as pd
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

class Captcha_Text_Dataset(Dataset):
    def __init__(self, labels_csv, captcha_dir, transform=None):
        self.data = pd.read_csv(labels_csv)
        self.captcha_dir = captcha_dir
        self.transform = transform
        self.char_map = {char: idx for idx, char in enumerate(string.ascii_letters + string.digits)}
        self.max_seq_len = 8

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        cap_name = row['filename']
        label = row['label']

        # Load and transform image
        cap_path = os.path.join(self.captcha_dir, cap_name)
        cap = Image.open(cap_path).convert("L")  # Convert to grayscale
        if self.transform:
            cap = self.transform(cap)

        # Convert label to integer sequence
        label_seq = [self.char_map[char] for char in label]
        label_seq += [0] * (self.max_seq_len - len(label_seq))  # Pad to max_seq_len

        return cap, torch.tensor(label_seq, dtype=torch.long)


############
# Neural Net
############

class Net(nn.Module):
    def __init__(self, num_classes, hidden_dim, num_lstm_layers):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # Grayscale input
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, num_layers=num_lstm_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, width, -1)

        x, _ = self.lstm(x)
        x = self.fc(x)
        x = x.permute(0, 2, 1)

        return x



# ====================
# Training Setup
# ====================
# Hyperparameters
num_classes = len(string.ascii_letters + string.digits)  # Total number of characters
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
if __name__ == "__main__":
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)  # Shape: (batch, num_classes, seq_len)

            # Reshape labels to match CrossEntropyLoss input requirements
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
