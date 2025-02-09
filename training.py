import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import string
from helpers import read_csv, early_stop_check


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


def train(): 
    # ====================
    # Training Setup
    # ====================
    # Hyperparameters

    # Total number of characters
    num_classes = len(string.ascii_letters + string.digits)
    hidden_dim = 256
    num_lstm_layers = 4
    learning_rate = 0.001
    num_epochs = 50
    batch_size = 64
    early_stop_threshhold = 10
    epsilon = 1e-4


    # File paths
    csv_file = "Training_Data_Mappings.csv"
    cap_dir = "Training_Data"

    # DataLoader
    dataset = Captcha_Text_Dataset(labels_csv=csv_file, captcha_dir=cap_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(num_classes, hidden_dim, num_lstm_layers).to(device)

    optimizer = optim.Adam(model.parameters(), 
                        lr=learning_rate,
                        weight_decay=1e-5)
    # Check if a saved model exists
    model_path = "captcha_recognition_model.pth"
    optimizer_path = "captcha_optimizer.pth"


    # Load the previous iteration
    # if os.path.exists(model_path):
    #     print("Loading existing model...")
    #     model.load_state_dict(torch.load(model_path))
    #     if os.path.exists(optimizer_path):
    #         print("Loading optimizer state...")
    #         optimizer.load_state_dict(torch.load(optimizer_path))

    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')


    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3
    )

    # ====================
    # Training Loop
    # ====================
    print("Training...")
    epoch_loss = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Learning Rate: {current_lr:.6f}")

        # Update learning rate based on loss
        scheduler.step(avg_loss)

        # Early stopping
        epoch_loss.append(avg_loss)
        if early_stop_check(epoch_loss, early_stop_threshhold, epsilon):
            print("Model stagnation has reached the early stop threshold")
            break



    # Save both model and optimizer states
    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path)
    print("Model and optimizer states saved")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("GPU being used as Torch device")
    else:
        print("CPU being used as Torch device")

    train()
