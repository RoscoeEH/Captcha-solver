import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import string
from helpers import early_stop_check
from setup import Net, Captcha_Text_Dataset, transform
import os



def train_model(hidden_dim=256, num_lstm_layers=4, learning_rate=0.001,
          num_epochs=50, batch_size=64, early_stop_threshhold=10,
          training_csv_file="Training_Data_Mappings.csv",
          training_data_dir="Training_Data"):
    
    # Total number of characters
    num_classes = len(string.ascii_letters + string.digits)
    epsilon = 1e-4

    # DataLoader
    dataset = Captcha_Text_Dataset(labels_csv=training_csv_file, captcha_dir=training_data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, Loss, and Optimizer
    print("Training on GPU..." if torch.cuda.is_available() else "Training on CPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(num_classes, hidden_dim, num_lstm_layers).to(device)

    optimizer = optim.Adam(model.parameters(), 
                        lr=learning_rate,
                        weight_decay=1e-5)
    # Check if a saved model exists
    model_path = "captcha_recognition_model.pth"
    optimizer_path = "captcha_optimizer.pth"


    # Load the previous iteration
    if os.path.exists(model_path):
        print("Loading existing model...")
        model.load_state_dict(torch.load(model_path))
        if os.path.exists(optimizer_path):
            print("Loading optimizer state...")
            optimizer.load_state_dict(torch.load(optimizer_path))

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
    train_model()
