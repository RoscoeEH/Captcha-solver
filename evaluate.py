import torch
from setup import Net, Captcha_Text_Dataset, transform
import string
from torch.utils.data import DataLoader
import sys

def evaluate_model():
    # Setup parameters
    num_classes = len(string.ascii_letters + string.digits)
    hidden_dim = 256
    num_lstm_layers = 4
    batch_size = 32

    # Setup device
    print("Running on GPU" if torch.cuda.is_available() else "Running on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = Net(num_classes, hidden_dim, num_lstm_layers).to(device)
    
    # Load the saved model
    model.load_state_dict(torch.load("captcha_recognition_model.pth"))
    model.eval()  # Set to evaluation mode
    
    # Create test dataset and dataloader
    test_dataset = Captcha_Text_Dataset(
        labels_csv="Test_Data_Mappings.csv",
        captcha_dir="Test_Data",
        transform=transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create reverse char map for converting indices back to characters
    char_list = string.ascii_letters + string.digits
    idx_to_char = {idx: char for idx, char in enumerate(char_list)}
    
    correct = 0
    total = 0
    
    print("Starting evaluation...")
    with torch.no_grad():  # No need to track gradients for evaluation
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=2)
            
            # Convert predictions to text
            for pred, label in zip(predictions, labels):
                pred_text = ''.join([idx_to_char[idx.item()] for idx in pred if idx.item() >= 0])
                true_text = ''.join([idx_to_char[idx.item()] for idx in label if idx.item() >= 0])
                
                if pred_text == true_text:
                    correct += 1
                total += 1

                # Check for verbose output
                if "-V" in sys.argv:
                    print(f"Predicted: {pred_text}, Actual: {true_text}")
    
    accuracy = (correct / total) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")
    print(f"Correctly predicted {correct} out of {total} captchas")

if __name__ == "__main__":
    evaluate_model() 
