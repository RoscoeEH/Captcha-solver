import torch
from setup import Net, Captcha_Text_Dataset, transform, HIDDEN_DIM, NUM_LSTM_LAYERS, BATCH_SIZE, NUM_CLASSES
import string
from torch.utils.data import DataLoader
import sys

def evaluate_model(flags={}):
    # Setup device
    print("Evaluating on GPU" if torch.cuda.is_available() else "Evaluating on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = Net(NUM_CLASSES, HIDDEN_DIM, NUM_LSTM_LAYERS).to(device)
    
    # Load the saved model
    model.load_state_dict(torch.load("captcha_recognition_model.pth"))
    model.eval()  # Set to evaluation mode
    
    # Create test dataset and dataloader
    test_dataset = Captcha_Text_Dataset(
        labels_csv="Test_Data_Mappings.csv",
        captcha_dir="Test_Data",
        transform=transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create reverse char map for converting indices back to characters
    char_list = string.ascii_letters + string.digits + '_'
    idx_to_char = {idx: char for idx, char in enumerate(char_list)}
    
    correct_strings = 0
    total_strings = 0
    correct_chars = 0
    total_chars = 0
    
    print("Starting evaluation...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=2)
            
            # Convert predictions to text and calculate accuracies
            for pred, label in zip(predictions, labels):
                pred_text = ''.join([idx_to_char[idx.item()] for idx in pred if idx.item() >= 0])
                true_text = ''.join([idx_to_char[idx.item()] for idx in label if idx.item() >= 0])
                
                # Full string accuracy
                if pred_text == true_text:
                    correct_strings += 1
                total_strings += 1

                # Character-level accuracy
                min_len = min(len(pred_text), len(true_text))
                correct_chars += sum(1 for i in range(min_len) if pred_text[i] == true_text[i])
                total_chars += len(true_text)

                # Check for verbose output
                if "ev" in flags:
                    print(f"Predicted: {pred_text}, Actual: {true_text}")
                if "es" in flags:
                    with open("evaluation_results.txt", "a") as f:
                        f.write(f"Predicted: {pred_text}, Actual: {true_text}\n")
    
    string_accuracy = (correct_strings / total_strings) * 100
    char_accuracy = (correct_chars / total_chars) * 100
    
    print("\nEvaluation Results:")
    print(f"Full String Accuracy: {string_accuracy:.2f}%")
    print(f"Character-Level Accuracy: {char_accuracy:.2f}%")
    print(f"Correctly predicted {correct_strings} out of {total_strings} full strings")
    print(f"Correctly predicted {correct_chars} out of {total_chars} characters")

    # Save evaluation results
    if 'es' in flags:
        with open("evaluation_results.txt", 'a') as f:
            f.write("\n")
            f.write(f"Evaluation Results:\n")
            f.write(f"Full String Accuracy: {string_accuracy:.2f}%\n")
            f.write(f"Character-Level Accuracy: {char_accuracy:.2f}%\n")
            f.write(f"Correctly predicted {correct_strings} out of {total_strings} full strings\n")
            f.write(f"Correctly predicted {correct_chars} out of {total_chars} characters\n")
        print(f"\nEvaluation results saved to {save_path}")


if __name__ == "__main__":
    evaluate_model()
