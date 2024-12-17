from captcha.image import ImageCaptcha
import random
import string
import os
import shutil
import sys

# Constants
IMAGE_DIR = "Training_Data"
OUTPUT_CSV = "Training_Data_Mappings.csv"
IMAGE_WIDTH = 280
IMAGE_HEIGHT = 90

def generate_captcha(image_name):
    """Generate a random CAPTCHA image and save it to IMAGE_DIR."""
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # Generate a random alphanumeric string
    num_chars = random.randint(5, 8)
    chars = string.ascii_letters + string.digits
    captcha_string = ''.join(random.choice(chars) for _ in range(num_chars))

    # Generate and save the image
    image = ImageCaptcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    image_path = os.path.join(IMAGE_DIR, f"{image_name}.png")

    try:
        image.write(captcha_string, image_path)
    except Exception as e:
        print(f"Error saving image {image_path}: {e}")
        return None

    return captcha_string

if __name__ == "__main__":
    # Argument parsing
    args = sys.argv
    if len(args) != 2 or not args[1].isdigit():
        print("Incorrect arguments")
        sys.exit(1)

    # Reset training data directory
    if os.path.exists(IMAGE_DIR):
        shutil.rmtree(IMAGE_DIR)

    # Generate images
    num_images = int(args[1])
    name_length = len(str(num_images))  # For zero-padding
    data = ["filename,label"]

    for i in range(1, num_images + 1):
        image_id = f"{i:0{name_length}}"
        captcha_string = generate_captcha(image_id)

        if captcha_string:  # Only add if the image generation succeeded
            data.append(f"{image_id}.png,{captcha_string}")

    # Write mappings to CSV
    with open(OUTPUT_CSV, 'w') as f:
        f.write("\n".join(data))

       
