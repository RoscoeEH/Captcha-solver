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
    if len(args) < 2 or not args[1].isdigit():
        print("Incorrect arguments")
        sys.exit(1)

    params = []
    if len(args) > 2:
        for a in args[2:]:
            params.append(a[1].upper())

    # Clear directory and CSV if not extending
    if "E" not in params:
        if os.path.exists(IMAGE_DIR):
            shutil.rmtree(IMAGE_DIR)
        # Reset CSV file by emptying data list
        data = []
        start_num = 1
        existing_max_digits = 1
    else:
        # Get starting image number and determine max digits needed
        start_num = 1
        existing_max_digits = 1
        if os.path.exists(OUTPUT_CSV):
            with open(OUTPUT_CSV, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_num = int(lines[-1].split('.')[0])
                    start_num = last_num + 1
                    existing_max_digits = len(str(last_num))

    # Calculate required digits based on both existing and new numbers
    num_images = int(args[1])
    final_num = start_num + num_images - 1
    name_length = max(existing_max_digits, len(str(final_num)))

    # Read existing data if extending and update padding if needed
    if "E" in params and os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'r') as f:
            for line in f:
                if line.strip():
                    # Update padding of existing entries if necessary
                    filename, captcha = line.strip().split(',')
                    num = int(filename.split('.')[0])
                    updated_filename = f"{num:0{name_length}}.png"
                    data.append(f"{updated_filename},{captcha}")

    # Generate new images
    for i in range(start_num, start_num + num_images):
        image_id = f"{i:0{name_length}}"
        captcha_string = generate_captcha(image_id)

        if captcha_string:
            data.append(f"{image_id}.png,{captcha_string}")

    # Write mappings to CSV
    with open(OUTPUT_CSV, 'w') as f:
        f.write("\n".join(data) + "\n")

       
