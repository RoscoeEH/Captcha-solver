from captcha.image import ImageCaptcha
import random
import string
import os
import shutil
import sys

# Constants
IMAGE_WIDTH = 280
IMAGE_HEIGHT = 90

def generate_captcha(image_name, image_dir):
    """Generate a random CAPTCHA image and save it to IMAGE_DIR."""
    os.makedirs(image_dir, exist_ok=True)

    # Generate a random alphanumeric string
    num_chars = random.randint(5, 8)
    chars = string.ascii_letters + string.digits
    captcha_string = ''.join(random.choice(chars) for _ in range(num_chars))

    # Generate and save the image
    image = ImageCaptcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    image_path = os.path.join(image_dir, f"{image_name}.png")

    try:
        image.write(captcha_string, image_path)
    except Exception as e:
        print(f"Error saving image {image_path}: {e}")
        return None

    return captcha_string


def generate_training_data(count=100_000, flags=[]):
    
    image_dir = "Training_Data"
    output_csv = "Training_Data_Mappings.csv"
    # Generate test data vs training data
    if "T" in flags:
        image_dir = "Test_Data"
        output_csv = "Test_Data_Mappings.csv"

    data = []
    # Clear directory and CSV if not extending
    if "E" not in flags:
        if os.path.exists(image_dir):
            shutil.rmtree(image_dir)
        # Reset CSV file by emptying data list
        start_num = 1
        existing_max_digits = 1
    else:
        # Get starting image number and determine max digits needed
        start_num = 1
        existing_max_digits = 1
        if os.path.exists(output_csv):
            with open(output_csv, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_num = int(lines[-1].split('.')[0])
                    start_num = last_num + 1
                    existing_max_digits = len(str(last_num))

    

    # Calculate required digits based on both existing and new numbers
    final_num = start_num + count - 1
    name_length = max(existing_max_digits, len(str(final_num)))

    # Read existing data if extending and update padding if needed
    if "E" in flags and os.path.exists(output_csv):
        with open(output_csv, 'r') as f:
            for line in f:
                if line.strip():
                    # Update padding of existing entries if necessary
                    filename, captcha = line.strip().split(',')
                    num = int(filename.split('.')[0])
                    updated_filename = f"{num:0{name_length}}.png"
                    data.append(f"{updated_filename},{captcha}")

    # Generate new images
    for i in range(start_num, start_num + count):
        image_id = f"{i:0{name_length}}"
        captcha_string = generate_captcha(image_id, image_dir)

        if captcha_string:
            data.append(f"{image_id}.png,{captcha_string}")

    # Write mappings to CSV
    with open(output_csv, 'w') as f:
        f.write("\n".join(data) + "\n")


        
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

    generate_training_data(count=int(args[1]), flags=params)

           
