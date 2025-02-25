from captcha.image import ImageCaptcha
import random
import string
import os
import shutil
import sys
import multiprocessing

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

    # Pad the captcha string with underscores
    captcha_string = captcha_string + '_' * (8 - len(captcha_string))

    # Generate and save the image
    image = ImageCaptcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    image_path = os.path.join(image_dir, f"{image_name}.png")

    try:
        image.write(captcha_string, image_path)
    except Exception as e:
        print(f"Error saving image {image_path}: {e}")
        return None

    return captcha_string

# Used for parallel processing
def add_captcha(args):
    x, _, image_dir = args 
    image_id = f"captcha_{x}"
    captcha_string = generate_captcha(image_id, image_dir)
    if captcha_string:
        return f"{image_id}.png,{captcha_string}"


def generate_data(count, image_dir, output_csv, flags):
    # Clear directory and CSV if not extending
    data = []
    if "ge" not in flags:
        if os.path.exists(image_dir):
            shutil.rmtree(image_dir)
        # Reset CSV file by emptying data list
        start_num = 0
    else:
        # Get starting image number
        start_num = 0
        if os.path.exists(output_csv):
            with open(output_csv, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_num = int(lines[-1].split('.')[0].replace('captcha_', ''))
                    start_num = last_num + 1

    # Read existing data if extending and update padding if needed
    if "ge" in flags and os.path.exists(output_csv):
        with open(output_csv, 'r') as f:
            for line in f:
                if line.strip():
                    filename, captcha = line.strip().split(',')
                    data.append(f"{filename},{captcha}")
                

    # Multi-threaded generation
    num_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(num_cores) as pool:
        # Create list of argument tuples for each task
        args_list = [(x, None, image_dir) for x in range(start_num, start_num + count)]
        # Map the function with the arguments
        results = pool.map(add_captcha, args_list)
        # Filter out None results and store in data
        data = [result for result in results if result is not None]

    # Write mappings to CSV
    with open(output_csv, 'w') as f:
        f.write("\n".join(data) + "\n")


        
def generate_training_data(count=10_000, flags={}):
    generate_data(count=count, image_dir="Training_Data", output_csv="Training_Data_Mapping", flags=flags)

def generate_test_data(count=1000, flags={}):
    generate_data(count=count, image_dir="Test_Data", output_csv="Training_Data_Mapping", flags=flags)

        
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

           
