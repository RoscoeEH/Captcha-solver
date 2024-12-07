from captcha.image import ImageCaptcha
import random
import string
import os

def generate_captcha(image_name):
    # Sets the dir to store the captcha 
    image_dir = "Training_Data"
    os.makedirs(image_dir, exist_ok=True)
    
    
    # Generate a random alphanumeric string 
    num_chars = random.randint(5,8)
    chars = string.ascii_letters + string.digits
    captcha_string = ''.join(random.choice(chars) for i in range(num_chars))

    # Generate the image and save it
    image = ImageCaptcha(width = 280, height = 90)
    image_path = os.path.join(image_dir, f"{image_name}.png")
    image.write(captcha_string, image_path)
        
       
    return captcha_string

if __name__ == "__main__":
    data = "filename,label\n"
    
    for i in range(100_000):
        image_id = f"{i:05}"
        captcha_string = generate_captcha(image_id)

        data += f"{image_id}.png,{captcha_string}\n"
            

    with open("Training_Data_Mappings.csv", 'w') as f:
        f.truncate(0)
        f.write(data)
