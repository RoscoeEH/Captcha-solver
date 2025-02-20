# Helper functions


# Used to read in the filenames and the string encoded in the captcha
# Reads a 2-col csv into a hashmap where the first value is the key and the second is the value
def read_csv(csv_file):
    """Read CSV file and return dictionary of filename to captcha text mappings."""
    data = {}
    with open(csv_file, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                items = line.strip().split(',')
                # Extract number from "captcha_X.png" format
                key = items[0].replace('captcha_', '').replace('.png', '')
                data[key] = items[1]
    return data

# Early stop data comparison
def early_stop_check(data, threshhold, epsilon):
    if len(data) >= threshhold:
        
        for i in range(-threshhold, -1):
            if abs(data[i] - data[i + 1]) > epsilon:
                return False
            
        return True
    
    return False



