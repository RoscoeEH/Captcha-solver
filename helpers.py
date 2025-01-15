# Helper functions


# Used to read in the filenames and the string encoded in the captcha
# Reads a 2-col csv into a hashmap where the first value is the key and the second is the value
def read_csv(path):
    hashMap = {}

    with open(path, "r") as file:
        for row in file.readlines():
            items = row.split(",")
            
            key = int(items[0][:-4])
            val = items[1][:-1]
                
            hashMap[key] = val

    return hashMap

# Early stop data comparison
def early_stop_check(data, threshhold, epsilon):
    if len(data) >= threshhold:
        
        for i in range(-threshhold, -1):
            if abs(data[i] - data[i + 1]) > epsilon:
                return False
            
        return True
    
    return False



