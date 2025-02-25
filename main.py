from evaluate import evaluate_model
from training import train_model
from generate import generate_training_data, generate_test_data
import sys



def parse_flags(args):
    # Valid flag characters for each flag type
    valid_flags = {
        'g': ['e', 'r'],
        't': ['v', 's'],
        'e': ['v', 's']
    }
    
    command_args = {"g", "gen", "generate", "t", "train", "e", "eval", "evaluate"}
    
    flags = {}
    
    # Skip the command argument if present
    start_idx = 1 if args and args[0].lower() in command_args else 0
    
    i = start_idx
    while i < len(args):
        arg = args[i].lower()
        
        # Handle flags
        if len(arg) >= 2 and arg[0] == "-" and arg[1] in valid_flags:
            flag_type = arg[1]
            # Process each character as a flag
            for char in arg[2:]:
                if char not in valid_flags[flag_type]:
                    raise Exception(f"Invalid flag character '{char}' for flag type '-{flag_type}'")
                
                # Check if next argument is a value
                if i + 1 < len(args) and not args[i + 1].startswith("-"):
                    flags[flag_type + char] = args[i + 1]
                    i += 1
                else:
                    flags[flag_type + char] = True
        else:
            raise Exception(f"Unrecognized argument: {arg}")
        
        i += 1

    return flags
        

def get_gen_count():
    count = input("How many samples should be generated (default 100,000)")
    if count == "":
        count = 100_000
    elif count.isdigit():
        count = int(count)
    else:
        raise Exception("Input must be a number")

    return count



if __name__ == "__main__":
    args = sys.argv[1:]
    flags = parse_flags(args)

    if len(args) > 1 and args[0].lower() in ["g", "gen", "generate"]:
        # Run the generate function only
        count = get_gen_count()
        generate_training_data(count=count, flags=flags)

    elif len(args) > 1 and args[0].lower() in ["t", "train"]:
        # Train on existing data
        train_model(flags=flags)

    elif len(args) > 1 and args[0].lower() in ["e", "eval", "evaluate"]:
        # Evaluate the data
        if "eg" in flags:
            generate_test_data(flags=flags)
        evaluate_model(flags)

    else:
        # Run everything
        if "gr" not in flags:
            count = get_gen_count()
            generate_training_data(count=count, flags=flags)

        train_model(flags=flags)

        generate_test_data(flags=flags)
        evaluate_model(flags)
        
