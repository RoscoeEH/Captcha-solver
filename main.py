from evaluate import evaluate_model
from training import train_model
from generate import generate_training_data, generate_test_data
import sys



def parse_flags(args):
    flags = {}

    if len(args) > 0:
        for arg in args:
            arg = arg.lower()
            
            if arg in ["g", "gen", "generate", "t", "train", "e", "eval", "evaluate"]:
                continue

            elif len(arg) < 3 or arg[0] != "-" or arg[1] not in ["g", "e", "t"]:
                raise Exception(f"Unrecognized arugment: {arg}")

            else:
                flag_type = arg[1]
                for char in arg[2:]:
                    flags[flag_type + char] = 0

            # TODO: add unrecongized arguments for other kinds of flags

    return flags
        

def get_gen_count():
    count = input("How many samples should be generated (default 10,000)")
    if count == "":
        count = 10_000
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
        
        count = get_gen_count()
        generate_training_data(count=count, flags=flags)

        train_model(flags=flags)

        generate_test_data(flags=flags)
        evaluate_model(flags)
        



# TODO: handle bug with flags in test v training generation
