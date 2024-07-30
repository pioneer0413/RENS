import os

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")

def write_metadata(path, status):
    with open(path, 'a') as file:
        file.write(f'\nstatus: {status}')

def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x} is not in range [0.0, 1.0]")
    return x