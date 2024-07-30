import os

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")

def write_metadata(path, status):
    with open(path, 'a') as file:
        file.write(f'\nstatus: {status}')