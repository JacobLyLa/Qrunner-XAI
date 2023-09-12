import os

def prepare_folders(folders_path):
    os.makedirs(folders_path, exist_ok=True)
    for file in os.listdir(folders_path):
        os.remove(f"{folders_path}/{file}")