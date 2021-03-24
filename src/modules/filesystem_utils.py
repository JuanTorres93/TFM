import os
import pathlib

def home_directory(as_str=True):
    home_dir = pathlib.Path.home()

    if as_str:
        return str(home_dir)
    else:
        return home_dir

def path_exists(path_str: str):
    path = pathlib.Path(path_str)

    if pathlib.Path.exists(path):
        return True
    else:
        return False

def create_directory(path: str):
    if not path_exists(path):
        os.mkdir(path)