import os
import shutil

def clear():
    if os.path.exists("world.log"):
        os.remove("world.log")

    if os.path.exists("models"):
        shutil.rmtree("models")
        os.makedirs("models")

if __name__ == "__main__":
    clear()