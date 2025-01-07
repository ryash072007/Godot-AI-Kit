import os
import random
import shutil

def create_testing_set(source_dir, dest_dir, test_ratio=0.1):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    files = os.listdir(source_dir)
    test_size = int(len(files) * test_ratio)
    test_files = random.sample(files, test_size)
    
    for file in test_files:
        shutil.move(os.path.join(source_dir, file), os.path.join(dest_dir, file))

def main():
    create_testing_set("./training_data/O", "./testing_data/O")
    create_testing_set("./training_data/X", "./testing_data/X")

if __name__ == "__main__":
    main()