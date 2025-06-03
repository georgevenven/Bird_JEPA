import os
import shutil

# Set your target directory here
target_dir = "/media/george-vengrovski/Desk SSD/BirdJEPA/pretrain1_train"

# File extensions to move
file_exts = (".pt", ".npz")

# Walk through all subdirectories
for root, dirs, files in os.walk(target_dir, topdown=False):
    for file in files:
        if file.endswith(file_exts):
            src = os.path.join(root, file)
            dst = os.path.join(target_dir, file)
            # If a file with the same name exists, rename the incoming file
            if os.path.exists(dst):
                base, ext = os.path.splitext(file)
                i = 1
                while os.path.exists(os.path.join(target_dir, f"{base}_{i}{ext}")):
                    i += 1
                dst = os.path.join(target_dir, f"{base}_{i}{ext}")
            shutil.move(src, dst)
    # Remove empty directories (except the top-level one)
    if root != target_dir and not os.listdir(root):
        os.rmdir(root)
