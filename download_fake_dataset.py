"""
Downloads the real/fake Pakistani currency dataset from Kaggle using kagglehub
and organises it into:
    dataset/fake_detection/real/
    dataset/fake_detection/fake/

Run once before training:
    python download_fake_dataset.py
"""

import os
import shutil
import kagglehub

DEST_ROOT = os.path.join("dataset", "fake_detection")


def _copy_images(src_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".jfif", ".webp", ".bmp"}
    copied = 0
    for fname in os.listdir(src_dir):
        if os.path.splitext(fname)[1].lower() in exts:
            shutil.copy2(os.path.join(src_dir, fname), dest_dir)
            copied += 1
    return copied


def main():
    print("Downloading dataset from Kaggle …")
    path = kagglehub.dataset_download("mmuzamil/real-and-fake-currency-pakistanis-dataset")
    print(f"Downloaded to: {path}\n")

    # Walk the downloaded tree and map folder names to real / fake
    real_dest = os.path.join(DEST_ROOT, "real")
    fake_dest = os.path.join(DEST_ROOT, "fake")

    real_count = fake_count = 0

    for root, dirs, files in os.walk(path):
        folder_lower = os.path.basename(root).lower()
        if "fake" in folder_lower:
            n = _copy_images(root, fake_dest)
            fake_count += n
            if n:
                print(f"  fake ← {root}  ({n} images)")
        elif "real" in folder_lower or "genuine" in folder_lower or "original" in folder_lower:
            n = _copy_images(root, real_dest)
            real_count += n
            if n:
                print(f"  real ← {root}  ({n} images)")

    if real_count == 0 and fake_count == 0:
        # Fallback: print the tree so the user can inspect manually
        print("\nCould not auto-detect real/fake sub-folders. Directory tree:")
        for root, dirs, files in os.walk(path):
            level = root.replace(path, "").count(os.sep)
            indent = "  " * level
            print(f"{indent}{os.path.basename(root)}/")
            for f in files[:5]:
                print(f"{indent}  {f}")
        print("\nEdit DEST_ROOT mapping in this script and re-run.")
    else:
        print(f"\nDone. real={real_count}  fake={fake_count}")
        print(f"Organised into: {os.path.abspath(DEST_ROOT)}")


if __name__ == "__main__":
    main()
