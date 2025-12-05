import os
import random
import shutil

images_path = r"C:\Users\deebi\Desktop\thermal_detection\datasets\coco128\images\train2017"
labels_path = r"C:\Users\deebi\Desktop\thermal_detection\datasets\coco128\labels\train2017"
val_images_path = r"C:\Users\deebi\Desktop\thermal_detection\datasets\coco128\images\val"
val_labels_path = r"C:\Users\deebi\Desktop\thermal_detection\datasets\coco128\labels\val"

os.makedirs(val_images_path, exist_ok=True)
os.makedirs(val_labels_path, exist_ok=True)

all_images = [f for f in os.listdir(images_path) if f.endswith((".jpg", ".png"))]

val_count = int(0.2 * len(all_images))
val_images = random.sample(all_images, val_count)

moved = 0

for img in val_images:
    label_file = os.path.splitext(img)[0] + ".txt"
    label_path = os.path.join(labels_path, label_file)

    # Skip images without label file
    if not os.path.exists(label_path):
        print(f"⚠️ Skipping {img} (no label found)")
        continue

    # Move image
    shutil.move(os.path.join(images_path, img), os.path.join(val_images_path, img))

    # Move label
    shutil.move(label_path, os.path.join(val_labels_path, label_file))

    moved += 1

print(f"\n✅ Successfully moved {moved} image-label pairs to val folder.")
