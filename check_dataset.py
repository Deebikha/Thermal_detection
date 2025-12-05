import os
from collections import defaultdict

root = "datasets/coco128"   # change to "dataset" if needed
img_train = os.path.join(root, "images", "train2017")
lbl_train = os.path.join(root, "labels", "train2017")
img_val = os.path.join(root, "images", "val")
lbl_val = os.path.join(root, "labels", "val")

def check_pair(img_dir, lbl_dir):
    imgs = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg",".png"))]
    missing_labels=[]
    for im in imgs:
        lbl = os.path.splitext(im)[0] + ".txt"
        if not os.path.exists(os.path.join(lbl_dir,lbl)):
            missing_labels.append(im)
    print(f"{img_dir}: {len(imgs)} images, missing labels: {len(missing_labels)}")
    if len(missing_labels)>0:
        print("Examples missing labels:", missing_labels[:5])

def class_counts(lbl_dir):
    from collections import defaultdict
    cnt = defaultdict(int)
    files = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")]
    for f in files:
        with open(os.path.join(lbl_dir,f)) as fh:
            for line in fh:
                if line.strip()=="":
                    continue
                cls = int(line.split()[0])
                cnt[cls]+=1
    return cnt

print("TRAIN CHECK")
check_pair(img_train, lbl_train)
print("VAL CHECK")
check_pair(img_val, lbl_val)

print("\nClass counts (train):")
print(class_counts(lbl_train))
print("\nClass counts (val):")
print(class_counts(lbl_val))
