import os
import cv2

# =========================================================
# Dataset options
dataset_root_path = "/media/andrew/b84c4d95-450e-4802-b12d-b33e25343b1b/home/andrew/MURA"
processed_dataset_root_path = "/media/andrew/b84c4d95-450e-4802-b12d-b33e25343b1b/home/andrew/MURA_PROCESSED"
bone_types = ["XR_ELBOW", "XR_FINGER", "XR_FOREARM", "XR_HAND", "XR_HUMERUS", "XR_SHOULDER", "XR_WRIST"]
dataset_types = [*(["train"] * len(bone_types)), *(["valid"] * len(bone_types))]

# Canny coefficients
K = 5
THRESHOLD1 = 17
THRESHOLD2 = 10
# =========================================================

pathName = ""
pathLastName = ""
nameImage = "image"
img_ext = ".png"
txt_ext = ".txt"
n = 1
i = 0


def make_dataset_path(dataset_type, bone_type):
    return os.path.join(dataset_root_path, dataset_type, bone_type), dataset_type, bone_type


for dataset_folder, dataset_type, bone_type in map(make_dataset_path, dataset_types, bone_types * 2):
    print("Folder: ", dataset_folder)

    for root, directories, filenames in os.walk(dataset_folder):
        for filename in filenames:
            print("filename: ", filename)

            pathName = os.path.join(root, filename)
            orig_img = cv2.imread(pathName, 0)

            if "negative" in pathName:
                t = "0"
            else:
                t = "1"

            blurred_img = cv2.GaussianBlur(orig_img, (5, 5), 0)
            edges_img = cv2.Canny(blurred_img, THRESHOLD1, THRESHOLD2)

            nameImage = "image"
            if pathLastName == pathName[:43]:
                n += 1
                nameImage = nameImage + (str(i)) + "" + (str(n))
            else:
                n = 1
                i = i + 1
                nameImage = nameImage + (str(i)) + "" + (str(n))
                pathLastName = pathName[:43]

            img_path = nameImage + img_ext
            labels_path = nameImage + txt_ext

            img_path = os.path.join(processed_dataset_root_path, dataset_type, bone_type, "X", img_path)
            labels_path = os.path.join(processed_dataset_root_path, dataset_type, bone_type, "Y", labels_path)

            print("output image: ", img_path)
            print("output labels: ", labels_path)

            if not os.path.exists(os.path.dirname(labels_path)):
                os.makedirs(os.path.dirname(labels_path))

            if not os.path.exists(os.path.dirname(img_path)):
                os.makedirs(os.path.dirname(img_path))

            f = open(labels_path, "w+")
            f.write(t)
            f.close()

            cv2.imwrite(img_path, edges_img)

