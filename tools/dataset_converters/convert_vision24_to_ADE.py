import os
import shutil
import random
import cv2  # OpenCV for image processing

def convert_labels(src_path, dest_path):
    # Load the image
    label_img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    
    # Convert 255 to 1
    label_img[label_img == 255] = 1
    
    # Save the modified label
    cv2.imwrite(dest_path, label_img)
    
def convert_to_ade_format(src_dir, dest_dir, train_split=0.8):
    image_train_dest = os.path.join(dest_dir, "images", "training")
    anno_train_dest = os.path.join(dest_dir, "annotations", "training")
    image_val_dest = os.path.join(dest_dir, "images", "validation")
    anno_val_dest = os.path.join(dest_dir, "annotations", "validation")

    os.makedirs(image_train_dest, exist_ok=True)
    os.makedirs(anno_train_dest, exist_ok=True)
    os.makedirs(image_val_dest, exist_ok=True)
    os.makedirs(anno_val_dest, exist_ok=True)

    for category in os.listdir(src_dir):
        category_path = os.path.join(src_dir, category)
        if os.path.isdir(category_path):
            for defect in os.listdir(category_path):
                defect_path = os.path.join(category_path, defect)
                if os.path.isdir(defect_path):
                    image_path = os.path.join(defect_path, "image")
                    label_path = os.path.join(defect_path, "label")

                    if os.path.exists(image_path) and os.path.exists(label_path):
                        files = os.listdir(image_path)
                        random.shuffle(files)

                        train_count = int(len(files) * train_split)
                        train_files = files[:train_count]
                        val_files = files[train_count:]

                        # Copy training files
                        for img_file in train_files:
                            img_src = os.path.join(image_path, img_file)
                            label_src = os.path.join(
                                label_path, img_file).replace('.jpg', '.png')

                            img_dest_name = f"{category}_{defect}_{img_file}"
                            label_dest_name = f"{category}_{defect}_{os.path.splitext(img_file)[0]}.png"

                            shutil.copy(img_src, os.path.join(
                                image_train_dest, img_dest_name))
                            shutil.copy(label_src, os.path.join(
                                anno_train_dest, label_dest_name))
                            convert_labels(label_src, os.path.join(anno_train_dest, label_dest_name))

                        # Copy validation files
                        for img_file in val_files:
                            img_src = os.path.join(image_path, img_file)
                            label_src = os.path.join(
                                label_path, img_file).replace('.jpg', '.png')

                            img_dest_name = f"{category}_{defect}_{img_file}"
                            label_dest_name = f"{category}_{defect}_{os.path.splitext(img_file)[0]}.png"

                            shutil.copy(img_src, os.path.join(
                                image_val_dest, img_dest_name))
                            shutil.copy(label_src, os.path.join(
                                anno_val_dest, label_dest_name))
                            convert_labels(label_src, os.path.join(anno_val_dest, label_dest_name))


# Example usage
src_directory = "/home/m32patel/projects/rrg-dclausi/vision24/VISION24-data-challenge-train/train"
dest_directory = "/home/m32patel/projects/rrg-dclausi/vision24/VISION24-data-challenge-train/ADE_format"

convert_to_ade_format(src_directory, dest_directory, train_split=0.9)
