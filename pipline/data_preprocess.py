import numpy as np
import pandas as pd
import src.data.utils as utils
import os
from skimage import io
import shutil

IMG_SIZE = 160
TRAIN_IMG_DIR_ORG = "../data/raw/train"
TEST_IMG_DIR_ORG = "../data/raw/test"
TRAIN_IMG_DIR = "../data/processed/train_modified"
TEST_IMG_DIR = "../data/processed/test_modified"
TRAIN_IMG_DIR_AUG = "../data/processed/train_aug"
IMG_ARRAY_PATH = "../data/processed/img_array.npy"
CODE_ARRAY_PATH = "../data/processed/code_array.npy"
NAME_ARRAY_PATH = "../data/processed/name_array.npy"
IMG_ARRAY_TEST_PATH = "../data/processed/img_array_test.npy"
NAME_ARRAY_TEST_PATH = "../data/processed/name_array_test.npy"
LABEL_ALL_PATH = "../data/processed/label_all.csv"

bounding_box = pd.read_csv("../data/external/bounding_boxes.csv")
label_df = pd.read_csv("../data/raw/train.csv")

#=======================
# image modification
#=======================
img_manager = utils.ImageManager()
if not os.path.exists(TRAIN_IMG_DIR):
    img_manager.img_modify(TRAIN_IMG_DIR_ORG, TRAIN_IMG_DIR, IMG_SIZE, bounding_box)
if not os.path.exists(TEST_IMG_DIR):
    img_manager.img_modify(TEST_IMG_DIR_ORG, TEST_IMG_DIR, IMG_SIZE, bounding_box)

#========================
# label preprocessing
#========================
### get the images that are both in the folder and label file ###
folder_img_names = set(os.listdir(TRAIN_IMG_DIR_ORG))
label_img_names = set(label_df["Image"].values)
valid_img_names = np.array(list(folder_img_names.intersection(label_img_names)))
label_valid = label_df.loc[label_df["Image"].isin(valid_img_names), :]

### rank the quanlity of images in each class ###
for img_name in label_valid["Image"]:
    img = io.imread(os.path.join(TRAIN_IMG_DIR_ORG, img_name))
    resolution = img.shape[0] * img.shape[1]
    label_valid.loc[label_valid.Image == img_name, "reso"] = resolution
label_valid = label_valid.sort_values(by=["Id", "reso"], ascending=[True, False])
label_valid["img_rank"] = -1
pre_id = ''
for row in range(len(label_valid)):
    print(row)
    if label_valid["Id"].iloc[row] != pre_id:
        pre_id = label_valid["Id"].iloc[row]
        rank = 1
    label_valid["img_rank"].iloc[row] = rank
    rank += 1

#=======================
# image augmentation
#=======================
### get number of samples for each label ###
label_count = label_valid["Id"].value_counts()

### data augmentation for label with small number of samples (less than 5) ###
label_to_num = {label: (5 - label_count[label]) for label in label_valid["Id"].unique() if label_count[label] < 5}
label_aug = img_manager.img_aug(label_valid, label_to_num, TRAIN_IMG_DIR, TRAIN_IMG_DIR_AUG)

### copy all image from train_modified to train_aug ###
for img_name in os.listdir(TRAIN_IMG_DIR):
    img_file = os.path.join(TRAIN_IMG_DIR, img_name)
    shutil.copy(img_file, TRAIN_IMG_DIR_AUG)
label_all = pd.concat([label_valid, label_aug], axis=0, ignore_index=True)

### get label to code dict ###
unique_labels = np.sort(label_all["Id"].unique())
label_to_code = {l: c for c, l in enumerate(unique_labels)}
for i in label_all.index:
    label_all.loc[i, "code"] = label_to_code[label_all.loc[i, "Id"]]
label_all["code"] = label_all["code"].astype(np.int32)
label_all.loc[label_all["img_rank"].isna(), "img_rank"] = -1
label_all.to_csv(LABEL_ALL_PATH, index=False)

#===================================
# save original image to numpy array
#===================================
### save all training images, names and codes to array ###
if not os.path.exists(IMG_ARRAY_PATH):
    img_array = []
    name_array = []
    code_array = []
    dm = utils.DataManager()
    data_loader = dm.fdata_loader(TRAIN_IMG_DIR_AUG, label_all, batch_size=300, shuffle=False)
    step = 1
    for dt in data_loader:
        print(step)
        imgs = dt["image"]
        codes = dt["code"]
        names = dt["name"]
        img_array.append(imgs)
        code_array.append(codes)
        name_array.append(names)
        step += 1
    img_array = np.concatenate(img_array, axis=0)
    code_array = np.concatenate(code_array, axis=0)
    name_array = np.concatenate(name_array, axis=0)
    np.save(IMG_ARRAY_PATH, img_array)
    np.save(CODE_ARRAY_PATH, code_array)
    np.save(NAME_ARRAY_PATH, name_array)

### save all testing images, names to array ###
if not os.path.exists(IMG_ARRAY_TEST_PATH):
    img_array = []
    name_array = []
    data_loader = dm.fdata_loader(TEST_IMG_DIR, batch_size=300, shuffle=False)
    step = 1
    for dt in data_loader:
        print(step)
        imgs = dt["image"]
        names = dt["name"]
        img_array.append(imgs)
        name_array.append(names)
        step += 1
    img_array = np.concatenate(img_array, axis=0)
    name_array = np.concatenate(name_array, axis=0)
    np.save(IMG_ARRAY_TEST_PATH, img_array)
    np.save(NAME_ARRAY_TEST_PATH, name_array)