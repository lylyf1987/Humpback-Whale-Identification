import numpy as np
import pandas as pd
import os
from src.model.models import oneshot_bsm, oneshot_sim, oneshot_inference
from src.model.train import train_oneshot
import keras.backend as K

config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = K.tf.Session(config=config)

EPOCH = 50
EPOCH_RG = 10
BATCH_SIZE = 32
IMG_SIZE = 160
NW_CODE = 0
NW_NUM = 100
NW_THRESHOLD = 0.9
LEARNING_RATE = 0.00006
LEARNING_RATE_UPDATE_STEP = 5000
LEARNING_RATE_UPDATE_THRESHOLD = 0.99
DENSE_SHAPE = 4096
VALID_STEP = 20000
VALID_NUM = 500
PAIR_NUM = 32
PAIR_PRE_NUM = 64
MODEL_SAVE_DIR = "../model"

def main():

    """
    train the model
    :return:
    """

    #----------------------------
    # load image data and label
    #----------------------------
    label_all = pd.read_csv("../data/processed/label_all.csv")
    img_array = np.load("../data/processed/image_array.npy")
    code_array = np.load("../data/processed/code_array.npy")
    name_array = np.load("../data/processed/name_array.npy")

    #----------------------------------------------
    # get reference, training, and validation data
    #----------------------------------------------
    ### select the top images in each class for reference
    best_img_names = label_all.loc[(label_all["img_rank"] >= 1) & (label_all["img_rank"] <= 5), "Image"]
    best_img_idx = []
    for name in best_img_names:
        best_img_idx.append(np.where(name_array == name)[0][0])
    img_array_ref = img_array[best_img_idx, ...]
    code_array_ref = code_array[best_img_idx]
    name_array_ref = name_array[best_img_idx]
    print(img_array_ref.shape)
    print(code_array_ref.shape)
    print(name_array_ref.shape)
    ### select images from each class for validation
    valid_idx = []

    for code in np.unique(code_array):
        img_idx = np.where(code_array == code)[0]
        ### select 100 images form new_whale
        if code == 0:
            ### eliminate the top images used in reference
            best_img_name = label_all.loc[(label_all["code"] == code) & ((label_all["img_rank"] >= 1) & (label_all["img_rank"] <= 3)), "Image"].values
            best_idx = np.where(np.isin(name_array, best_img_name))[0]
            img_idx = img_idx[~np.isin(img_idx, best_idx)]
            valid_idx.append(np.random.choice(img_idx, 100, replace=False))
        ### select 10 images for code with more than 40 images
        elif len(img_idx) > 40:
            best_img_name = label_all.loc[(label_all["code"] == code) & ((label_all["img_rank"] >= 1) & (label_all["img_rank"] <= 3)), "Image"].values
            best_idx = np.where(np.isin(name_array, best_img_name))[0]
            img_idx = img_idx[~np.isin(img_idx, best_idx)]
            valid_idx.append(np.random.choice(img_idx, 10, replace=False))
        ### select 5 images for code with more than 20 less than 40 images
        elif len(img_idx) <= 40 and len(img_idx) > 20:
            best_img_name = label_all.loc[(label_all["code"] == code) & ((label_all["img_rank"] >= 1) & (label_all["img_rank"] <= 3)), "Image"].values
            best_idx = np.where(np.isin(name_array, best_img_name))[0]
            img_idx = img_idx[~np.isin(img_idx, best_idx)]
            valid_idx.append(np.random.choice(img_idx, 5, replace=False))
        ### select 1 image for code with less than 20 images
        else:
            best_img_name = label_all.loc[(label_all["code"] == code) & ((label_all["img_rank"] >= 1) & (label_all["img_rank"] <= 3)), "Image"].values
            best_idx = np.where(np.isin(name_array, best_img_name))[0]
            img_idx = img_idx[~np.isin(img_idx, best_idx)]
            valid_idx.append(np.random.choice(img_idx, 1, replace=False))

    valid_idx = np.concatenate(valid_idx, axis=0)
    img_array_train = np.delete(img_array, valid_idx, axis=0)
    code_array_train = np.delete(code_array, valid_idx, axis=0)
    name_array_train = np.delete(name_array, valid_idx, axis=0)
    img_array_valid = img_array[valid_idx, ...]
    code_array_valid = code_array[valid_idx]
    name_array_valid = name_array[valid_idx]
    del img_array

    #-------------
    # get model
    #-------------
    base_model = oneshot_bsm()
    sim = oneshot_sim()
    model = oneshot_inference(base_model, sim)
    #model.load_weights("../model_keras/model_oneshot_120000.h5")

    #---------------
    # training
    #---------------
    param_dict = {"base_model": base_model,
                  "sim": sim,
                  "model": model,
                  "label_all": label_all,
                  "img_array_train": img_array_train,
                  "code_array_train": code_array_train,
                  "name_array_train": name_array_train,
                  "img_array_valid_org": img_array_valid,
                  "code_array_valid_org": code_array_valid,
                  "img_array_ref": img_array_ref,
                  "code_array_ref": code_array_ref,
                  "epoch": EPOCH,
                  "epoch_rg": EPOCH_RG,
                  "batch_size": BATCH_SIZE,
                  "lr": LEARNING_RATE,
                  "lr_update_step": LEARNING_RATE_UPDATE_STEP,
                  "lr_update_threshold": LEARNING_RATE_UPDATE_THRESHOLD,
                  "pair_num": PAIR_NUM,
                  "pair_pre_num": PAIR_PRE_NUM,
                  "valid_step": VALID_STEP,
                  "valid_num": VALID_NUM,
                  "nw_threshold": NW_THRESHOLD,
                  "model_save_dir": MODEL_SAVE_DIR}
    status, accuracy = train_oneshot(**param_dict)
    np.save(os.path.join(MODEL_SAVE_DIR, "oneshot_status.npy"), status)
    np.save(os.path.join(MODEL_SAVE_DIR, "oneshot_accuracy.npy"), accuracy)

if __name__ == "__main__":
    main()