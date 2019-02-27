import numpy as np
import os
from sklearn.model_selection import train_test_split
from src.model.models import nw_inference
from src.model.train import train_nw
import keras.backend as K

config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = K.tf.Session(config=config)

EPOCH = 10
BATCH_SIZE = 64
IMG_SIZE = 160
LEARNING_RATE = 0.0001
LEARNING_RATE_UPDATE_STEP = 500
LEARNING_RATE_UPDATE_THRESHOLD = 0.99
VALID_STEP = 1000
VALID_THRESHOLD = 0.7
MODEL_SAVE_DIR = "../model"

def main():

    """
    train the model
    :return:
    """

    #---------------------------
    # load image data and label
    #---------------------------
    img_array = np.load("../data/processed/image_array.npy")
    code_array = np.load("../data/processed/code_array.npy")
    name_array = np.load("../data/processed/name_array.npy")
    img_array_test = np.load("../data/processed/image_array_test.npy")
    name_array_test = np.load("../data/processed/name_array_test.npy")

    #-------------
    # get model
    #-------------
    model = nw_inference()

    #---------------
    # train model
    #---------------
    ### generate data
    code_array_cp = code_array.copy()
    code_array_cp[code_array_cp != 0] = -1
    code_array_cp[code_array_cp == 0] = 1
    code_array_cp[code_array_cp == -1] = 0
    train_imgs, valid_imgs, train_names, valid_names, train_codes, valid_codes = train_test_split(img_array, name_array, code_array_cp, test_size=0.03, stratify=code_array_cp)

    ### train
    param_dict = {"model": model,
                  "train_imgs": train_imgs,
                  "train_names": train_names,
                  "train_codes": train_codes,
                  "valid_imgs": valid_imgs,
                  "valid_codes": valid_codes,
                  "epoch": EPOCH,
                  "batch_size": BATCH_SIZE,
                  "lr": LEARNING_RATE,
                  "lr_update_step": LEARNING_RATE_UPDATE_STEP,
                  "lr_update_threshold": LEARNING_RATE_UPDATE_THRESHOLD,
                  "valid_step": VALID_STEP,
                  "valid_threshold": VALID_THRESHOLD,
                  "model_save_dir": MODEL_SAVE_DIR}
    status, accuracy, precision, recall, f1 = train_nw(**param_dict)
    np.save(os.path.join(MODEL_SAVE_DIR, "nw_status.npy"), status)
    np.save(os.path.join(MODEL_SAVE_DIR, "nw_accuracy.npy"), accuracy)
    np.save(os.path.join(MODEL_SAVE_DIR, "nw_precision.npy"), precision)
    np.save(os.path.join(MODEL_SAVE_DIR, "nw_recall.npy"), recall)
    np.save(os.path.join(MODEL_SAVE_DIR, "nw_f1.npy"), f1)

if __name__ == "__main__":
    main()