import numpy as np
import pandas as pd
from src.model.models import nw_inference
from src.model.predict import predict_nw


def main():

    """
    make prediction
    :return:
    """

    #----------------------------
    # load image data and label
    #----------------------------
    img_array_test = np.load("../data/processed/image_array_test.npy")
    name_array_test = np.load("../data/processed/name_array_test.npy")

    #-----------------
    # get best model
    #------------------
    f1 = np.load("../model/nw_f1.npy")
    status = np.load("../model/nw_status.npy")
    step = status[np.argmax(f1)][1]
    model = nw_inference()
    model.load_weights("../model/nw_weight_{}.h5".format(step))

    #----------------
    # predict
    #----------------
    res_prob = predict_nw(model, img_array_test)
    res_prob = res_prob.reshape(-1, )

    np.save("../submission/nw_prob.npy", res_prob)
    np.save("../submission/nw_name.npy", name_array_test)