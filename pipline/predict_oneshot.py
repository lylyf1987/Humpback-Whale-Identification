import numpy as np
import pandas as pd
from src.model.models import oneshot_bsm, oneshot_sim, oneshot_inference
from src.model.predict import predict_oneshot

def main():

    """
    make prediction
    :return:
    """

    #----------------------------
    # load image data and label
    #----------------------------
    label_all = pd.read_csv("../data/processed/label_all.csv")
    img_array = np.load("../data/processed/image_array.npy")
    code_array = np.load("../data/processed/code_array.npy")
    name_array = np.load("../data/processed/name_array.npy")
    img_array_test = np.load("../data/processed/image_array_test.npy")
    name_array_test = np.load("../data/processed/name_array_test.npy")

    #-------------------
    # get reference data
    #-------------------
    ### select the top images in each class for reference
    best_img_names = label_all.loc[(label_all["img_rank"] >= 1) & (label_all["img_rank"] <= 10), "Image"]
    best_img_idx = []
    for name in best_img_names:
        best_img_idx.append(np.where(name_array == name)[0][0])
    img_array_ref = img_array[best_img_idx, ...]
    code_array_ref = code_array[best_img_idx]

    #------------------
    # get best model
    #-------------------
    accuracy = np.load("../model/oneshot_accuracy.npy")
    status = np.load("../model/oneshot_status.npy")
    step = status[np.argmax(accuracy)][1]
    base_model = oneshot_bsm()
    sim = oneshot_sim()
    model = oneshot_inference(base_model, sim)
    model.load_weights("../model/oneshot_weight_{}.h5".format(step))

    #-----------------
    # predict
    #-----------------
    submission = predict_oneshot(base_model, sim, img_array_test, name_array_test, img_array_ref, code_array_ref, label_all)

    #-----------------------
    # detect new whale
    #------------------------
    for i in submission.index:
        ID = submission.loc[i, "Id"].split(" ")
        if submission.loc[i, "prob_top_5"] < 0.997:
            submission.loc[i, "Id"] = " ".join(["new_whale", *ID[0:4]])

    submission = submission[["Image", "Id"]]
    submission.to_csv("../submission/sub_oneshot.csv", index=False)

if __name__ == "__main__":
    main()
