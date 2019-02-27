import numpy as np
import pandas as pd

def predict_single_oneshot(i, pair_1_feature, name, pair_2_features, code_array_ref, sim, label_all):

    """
    predict top five image labels for one image
    """

    n_ref = len(pair_2_features)
    pair_1 = np.stack([pair_1_feature] * n_ref, axis=0)
    probs = sim.predict([pair_1, pair_2_features])
    probs = probs.reshape(-1, )
    temp = pd.DataFrame({"code": code_array_ref, "prob": probs})
    temp_sum = temp.groupby("code").max()
    temp_sum = temp_sum.reset_index("code")
    temp_sum = temp_sum.sort_values("prob", ascending=False)
    label = []
    for j in range(5):
        label.append(label_all.loc[label_all["code"] == temp_sum["code"].iloc[j], "Id"].values[0])
    label = " ".join(label)
    code_top = temp_sum["code"].iloc[0:5]
    prob_top = temp_sum["prob"].iloc[0:5]
    code_top_1 = code_top.iloc[0]
    code_top_2 = code_top.iloc[1]
    code_top_3 = code_top.iloc[2]
    code_top_4 = code_top.iloc[3]
    code_top_5 = code_top.iloc[4]
    prob_top_1 = prob_top.iloc[0]
    prob_top_2 = prob_top.iloc[1]
    prob_top_3 = prob_top.iloc[2]
    prob_top_4 = prob_top.iloc[3]
    prob_top_5 = prob_top.iloc[4]
    return (name, label, code_top_1, code_top_2, code_top_3, code_top_4, code_top_5, prob_top_1, prob_top_2, prob_top_3, prob_top_4, prob_top_5)

def predict_oneshot(base_model, sim, img_array_test, name_array_test, img_array_ref, code_array_ref, label_all):
    """
    make prediction for all test images
    :param base_model:
    :param sim:
    :param img_array_test:
    :param name_array_test:
    :param img_array_ref:
    :param code_array_ref:
    :param label_all:
    :return:
    """

    #--------------------
    # get features
    #--------------------
    pair_1_features = base_model.predict(img_array_test / 255)
    pair_2_features = base_model.predict(img_array_ref / 255)

    #----------------
    # predict
    #----------------
    res = []
    for i in range(len(pair_1_features)):
        print(i)
        res.append(predict_single_oneshot(i, pair_1_features[i, ...], name_array_test[i], pair_2_features, code_array_ref, sim, label_all))
    name, label_predict, code_top_1, code_top_2, code_top_3, code_top_4, code_top_5, prob_top_1, prob_top_2, prob_top_3, prob_top_4, prob_top_5 = zip(*res)
    submission = pd.DataFrame({"Image": name,
                               "Id": label_predict,
                               "code_top_1": code_top_1,
                               "code_top_2": code_top_2,
                               "code_top_3": code_top_3,
                               "code_top_4": code_top_4,
                               "code_top_5": code_top_5,
                               "prob_top_1": prob_top_1,
                               "prob_top_2": prob_top_2,
                               "prob_top_3": prob_top_3,
                               "prob_top_4": prob_top_4,
                               "prob_top_5": prob_top_5})

    return submission

def predict_nw(model, img_array_test):

    """
    make prediction
    :param model:
    :param img_array_test:
    :param name_array_test:
    :return:
    """
    res_prob = model.predict(img_array_test)
    return res_prob