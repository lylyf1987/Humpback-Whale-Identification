import numpy as np
import pandas as pd
import os
from collections import Counter
import src.data.utils as data_utils
import src.model.utils as model_utils
import keras.backend as K
from sklearn import metrics

def train_oneshot(**param_dict):
    """
    function used for training model
    :param param_dict:
    :return: two numpy array: status and accuracy
    """

    #-------------------------
    # get parameters
    #-------------------------
    base_model = param_dict["base_model"]
    sim = param_dict["sim"]
    model = param_dict["model"]
    label_all = param_dict["label_all"]
    img_array_train = param_dict["img_array_train"]
    code_array_train = param_dict["code_array_train"]
    name_array_train = param_dict["name_array_train"]
    img_array_valid_org = param_dict["img_array_valid_org"]
    code_array_valid_org = param_dict["code_array_valid_org"]
    img_array_ref = param_dict["img_array_ref"]
    code_array_ref = param_dict["code_array_ref"]
    epoch = param_dict["epoch"]
    epoch_rg = param_dict["epoch_rg"]
    batch_size = param_dict["batch_size"]
    lr = param_dict["lr"]
    lr_update_step = param_dict["lr_update_step"]
    lr_update_threshold = param_dict["lr_update_threshold"]
    pair_num = param_dict["pair_num"]
    pair_pre_num = param_dict["pair_pre_num"]
    valid_step = param_dict["valid_step"]
    valid_num = param_dict["valid_num"]
    nw_threshold = param_dict["nw_threshold"]
    model_save_dir = param_dict["model_save_dir"]
    #-----------------
    # generate pairs
    #-----------------
    dm = data_utils.DataManager()
    pair_1_idx_train, pair_2_idx_train, flag_train = dm.gen_pairs_train(code_array_train, pair_num=pair_num)
    print("number of training pairs: {}".format(len(pair_1_idx_train)))
    ### check incorrect pairs
    idx_train = np.random.choice(range(len(pair_1_idx_train)), 100)
    for i in idx_train:
        if flag_train[i] == 1 and (
                label_all.loc[label_all.Image == name_array_train[pair_1_idx_train[i]], "Id"].values[0] !=
                label_all.loc[label_all.Image == name_array_train[pair_2_idx_train[i]], "Id"].values[0]):
            print("train not same:", name_array_train[pair_1_idx_train[i]], name_array_train[pair_2_idx_train[i]])
        if flag_train[i] == 0 and (
                label_all.loc[label_all.Image == name_array_train[pair_1_idx_train[i]], "Id"].values[0] ==
                label_all.loc[label_all.Image == name_array_train[pair_2_idx_train[i]], "Id"].values[0]):
            print("train not diff:", name_array_train[pair_1_idx_train[i]], name_array_train[pair_2_idx_train[i]])

    #----------------------
    # prepare for training
    #----------------------
    ### metrics
    accuracy = []
    status = []

    ### create training monitor
    monitor_loss = model_utils.TrainMonitor(xlim=(0, int(epoch * (len(pair_1_idx_train) / batch_size))))
    monitor_accuracy = model_utils.TrainMonitor(color="blue", xlim=(0, int(epoch * (len(pair_1_idx_train) / batch_size))))
    monitor_loss.start()
    monitor_accuracy.start()

    ### create array data loader
    dm = data_utils.DataManager()
    step = 0

    ### learning rate
    K.set_value(model.optimizer.lr, lr)
    print(K.get_value(model.optimizer.lr))

    #--------------------
    # start training
    #--------------------
    for e in range(1, epoch+1):
        data_loader = dm.pdata_loader_train(img_array_train, name_array_train, pair_1_idx_train, pair_2_idx_train, flag_train, batch_size, shuffle=True)

        ### loop through all pairs
        for dt in data_loader:
            pair_1_img = dt["pair_1_image"] / 255
            pair_2_img = dt["pair_2_image"] / 255
            pair_flag = np.array(dt["pair_flag"]).reshape(-1, 1)
            res = model.train_on_batch([pair_1_img, pair_2_img], pair_flag)
            monitor_loss.update(step, res)
            step += 1
            print(step)
            print(res)

            ### save and validate
            if (step % valid_step) == 0:
                model.save_weights(os.path.join(model_save_dir, "oneshot_weight_{}.h5".format(step)))
                ### randomly select validation data
                valid_idx = np.random.choice(range(len(img_array_valid_org)), valid_num, replace=False)
                img_array_valid = img_array_valid_org[valid_idx, ...]
                code_array_valid = code_array_valid_org[valid_idx]
                ### check weight
                print("unequal weight number base model: {}".format((model.layers[-2].layers[-1].get_weights()[0] != base_model.layers[-1].get_weights()[0]).sum()))
                print("unequal weight number sim model: {}".format((model.layers[-1].layers[-1].get_weights()[0] != sim.layers[-1].get_weights()[0]).sum()))
                ### extract features for validation images and reference images
                pair_1_features = base_model.predict(img_array_valid/255)
                pair_2_features = base_model.predict(img_array_ref/255)
                n_ref = len(img_array_ref)
                y_hat_valid_code = []
                ### get predicted code for each validation image
                for i in range(len(img_array_valid)):
                    print(i)
                    pair_1 = np.stack([pair_1_features[i, ]]*n_ref, axis=0)
                    probs = sim.predict([pair_1, pair_2_features])
                    probs = probs.reshape(-1,)
                    temp = pd.DataFrame({"code": code_array_ref, "prob": probs})
                    temp_sum = temp.groupby("code").max()
                    temp_sum = temp_sum.reset_index("code")
                    temp_sum = temp_sum.sort_values("prob", ascending=False)
                    if temp_sum["prob"].iloc[0] < nw_threshold:
                        y_hat_valid_code.append(0)
                    else:
                        y_hat_valid_code.append(temp_sum["code"].iloc[0])
                    print("true code: {}     predicted code: {}".format(code_array_valid[i], y_hat_valid_code[-1]))
                ### calculate metric
                y_hat_valid_code = np.array(y_hat_valid_code)
                accuracy_temp = (y_hat_valid_code == code_array_valid).sum() / len(code_array_valid)
                wrong_code = code_array_valid[code_array_valid != y_hat_valid_code].tolist()
                print(Counter(wrong_code))
                accuracy.append(accuracy_temp)
                status.append((e, step))
                print("in epoch {} after {} steps -------> accuracy is: {}".format(e, step, accuracy))
                print(accuracy)
                print(status)
                monitor_accuracy.update(step, accuracy_temp)
            if (step % lr_update_step) == 0:
                lr = lr_update_threshold * lr
                K.set_value(model.optimizer.lr, lr)
                print(K.get_value(model.optimizer.lr))

        ### re-generate training pairs
        if (e % epoch_rg) == 0:
            ### generate pairs
            pair_1_idx_train, pair_2_idx_train, flag_train = dm.gen_pairs_train(code_array_train, img_array_train, pair_num=pair_num, pair_pre_num=pair_pre_num, model=model)
            print("number of training pairs: {}".format(len(pair_1_idx_train)))
            ### check incorrect pairs
            idx_train = np.random.choice(range(len(pair_1_idx_train)), 100)
            for i in idx_train:
                if flag_train[i] == 1 and (
                        label_all.loc[label_all.Image == name_array_train[pair_1_idx_train[i]], "Id"].values[0] !=
                        label_all.loc[label_all.Image == name_array_train[pair_2_idx_train[i]], "Id"].values[0]):
                    print("train not same:", name_array_train[pair_1_idx_train[i]],
                          name_array_train[pair_2_idx_train[i]])
                if flag_train[i] == 0 and (
                        label_all.loc[label_all.Image == name_array_train[pair_1_idx_train[i]], "Id"].values[0] ==
                        label_all.loc[label_all.Image == name_array_train[pair_2_idx_train[i]], "Id"].values[0]):
                    print("train not diff:", name_array_train[pair_1_idx_train[i]],
                          name_array_train[pair_2_idx_train[i]])
    return np.array(status), np.array(accuracy)

def train_nw(**param_dict):

    ### get parameters
    model = param_dict["model"]
    train_imgs = param_dict["train_imgs"]
    train_names = param_dict["train_names"]
    train_codes = param_dict["train_codes"]
    valid_imgs = param_dict["valid_imgs"]
    valid_codes = param_dict["valid_codes"]
    epoch = param_dict["epoch"]
    batch_size = param_dict["batch_size"]
    lr = param_dict["lr"]
    lr_update_step = param_dict["lr_update_step"]
    lr_update_threshold = param_dict["lr_update_threshold"]
    valid_step = param_dict["valid_step"]
    valid_threshold = param_dict["valid_threshold"]
    model_save_dir = param_dict["model_save_dir"]
    ### metrics
    accuracy = []
    precision = []
    recall = []
    f1 = []
    status = []

    ### create training monitor
    monitor_loss = model_utils.TrainMonitor()
    monitor_accuracy = model_utils.TrainMonitor(color="blue")
    monitor_loss.start()
    monitor_accuracy.start()

    ### create array data loader
    dm = data_utils.DataManager()
    step = 1
    K.set_value(model.optimizer.lr, lr)
    print(K.get_value(model.optimizer.lr))
    for e in range(epoch):
        data_loader = dm.adata_loader(train_imgs, train_names, train_codes, batch_size=batch_size, shuffle=True)
        for dt in data_loader:
            imgs = dt["image"] / 255
            codes = dt["code"]
            res = model.train_on_batch(imgs, codes)
            print(step, res)
            monitor_loss.update(step, res)
            step += 1
            ### save and validate every 2000 steps
            if (step % valid_step) == 0:
                model.save_weights(os.path.join(model_save_dir, "nw_weight_{}.h5".format(step)))
                y_hat = model.predict(valid_imgs/255)
                y_hat = (y_hat > valid_threshold).astype(np.float32)
                acc = metrics.accuracy_score(valid_codes, y_hat)
                accuracy.append(acc)
                precision.append(metrics.precision_score(valid_codes, y_hat))
                recall.append(metrics.recall_score(valid_codes, y_hat))
                f1.append(metrics.f1_score(valid_codes, y_hat))
                status.append((e, step))
                print("in epoch {} after {} steps===================".format(e, step))
                print(accuracy)
                print(precision)
                print(recall)
                print(f1)
                print(status)
                monitor_accuracy.update(step, acc)
            if (step % lr_update_step) == 0:
                lr = lr_update_threshold * lr
                K.set_value(model.optimizer.lr, lr)
                print(K.get_value(model.optimizer.lr))
    return np.array(status), np.array(accuracy), np.array(precision), np.array(recall), np.array(f1)
