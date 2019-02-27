import numpy as np
import pandas as pd
import os
from skimage import io, color, transform, util
import multiprocessing as mp


class DataManager():
    """
    a class for data loading
    """

    def __init__(self):
        pass

    def fdata_loader(self, input_dir, label_df=None, batch_size=64, shuffle=False):

        """
        parallely load data from an image folder and go through all images once

        :param input_dir: input folder
        :param label_df: image label (DataFrame)
        :param batch_size: batch size
        :param shuffle: shuflle flag
        :return: a dictionary with keys "image", "code", and "name". "image": [batch size, image size, image size, 3]; "code": [batch size,]; "name": [batch size, ] for training
                or a dictionary with keys "image" and "name" for testing
        """

        #------------------------
        # read data for training
        #------------------------
        if label_df is not None:

            ### get image file paths and label ###
            label_df_cp = label_df.copy()
            img_names = label_df_cp["Image"].values
            n = len(img_names)
            img_files = list(map(lambda x: os.path.join(input_dir, x), img_names))
            # set image names as index
            label_df_cp = label_df_cp.set_index("Image")
            # shuffle all training files
            if shuffle:
                idx = np.random.permutation(range(n))
                img_files = img_files[idx]

            ### read image in batch ###
            start = 0
            end = start + batch_size - 1
            if end >= n:
                end = n - 1
            # loop through all image files ###
            while start <= (n - 1):
                try:
                    img_info = []
                    # get each image and the corresponding label
                    with mp.Pool(processes=mp.cpu_count()) as pool:
                        for file in img_files[start:(end+1)]:
                            img_info.append(pool.apply_async(self._fdata_loader_single, args=(file, label_df_cp,)).get())
                    imgs, codes, names = zip(*img_info)
                    assert len(imgs) == len(codes)
                    start = end + 1
                    end = start + batch_size - 1
                    if end > (n - 1):
                        end = n - 1
                    yield {"image": np.array(imgs), "code": np.array(codes), "name": np.array(names)}
                except AssertionError:
                    print("issue")
        #-----------------------
        # read data for testing
        #-----------------------
        else:

            ### get image files paths ###
            img_names = os.listdir(input_dir)
            n = len(img_names)
            img_files = list(map(lambda x: os.path.join(input_dir, x), img_names))

            ### read image in batch ###
            start = 0
            end = start + batch_size - 1
            if end >= n:
                end = n - 1
            # loop through all image files
            while start <= (n - 1):
                try:
                    img_info = []
                    # get each image
                    with mp.Pool(processes=mp.cpu_count()) as pool:
                        for file in img_files[start:(end+1)]:
                            img_info.append(pool.apply_async(self._fdata_loader_single, args=(file,)).get())
                    imgs, names = zip(*img_info)
                    assert len(imgs) == len(names)
                    start = end + 1
                    end = start + batch_size - 1
                    if end > (n - 1):
                        end = n - 1
                    yield {"image": np.array(imgs), "name": np.array(names)}
                except AssertionError:
                    print("issue")

    def _fdata_loader_single(self, file, label_df_cp=None):

        """
        load a single image

        :param file: image file
        :param label_df_cp: label data frame
        :return: a tuple as (image, code)
        """

        #------------------------
        # read data for training
        #------------------------
        if label_df_cp is not None:
            try:
                img = io.imread(file)
                img_name = os.path.basename(file)
                code = label_df_cp.loc[img_name, "code"]
                return (img, code, img_name)
            except:
                print("error:   {}".format(img_name))
        #-----------------------
        # read data for testing
        #-----------------------
        else:
            try:
                img = io.imread(file)
                img_name = os.path.basename(file)
                return (img, img_name)
            except:
                print("error:   {}".format(img_name))

    def adata_loader(self, img_array, name_array, code_array=None, batch_size=64, shuffle=False):

        """
        load data from image array

        :param img_array: an array of image data
        :param name_array: an array of image names
        :param code_array: an array of image codes
        :param batch_size: batch size
        :param shuffle: shuflle flag
        :return: a dictionary with keys "image", "code", and "name". "image": [batch size, image size, image size, 3]; "code": [batch size,]; "name": [batch size, ] for training
                or a dictionary with keys "image" and "name" for testing
        """

        #-------------------------
        # read data for training
        #-------------------------
        if code_array is not None:
            n = len(img_array)
            ### shuffle all training files
            if shuffle:
                idx = np.random.permutation(range(n))
            else:
                idx = list(range(n))

            ### read image in batch ###
            start = 0
            end = start + batch_size - 1
            if end >= n:
                end = n - 1
            # loop through all images
            while start <= (n - 1):
                try:
                    imgs = img_array[idx[start:(end+1)], ...]
                    names = name_array[idx[start:(end+1)]]
                    codes = code_array[idx[start:(end+1)]]
                    assert len(imgs) == len(codes)
                    start = end + 1
                    end = start + batch_size - 1
                    if end > (n - 1):
                        end = n - 1
                    yield {"image": imgs, "code": codes, "name": names}
                except AssertionError:
                    print("issue")
        #-------------------------
        # read data for testing
        #-------------------------
        else:
            n = len(img_array)

            ### read image in batch ###
            start = 0
            end = start + batch_size - 1
            # loop through all images
            while start <= (n - 1):
                try:
                    imgs = img_array[start:(end+1), ...]
                    names = name_array[start:(end+1)]
                    assert len(imgs) == len(names)
                    start = end + 1
                    end = start + batch_size - 1
                    if end > (n - 1):
                        end = n - 1
                    yield {"image": imgs, "name": names}
                except AssertionError:
                    print("issue")

    def pdata_loader_train(self, img_array, name_array, pair_1_idx, pair_2_idx, flag, batch_size=64, shuffle=False):

        """
        load pair data for training from image array

        :param img_array: an array of image data
        :param name_array: an array of image names
        :param pair_1_idx: index of pair 1 images
        :param pair_2_idx: index of pair 2 images
        :param flag: an array of flags (0 or 1, 0 for different and 1 for same)
        :param batch_size: batch size
        :param shuffle: shuflle flag
        :return: a dictionary with keys "pair_1_image", "pair_2_image", and "pair_flag", "pair_1_name", and "pair_2_name".
        """

        n = len(flag)
        # shuffle all training files
        if shuffle:
            idx = np.random.permutation(range(n))
            pair_1_idx = pair_1_idx[idx]
            pair_2_idx = pair_2_idx[idx]
            flag = flag[idx]

        ### read pairs in batch ###
        start = 0
        end = start + batch_size - 1
        if end >= n:
            end = n - 1
        # loop through all pairs
        while start <= (n - 1):
            try:
                pair_1_img = img_array[pair_1_idx[start:(end+1)], ...]
                pair_2_img = img_array[pair_2_idx[start:(end+1)], ...]
                pair_1_name = name_array[pair_1_idx[start:(end+1)]]
                pair_2_name = name_array[pair_2_idx[start:(end+1)]]
                pair_flag = flag[start:(end+1)]
                assert len(pair_1_img) == len(pair_2_img)
                assert len(pair_1_img) == len(pair_flag)
                start = end + 1
                end = start + batch_size - 1
                if end > (n - 1):
                    end = n - 1
                yield {"pair_1_image": pair_1_img, "pair_2_image": pair_2_img, "pair_flag": pair_flag, "pair_1_name": pair_1_name, "pair_2_name": pair_2_name}
            except AssertionError:
                print("issue")

    def pdata_loader_valid(self, img_array_ref, name_array_ref, img_array_valid, name_array_valid, pair_1_idx, pair_2_idx, flag, batch_size=64, shuffle=False):

        """
        load pair data for testing from image array

        :param img_array_ref: an array of reference image
        :param name_array_ref: an array of reference image names
        :param img_array_valid: an array of validation image
        :param name_array_valid: an array of validation image names
        :param pair_1_idx: index of pair 1 images
        :param pair_2_idx: index of pair 2 images
        :param flag: an array of flags (0 or 1)
        :param batch_size: batch size
        :param shuffle: shuflle flag
        :return: a dictionary with keys "pair_1_image", "pair_2_image", and "pair_flag", "pair_1_name", and "pair_2_name".
        """

        n = len(flag)
        # shuffle all training files
        if shuffle:
            idx = np.random.permutation(range(n))
            pair_1_idx = pair_1_idx[idx]
            pair_2_idx = pair_2_idx[idx]
            flag = flag[idx]

        ### read pairs in batch ###
        start = 0
        end = start + batch_size - 1
        if end >= n:
            end = n - 1
        # loop through all pairs
        while start <= (n - 1):
            try:
                pair_1_img = img_array_valid[pair_1_idx[start:(end+1)], ...]
                pair_2_img = img_array_ref[pair_2_idx[start:(end+1)], ...]
                pair_1_name = name_array_valid[pair_1_idx[start:(end+1)]]
                pair_2_name = name_array_ref[pair_2_idx[start:(end+1)]]
                pair_flag = flag[start:(end+1)]
                assert len(pair_1_img) == len(pair_2_img)
                assert len(pair_1_img) == len(pair_flag)
                start = end + 1
                end = start + batch_size - 1
                if end > (n - 1):
                    end = n - 1
                yield {"pair_1_image": pair_1_img, "pair_2_image": pair_2_img, "pair_flag": pair_flag, "pair_1_name": pair_1_name, "pair_2_name": pair_2_name}
            except AssertionError:
                print("issue")

    def gen_pairs_train(self, code_array, img_array=None, pair_num=4, pair_pre_num=10, model=None):

        """
        generate pairs data for training

        :param code_array: code array
        :param img_array: image array
        :param pair_num: pair number
        :param pair_pre_num: number of pairs used for random selection
        :param model: trained model
        :return: 3 numpy arrays, pair_1_idx, pair_2_idx, flag
        """

        pair_1_idx = []
        pair_2_idx = []
        flag = []
        unique_code = np.unique(code_array)
        #--------------------------------------------------
        # generate pairs without selecting the worst pairs
        #--------------------------------------------------
        if model is None:
            ### generate pairs for each class ###
            for code in unique_code:
                ### eliminate new_whale
                if code != 0:
                    ### generate different images in pair 2 ###
                    diff_code_idx_rd = []
                    """
                    # randomly select different classes
                    if len(unique_code[unique_code != code]) >= pair_num:
                        diff_code_rd = np.random.choice(unique_code[unique_code != code], pair_num, replace=False)
                    else:
                        diff_code_rd = np.random.choice(unique_code[unique_code != code], pair_num, replace=True)
                    # for each different class, randomly select one image
                    for code_diff in diff_code_rd:
                        diff_code_idx = np.where(code_array == code_diff)[0]
                        if len(diff_code_idx) >= 1:
                            diff_code_idx_rd.append(np.random.choice(diff_code_idx, 1, replace=False))
                        else:
                            print("pair generation error for pair 2 diff code: {}".format(code_diff))
                    diff_code_idx_rd = np.concatenate(diff_code_idx_rd, axis=0).tolist()
                    """
                    # randomly select different classes
                    if len(unique_code[(unique_code != code) & (unique_code != 0)]) >= (pair_num - 10):
                        diff_code_rd = np.random.choice(unique_code[unique_code != code], pair_num-10, replace=False)
                    else:
                        diff_code_rd = np.random.choice(unique_code[unique_code != code], pair_num-10, replace=True)
                    # for each different class, randomly select one image (except 10 images from new whale)
                    diff_code_idx_rd.append(np.random.choice(np.where(code_array == 0)[0], 10, replace=False))
                    for code_diff in diff_code_rd:
                        diff_code_idx = np.where(code_array == code_diff)[0]
                        if len(diff_code_idx) >= 1:
                            diff_code_idx_rd.append(np.random.choice(diff_code_idx, 1, replace=False))
                        else:
                            print("pair generation error for pair 2 diff code: {}".format(code_diff))
                    diff_code_idx_rd = np.concatenate(diff_code_idx_rd, axis=0).tolist()

                    ### generate same images in pair 2 ###
                    same_code_idx = np.where(code_array == code)[0]
                    # randomly select images with same code
                    if len(same_code_idx) >= pair_num:
                        same_code_idx_rd = np.random.choice(same_code_idx, pair_num, replace=False).tolist()
                    else:
                        same_code_idx_rd = np.random.choice(same_code_idx, pair_num, replace=True).tolist()

                    ### generate same images in pair 1 ###
                    idx_1_same = []
                    # for each image used for same pairs in pair 2, randomly select one different image with the same code for pair 1
                    for idx in same_code_idx_rd:
                        same_code_idx_rv = same_code_idx[same_code_idx != idx]
                        if len(same_code_idx_rv) >= 1:
                            idx_1_same.append(np.random.choice(same_code_idx_rv, 1, replace=False))
                        else:
                            print("pair generation error for pair 1 same code: {}".format(code))
                    idx_1_same = np.concatenate(idx_1_same, axis=0).tolist()

                    ### generate different images in pair 1 ###
                    if len(same_code_idx) >= pair_num:
                        idx_1_diff = np.random.choice(same_code_idx, pair_num, replace=False).tolist()
                    else:
                        idx_1_diff = np.random.choice(same_code_idx, pair_num, replace=True).tolist()

                    ### create pairs ###
                    pair_1_idx.extend(idx_1_same)
                    pair_2_idx.extend(same_code_idx_rd)
                    pair_1_idx.extend(idx_1_diff)
                    pair_2_idx.extend(diff_code_idx_rd)
                    flag.extend([1.0] * pair_num)
                    flag.extend([0.0] * pair_num)
                    # check pairs
                    if np.any(code_array[same_code_idx_rd] != code):
                        print("error: same_code_idx_rd")
                    if np.any(code_array[diff_code_idx_rd] == code):
                        print("error: diff_code_idx_rd")
                    if np.any(code_array[idx_1_same] != code):
                        print("error: idx_1_same")
                    if np.any(code_array[idx_1_diff] != code):
                        print("error: idx_1_diff")
            assert len(pair_1_idx) == len(pair_2_idx)
            assert len(pair_1_idx) == len(flag)
            return np.array(pair_1_idx), np.array(pair_2_idx), np.array(flag)
        # --------------------------------------------------
        # generate pairs by selecting the worst pairs
        # --------------------------------------------------
        else:
            ### generate pairs for each class ###
            for code in unique_code:
                if code != 0:
                    print(code)

                    ### generate different images for selection in pair 2 ###
                    diff_code_idx_rd = []
                    """
                    # randomly select different classes
                    if len(unique_code[unique_code != code]) >= pair_pre_num:
                        diff_code_rd = np.random.choice(unique_code[unique_code != code], pair_pre_num, replace=False)
                    else:
                        diff_code_rd = np.random.choice(unique_code[unique_code != code], pair_pre_num, replace=True)
                    # for each different class, randomly select one image
                    for code_diff in diff_code_rd:
                        diff_code_idx = np.where(code_array == code_diff)[0]
                        if len(diff_code_idx) >= 1:
                            diff_code_idx_rd.append(np.random.choice(diff_code_idx, 1, replace=False))
                        else:
                            print("pair generation error for pair 2 diff code: {}".format(code_diff))
                    """
                    # randomly select different classes
                    if len(unique_code[(unique_code != code) & (unique_code != 0)]) >= (pair_pre_num - 10):
                        diff_code_rd = np.random.choice(unique_code[unique_code != code], pair_pre_num - 10, replace=False)
                    else:
                        diff_code_rd = np.random.choice(unique_code[unique_code != code], pair_pre_num - 10, replace=True)
                    # for each different class, randomly select one image (except 10 images from new whale)
                    diff_code_idx_rd.append(np.random.choice(np.where(code_array == 0)[0], 10, replace=False))
                    for code_diff in diff_code_rd:
                        diff_code_idx = np.where(code_array == code_diff)[0]
                        if len(diff_code_idx) >= 1:
                            diff_code_idx_rd.append(np.random.choice(diff_code_idx, 1, replace=False))
                        else:
                            print("pair generation error for pair 2 diff code: {}".format(code_diff))
                    diff_code_idx_rd = np.concatenate(diff_code_idx_rd, axis=0)

                    ### generate same images for selection in pair 2 ###
                    same_code_idx = np.where(code_array == code)[0]
                    # randomly select images with same code
                    if len(same_code_idx) >= pair_pre_num:
                        same_code_idx_rd = np.random.choice(same_code_idx, pair_pre_num, replace=False)
                    else:
                        same_code_idx_rd = np.random.choice(same_code_idx, pair_pre_num, replace=True)

                    ### generate same images for selection in pair 1 ###
                    idx_1_same = []
                    # for each image used for same pairs in pair 2, randomly select one different image with the same code for pair 1
                    for idx in same_code_idx_rd:
                        same_code_idx_rv = same_code_idx[same_code_idx != idx]
                        if len(same_code_idx_rv) >= 1:
                            idx_1_same.append(np.random.choice(same_code_idx_rv, 1, replace=False))
                        else:
                            print("pair generation error for code: {}".format(code))
                    idx_1_same = np.concatenate(idx_1_same, axis=0)

                    ### generate different images for selection in pair 1 ###
                    if len(same_code_idx) >= pair_pre_num:
                        idx_1_diff = np.random.choice(same_code_idx, pair_pre_num, replace=False)
                    else:
                        idx_1_diff = np.random.choice(same_code_idx, pair_pre_num, replace=True)

                    ### calculate the similarity between pairs ###
                    assert len(idx_1_same) == len(same_code_idx_rd)
                    assert len(idx_1_diff) == len(diff_code_idx_rd)
                    prob_diff = model.predict([img_array[idx_1_diff, ...], img_array[diff_code_idx_rd, ...]])
                    prob_same = model.predict([img_array[idx_1_same, ...], img_array[same_code_idx_rd, ...]])
                    prob_diff = prob_diff.reshape(-1, )
                    prob_same = prob_same.reshape(-1, )
                    # sort the similarity and get corresponding index
                    prob_diff = np.argsort(prob_diff)
                    prob_same = np.argsort(prob_same)

                    ### select "same" pairs with lowest similarity and "diff" pairs with highest similarity ###
                    pair_1_idx.extend(idx_1_same[prob_same[0:pair_num]].tolist())
                    pair_2_idx.extend(same_code_idx_rd[prob_same[0:pair_num]].tolist())
                    pair_1_idx.extend(idx_1_diff[prob_diff[-pair_num:]].tolist())
                    pair_2_idx.extend(diff_code_idx_rd[prob_diff[-pair_num:]].tolist())
                    flag.extend([1.0] * pair_num)
                    flag.extend([0.0] * pair_num)
                    # check pairs
                    if np.any(code_array[same_code_idx_rd] != code):
                        print("error: same_code_idx_rd")
                    if np.any(code_array[diff_code_idx_rd] == code):
                        print("error: diff_code_idx_rd")
                    if np.any(code_array[idx_1_same] != code):
                        print("error: idx_1_same")
                    if np.any(code_array[idx_1_diff] != code):
                        print("error: idx_1_diff")
            assert len(pair_1_idx) == len(pair_2_idx)
            assert len(pair_1_idx) == len(flag)

            return np.array(pair_1_idx), np.array(pair_2_idx), np.array(flag)

    def gen_pairs_valid(self, code_array_ref, code_array_valid, pair_num=4):

        """
        generate pairs data for testing

        :param code_array_ref: code array of reference images
        :param code_array_valid: code array of validation images
        :param pair_num: pair number
        :return: 3 numpy arrays, pair_1_idx, pair_2_idx, flag
        """

        pair_1_idx = []
        pair_2_idx = []
        flag = []

        ### generate pairs for each image ###
        for code in np.unique(code_array_valid):
            pair_2_idx_diff = np.where(code_array_ref != code)[0]
            pair_2_idx_same = np.where(code_array_ref == code)[0]
            pair_1_idx_diff = np.where(code_array_valid == code)[0]
            pair_1_idx_same = np.where(code_array_valid == code)[0]
            if len(pair_2_idx_diff) >= pair_num:
                pair_2_idx_diff = np.random.choice(pair_2_idx_diff, pair_num, replace=False).tolist()
            else:
                pair_2_idx_diff = np.random.choice(pair_2_idx_diff, pair_num, replace=True).tolist()
            if len(pair_2_idx_same) >= pair_num:
                pair_2_idx_same = np.random.choice(pair_2_idx_same, pair_num, replace=False).tolist()
            else:
                pair_2_idx_same = np.random.choice(pair_2_idx_same, pair_num, replace=True).tolist()
            if len(pair_1_idx_diff) >= pair_num:
                pair_1_idx_diff = np.random.choice(pair_1_idx_diff, pair_num, replace=False).tolist()
            else:
                pair_1_idx_diff = np.random.choice(pair_1_idx_diff, pair_num, replace=True).tolist()
            if len(pair_1_idx_same) >= pair_num:
                pair_1_idx_same = np.random.choice(pair_1_idx_same, pair_num, replace=False).tolist()
            else:
                pair_1_idx_same = np.random.choice(pair_1_idx_same, pair_num, replace=True).tolist()
            pair_1_idx.extend(pair_1_idx_same)
            pair_2_idx.extend(pair_2_idx_same)
            pair_1_idx.extend(pair_1_idx_diff)
            pair_2_idx.extend(pair_2_idx_diff)
            flag.extend([1.0] * pair_num)
            flag.extend([0.0] * pair_num)
            if np.any(code_array_ref[pair_2_idx_same] != code):
                print("error: pair_2_idx_same")
            if np.any(code_array_ref[pair_2_idx_diff] == code):
                print("error: pair_2_idx_diff")
            if np.any(code_array_valid[pair_1_idx_same] != code):
                print("error: pair_1_idx_same")
            if np.any(code_array_valid[pair_1_idx_diff] != code):
                print("error: pair_1_idx_diff")
        assert len(pair_1_idx) == len(pair_2_idx)
        assert len(pair_1_idx) == len(flag)

        return np.array(pair_1_idx), np.array(pair_2_idx), np.array(flag)


class ImageManager():

    """
    a class for image manipulation
    """

    def __init__(self):
        pass

    def img_modify(self, input_dir, output_dir, img_size, boxes=None):

        """
        parallel image modification

        :param input_dir: input folder
        :param output_dir: output folder
        :param img_size: new image size
        :return: None
        """

        ### get all image paths in a folder ###
        img_files = list(map(lambda x: os.path.join(input_dir, x), os.listdir(input_dir)))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        ### modify image parallely ###
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for i, img_file in enumerate(img_files):
                if boxes is not None:
                    box = boxes.loc[boxes["Image"] == os.path.basename(img_file), ["x0", "y0", "x1", "y1"]].values.reshape(-1,)
                    pool.apply(self._img_modify_single, args=(i, img_file, output_dir, img_size, box,))
                else:
                    pool.apply(self._img_modify_single, args=(i, img_file, output_dir, img_size,))

    def _img_modify_single(self, i, img_file, output_dir, img_size, box=None):

        """
        single image modification

        :param img_file: image file path
        :param output_dir: output folder
        :param img_size: new image size
        :return: None
        """

        print(i)
        img = io.imread(img_file)

        ### rgb to grey ###
        if img.ndim == 3:
            img = color.rgb2grey(img)
        if box is not None:
            x0 = box[0]
            y0 = box[1]
            x1 = box[2]
            y1 = box[3]
            img = img[y0:(y1+1), x0:(x1+1), ...]

        ### resize ###
        img = transform.resize(img, [img_size, img_size])
        img = np.stack((img,)*3, -1)

        ### save ###
        io.imsave(os.path.join(output_dir, os.path.basename(img_file)), img)

    def img_aug(self, label_df, label_to_num, input_dir, output_dir):

        """
        parallel image augmentation

        :param label_df: label data frame
        :param label_to_num: a dictionary of number of samples needed for each label
        :param input_dir: input folder
        :param output_dir: output folder
        :return: a data frame with columns "Image", "Id"
        """

        img_aug_res = []
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        for label, num in label_to_num.items():
            with mp.Pool(processes=mp.cpu_count()) as pool:
                img_aug_res.extend(pool.apply_async(self._img_aug_single, args=(label, num, label_df, input_dir, output_dir,)).get())
        file_name, label = zip(*img_aug_res)
        res = pd.DataFrame({"Image": file_name, "Id": label})
        return res

    def _img_aug_single(self, label, num, label_df, input_dir, output_dir):

        """
        image augmentation for a single label

        :param label: label need to be augmented
        :param num: extra number of samples needed
        :param label_df: label data frame
        :param input_dir: input folder
        :param output_dir: output folder
        :return: a list of tuples which contain image name, label
        """

        res = []
        ### get all images for a label ###
        img_names = label_df.loc[label_df["Id"] == label, "Image"].values

        ### randomly select images to be augmented ###
        img_names_aug = np.random.choice(img_names, num, replace=True).tolist()
        img_files_aug = [os.path.join(input_dir, img_name) for img_name in img_names_aug]
        id = 1

        ### do random augmentation for each image ###
        for img_file in img_files_aug:
            img = io.imread(img_file)
            # choose a type of augmentation
            k = np.random.choice([1, 2, 3], 1)
            # rotate
            if k == 1:
                degree = np.random.uniform(-90, 90)
                img_out = transform.rotate(img, degree)
            # random noise
            elif k == 2:
                img_out = util.random_noise(img)
            # both
            elif k == 3:
                degree = np.random.uniform(-90, 90)
                img_out = transform.rotate(img, degree)
                img_out = util.random_noise(img_out)
            # save image
            img_name = (os.path.basename(img_file)).split(".")[0]
            io.imsave(os.path.join(output_dir, img_name+"_aug_"+str(id)+".jpg"), img_out)
            print(img_name+"_aug_"+str(id))
            res.append((img_name+"_aug_"+str(id)+".jpg", label))
            id += 1

        return res