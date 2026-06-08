import shutil

import numpy as np

import lib
import os
from PIL import Image

import subprocess

import json

# prequisition
# u need to setup yolov5 and can runable, firstly


# --THIS FILE is preprocessing and creating strip encode from k-fold--
if __name__ == "__main__":

    _output_directory = os.path.dirname(os.path.abspath(__file__))
    _output_label_list = os.path.join(_output_directory, "output_strip_label_map.txt")
    _output_kfold_data = os.path.join(_output_directory, "output_kfold_data.json")

    # algor1
    _output_strip_encode_v1 = os.path.join(_output_directory, "out_strip")
    _output_strip_encode_label_v1 = os.path.join(_output_directory, "out_strip_label")
    _output_k_fold_v1 = os.path.join(_output_directory, "output_k_fold")

    # algor2
    _output_strip_encode_v2 = os.path.join(_output_directory, "out_strip2")
    _output_strip_encode_label_v2 = os.path.join(_output_directory, "out_strip_label2")
    _output_k_fold_v2 = os.path.join(_output_directory, "output_k_fold2")

    # algor3
    _output_strip_encode_v3 = os.path.join(_output_directory, "out_strip3")
    _output_strip_encode_label_v3 = os.path.join(_output_directory, "out_strip_label3")
    _output_k_fold_v3 = os.path.join(_output_directory, "output_k_fold3")

    folders = [_output_strip_encode_v1, _output_strip_encode_label_v1, _output_k_fold_v1,
               _output_strip_encode_v2, _output_strip_encode_label_v2, _output_k_fold_v2,
               _output_strip_encode_v3, _output_strip_encode_label_v3, _output_k_fold_v3]

    lib.remove_and_create_folder(folders)

    # ======== step:0 select dataset ========



    #https://www.kaggle.com/datasets/sanadalali/food-101-nutritional-information
    # _path_csv, _header_name, _label_header, _pca_csv, header_name_pca = lib.load_header_nutrition(_output_directory)
    #https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset
    _path_csv, _header_name, _label_header, _pca_csv, header_name_pca = lib.load_header_student_performance(_output_directory)

    # ****--load yeast-- https://www.kaggle.com/datasets/samanemami/yeastcsv 10 clasees (high imbalance) 1000 items
    # _path_csv, _header_name, _label_header, _pca_csv, header_name_pca = lib.load_header_yeast(_output_directory)

    # _path_csv, _header_name, _label_header, _pca_csv, header_name_pca = lib.load_header_breast_cancer(_output_directory)

    # ****--load iris-- https://archive.ics.uci.edu/
    # _path_csv, _header_name, _label_header, _pca_csv, header_name_pca = lib.load_header_iris(_output_directory)

    # ****--load wine 3 class--
    # _path_csv, _header_name, _label_header, _pca_csv, header_name_pca = lib.load_wine(_output_directory)

    # ======== step:0.1 OPTIONAL PCA (disable by comment this 2 lines) ========
    # _path_csv = _pca_csv
    # _header_name = header_name_pca
    # print(os.path.join(_output_directory, "PCA.jpg"))
    # lib.data_scatterplot2column(_path_csv, "PC1", "PC2", os.path.join(_output_directory, "PCA.jpg")).=j


    # ======== step 1: create k-fold and explore min and max from features =========
    print("======== step 1: create k-fold and explore min and max from features =========")

    if os.path.exists(_output_kfold_data):
        #load crossvalidation
        print(
            "NOTE: K-fold file existing " + _output_kfold_data + ". Load K-fold file. if want to renew data, delete the file.")
        print(
            "This might error, if data in "+_output_kfold_data+ " is not macth with the image in strip folder this need be renewed data. (delete the file)")
    else:
        #create cross validation
        print("CREATE new CROSS VALIDATION")
        (result_k_fold, train_index_k_fold, validate_index_k_fold, test_feature_k_fold,
            min_features_k_fold, max_features_k_fold,
         indices_test, indices_train_validate) = lib.preprocess_create_k_fold(_path_csv,
            _header_name,_label_header)

        #save
        data_serializable = lib.convert_np_types([result_k_fold, train_index_k_fold, validate_index_k_fold, test_feature_k_fold,
            min_features_k_fold, max_features_k_fold, indices_test, indices_train_validate])
        with open(_output_kfold_data, 'w') as f:
            json.dump(data_serializable, f)

    # load crossvalidation
    with open(_output_kfold_data, 'r') as f:
        [result_k_fold, train_index_k_fold, validate_index_k_fold, test_feature_k_fold,
         min_features_k_fold, max_features_k_fold, indices_test, indices_train_validate] = json.load(f)

    (result_k_fold, train_index_k_fold, validate_index_k_fold, test_feature_k_fold,
     min_features_k_fold, max_features_k_fold,
     indices_test, indices_train_validate) = lib.preprocess_create_k_fold(_path_csv, _header_name,
                                                                                _label_header,
                                                                                indices_test, indices_train_validate)

    # ======== step 2: create strip encoder from the K-fold ========
    print("======== step 2: create strip encoder from the K-fold ========")
    for fold_num in range(len(train_index_k_fold)):
        output_strip = os.path.join(_output_strip_encode_v1, str(fold_num))
        output_strip_label = os.path.join(_output_strip_encode_label_v1, str(fold_num))

        output_strip2 = os.path.join(_output_strip_encode_v2, str(fold_num))
        output_strip_label2 = os.path.join(_output_strip_encode_label_v2, str(fold_num))

        output_strip3 = os.path.join(_output_strip_encode_v3, str(fold_num))
        output_strip_label3 = os.path.join(_output_strip_encode_label_v3, str(fold_num))

        Ls = min_features_k_fold[fold_num]
        Hs = max_features_k_fold[fold_num]
        Ls, Hs = lib.max_min_range_expand(Ls, Hs)
        print('---create strip encoder algor1 (K-fold=' + str(fold_num) + ')---')
        lib.strips_encode(_path_csv, _header_name, _label_header, _output_label_list,
                          Ls, Hs, output_strip, output_strip_label)

        print('---create strip encoder algor2 (K-fold=' + str(fold_num) + ')---')
        lib.strips_encode2(_path_csv, _header_name, _label_header, _output_label_list,
                           Ls, Hs, output_strip2, output_strip_label2)

        print('---create strip encoder algor3 (K-fold=' + str(fold_num) + ')---')
        lib.strips_encode3(_path_csv, _header_name, _label_header, _output_label_list,
                           Ls, Hs, output_strip3, output_strip_label3)

    # ======== step 3: preprocess_training_strategy ======== (cross validation)
    print("======== step 3: preprocess_training_strategy ======== (cross validation)")
    # algor1
    yamls, test_folders_img, test_folders_label, result_k_fold = \
        lib.preprocess_training_strategy(_path_csv, _output_directory, _output_strip_encode_v1,
                                         _output_strip_encode_label_v1, _output_label_list,
                                         _output_k_fold_v1, result_k_fold)

    # algor2
    yamls, test_folders_img, test_folders_label, result_k_fold = \
        lib.preprocess_training_strategy(_path_csv, _output_directory, _output_strip_encode_v2,
                                         _output_strip_encode_label_v2, _output_label_list,
                                         _output_k_fold_v2, result_k_fold)

    # algor3
    yamls, test_folders_img, test_folders_label, result_k_fold = \
        lib.preprocess_training_strategy(_path_csv, _output_directory, _output_strip_encode_v3,
                                         _output_strip_encode_label_v3, _output_label_list,
                                         _output_k_fold_v3, result_k_fold)

    print("--finished preprocessing step. next go train model YOLOv5 using. (see folder output_k_fold)--")
    print(f"======== step 4: training model and cmd. note: run training respectively ======== (see at output)\n"
          f"If YOLOv5 installed and it can be run, everything should be fined for training strip encdoer.\n\n"
          f"ex. to train model run the train.py at: (windows) " + os.path.join(_output_k_fold_v1, "train.bat") + "\n"
                                                                                                                 f"Linux might be adapted the command \n\n"

                                                                                                                 f"--step 5 (strip_test_scores.py): go to file strip_test_score.py-- prediction and see accuracy\n")

    # ======== step 4: training model and cmd. note: run training respectively ======== (see at output)
    # ex. to train model run the train.py at: C:\yolov5_strip\output_k_fold\0\train.bat

    # ======== step 5: (strip_test_scores.py): go to file strip_test_score.py ======== prediction and see accuracy





# ======== step 3: preprocess_training_strategy ======== (cross validation)
# to train model run (CMD) >
# python C:\yolov5_strip\train.py --data C:\yolov5_strip\output_k_fold\0\dataset.yaml --weights C:\yolov5_strip\yolov5s.pt --img 640 --batch-size 2 --epochs 1000
# OR to train model run the train.py at: C:\yolov5_strip\output_k_fold\0\train.bat
# to train model run (CMD) >
# python C:\yolov5_strip\train.py --data C:\yolov5_strip\output_k_fold\1\dataset.yaml --weights C:\yolov5_strip\yolov5s.pt --img 640 --batch-size 2 --epochs 1000
# OR to train model run the train.py at: C:\yolov5_strip\output_k_fold\1\train.bat
# to train model run (CMD) >
# python C:\yolov5_strip\train.py --data C:\yolov5_strip\output_k_fold\2\dataset.yaml --weights C:\yolov5_strip\yolov5s.pt --img 640 --batch-size 2 --epochs 1000
# OR to train model run the train.py at: C:\yolov5_strip\output_k_fold\2\train.bat
# to train model run (CMD) >
# python C:\yolov5_strip\train.py --data C:\yolov5_strip\output_k_fold\3\dataset.yaml --weights C:\yolov5_strip\yolov5s.pt --img 640 --batch-size 2 --epochs 1000
# OR to train model run the train.py at: C:\yolov5_strip\output_k_fold\3\train.bat
# to train model run (CMD) >
# python C:\yolov5_strip\train.py --data C:\yolov5_strip\output_k_fold\4\dataset.yaml --weights C:\yolov5_strip\yolov5s.pt --img 640 --batch-size 2 --epochs 1000
# OR to train model run the train.py at: C:\yolov5_strip\output_k_fold\4\train.bat
# to train model run (CMD) >
# python C:\yolov5_strip\train.py --data C:\yolov5_strip\output_k_fold2\0\dataset.yaml --weights C:\yolov5_strip\yolov5s.pt --img 640 --batch-size 2 --epochs 1000
# OR to train model run the train.py at: C:\yolov5_strip\output_k_fold2\0\train.bat
# to train model run (CMD) >
# python C:\yolov5_strip\train.py --data C:\yolov5_strip\output_k_fold2\1\dataset.yaml --weights C:\yolov5_strip\yolov5s.pt --img 640 --batch-size 2 --epochs 1000
# OR to train model run the train.py at: C:\yolov5_strip\output_k_fold2\1\train.bat
# to train model run (CMD) >
# python C:\yolov5_strip\train.py --data C:\yolov5_strip\output_k_fold2\2\dataset.yaml --weights C:\yolov5_strip\yolov5s.pt --img 640 --batch-size 2 --epochs 1000
# OR to train model run the train.py at: C:\yolov5_strip\output_k_fold2\2\train.bat
# to train model run (CMD) >
# python C:\yolov5_strip\train.py --data C:\yolov5_strip\output_k_fold2\3\dataset.yaml --weights C:\yolov5_strip\yolov5s.pt --img 640 --batch-size 2 --epochs 1000
# OR to train model run the train.py at: C:\yolov5_strip\output_k_fold2\3\train.bat
# to train model run (CMD) >
# python C:\yolov5_strip\train.py --data C:\yolov5_strip\output_k_fold2\4\dataset.yaml --weights C:\yolov5_strip\yolov5s.pt --img 640 --batch-size 2 --epochs 1000
# OR to train model run the train.py at: C:\yolov5_strip\output_k_fold2\4\train.bat
# to train model run (CMD) >
# python C:\yolov5_strip\train.py --data C:\yolov5_strip\output_k_fold3\0\dataset.yaml --weights C:\yolov5_strip\yolov5s.pt --img 640 --batch-size 2 --epochs 1000
# OR to train model run the train.py at: C:\yolov5_strip\output_k_fold3\0\train.bat
# to train model run (CMD) >
# python C:\yolov5_strip\train.py --data C:\yolov5_strip\output_k_fold3\1\dataset.yaml --weights C:\yolov5_strip\yolov5s.pt --img 640 --batch-size 2 --epochs 1000
# OR to train model run the train.py at: C:\yolov5_strip\output_k_fold3\1\train.bat
# to train model run (CMD) >
# python C:\yolov5_strip\train.py --data C:\yolov5_strip\output_k_fold3\2\dataset.yaml --weights C:\yolov5_strip\yolov5s.pt --img 640 --batch-size 2 --epochs 1000
# OR to train model run the train.py at: C:\yolov5_strip\output_k_fold3\2\train.bat
# to train model run (CMD) >
# python C:\yolov5_strip\train.py --data C:\yolov5_strip\output_k_fold3\3\dataset.yaml --weights C:\yolov5_strip\yolov5s.pt --img 640 --batch-size 2 --epochs 1000
# OR to train model run the train.py at: C:\yolov5_strip\output_k_fold3\3\train.bat
# to train model run (CMD) >
# python C:\yolov5_strip\train.py --data C:\yolov5_strip\output_k_fold3\4\dataset.yaml --weights C:\yolov5_strip\yolov5s.pt --img 640 --batch-size 2 --epochs 1000
# OR to train model run the train.py at: C:\yolov5_strip\output_k_fold3\4\train.bat
# --finished preprocessing step. next go train model YOLOv5 using. (see folder output_k_fold)--
# ======== step 4: training model and cmd. note: run training respectively ======== (see at output)
# If YOLOv5 installed and it can be run, everything should be fined for training strip encdoer.
#
# ex. to train model run the train.py at: (windows) C:\yolov5_strip\output_k_fold\train.bat
# Linux might be adapted the command
#
# --step 5 (strip_test_scores.py): go to file strip_test_score.py-- prediction and see accuracy
#
#
# Process finished with exit code 0
