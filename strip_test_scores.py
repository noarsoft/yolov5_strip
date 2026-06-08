import shutil

import lib
import os
from pathlib import Path
from PIL import Image

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import json
import time


#--print example for runing--
def print_example_run_detection():
    # yolo exp in runs folder
    output_directory = os.path.dirname(os.path.abspath(__file__))
    _exp_weight_pts_map = ["exp", "exp2"]  # k fold 0,1,... respectively training
    _exp_kfold_map = [0, 1]  # k fold 0,1,... respectively as _exp_weight_pts_map training

    _output_k_fold_path = os.path.join(output_directory, "output_k_fold")

    for i in range(len(_exp_weight_pts_map)):
        file_weight = os.path.join(output_directory, "runs", "train", _exp_weight_pts_map[i], "weights", "best.pt")
        detect_strip_py = os.path.join(output_directory, "detect_strip.py")
        detect_strip_bat_str = "python " + detect_strip_py + " --weights " + file_weight

        # with open(os.path.join(output_directory, 'output_k_fold', str(_exp_kfold_map[i]), 'detect.bat'), 'w') as file:
        #     file.write(detect_strip_bat_str)
        print("object detection k-fold[" + str(_exp_kfold_map[i]) + "] (cmd) > " + detect_strip_bat_str)

if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description="Select mode train or test")
    # # Add an argument
    # parser.add_argument('mode', type=str, choices=['preprocess', 'train', 'test'], help='Mode to select: train or test')
    # # Parse the arguments
    # args = parser.parse_args()

    #project_directory

    #----select test kfold---

    # (select algor 1, 2, 3) ==> (output_k_fold (algor1), output_k_fold2 (algor2), output_k_fold2 (algor3))


    # need run yolo already trainining model fold0 ex. python D:\yolov5_strip\detect_strip.py --weights D:\yolov5_strip\backup_wine_train_and_kfold_pca2\runs\train\exp0_algor1\weights\best.pt
    # example of using
    # _output_k_fold = "output_k_fold"  # ***(need to set) k fold data folder location
    # _exp_kfold_no = 0  # ***(need to set) k-fold folder number
    # need run yolo already trainining model fold1 ex. python D:\yolov5_strip\detect_strip.py --weights D:\yolov5_strip\backup_wine_train_and_kfold_pca2\runs\train\exp1_algor1\weights\best.pt
    # _output_k_fold = "output_k_fold"  # ***(need to set) k fold data folder location
    # _exp_kfold_no = 1  # ***(need to set) k-fold folder number
    # need run yolo already trainining model fold2 ex. python D:\yolov5_strip\detect_strip.py --weights D:\yolov5_strip\backup_wine_train_and_kfold_pca2\runs\train\exp2_algor1\weights\best.pt
    # _output_k_fold = "output_k_fold"  # ***(need to set) k fold data folder location
    # _exp_kfold_no = 1  # ***(need to set) k-fold folder number
    # ...

    #output_k_fold (algor1), output_k_fold2 (algor2), output_k_fold3 (algor3)
    # _output_k_fold = os.path.join("backup_wine_train_and_kfold_pca2","output_k_fold3")  # ***(need to set) k fold data folder location



    # _folder_of_training_weight = "backup_iris_train_and_kfold2"  # ***(need to set) k fold data folder location
    # _folder_of_training_weight = "backup_iris_train_and_kfold_pca2"  # ***(need to set) k fold data folder location
    # _folder_of_training_weight = "backup_wine_train_and_kfold2" # ***(need to set) k fold data folder location
    # _folder_of_training_weight = "backup_wine_train_and_kfold_pca2"  # ***(need to set) k fold data folder location
    # _folder_of_training_weight = "backup_student_train_and_kfold2"  # ***(need to set) k fold data folder location
    # _folder_of_training_weight = "backup_student_train_and_kfold_pca2"  # ***(need to set) k fold data folder location
    # _folder_of_training_weight = "backup_beast_cancer_train_and_kfold2"  # ***(need to set) k fold data folder location
    _folder_of_training_weight = "backup_beast_cancer_train_and_kfold_pca2"  # ***(need to set) k fold data folder location

    _exp_kfold_no = 4 # ***(need to set) fold
    _algorV = '3'  # ***(need to set) algor subfix (output_k_fold, output_k_fold2, output_k_fold3)





    if _algorV == '1':
        _algorV = ''
    _output_k_fold = os.path.join(_folder_of_training_weight, "output_k_fold" + _algorV)
    _output_directory = os.path.dirname(os.path.abspath(__file__))

    _output_k_fold_path = os.path.join(_output_directory, _output_k_fold)
    _input_img_detection_path = os.path.join(_output_directory, "detect_input")
    _output_img_detection_path = os.path.join(_output_directory, "detect_out_crop_label")
    lib.clear_folder(_output_img_detection_path)

    #read test folder
    test_img_path = os.path.join(_output_k_fold_path, str(_exp_kfold_no), "test", "images")
    test_label_path = os.path.join(_output_k_fold_path, str(_exp_kfold_no), "test", "labels")

    img_file_tests = lib.list_files_in_folder(test_img_path)
    label_file_tests = lib.list_files_in_folder(test_label_path)

    #feed check validate
    y_pred = []
    y_true = []

    print("====== need to run the detect_strip.py with the training k-fold model ======")
    print_example_run_detection()

    for file_img in img_file_tests:
        file_image_name = lib.get_file_name_without_extension(file_img)
        label_file = os.path.join(test_label_path, file_image_name + ".txt")
        with open(label_file, 'r') as file:
            label_and_bb = file.read()
        y_true.append( int(label_and_bb.split(' ')[0]) )

        source_file = os.path.join(_output_directory, _output_k_fold, str(_exp_kfold_no), 'test','images', file_img)
        print("copy to detection: ", source_file, " > ", os.path.join(_input_img_detection_path, file_img) )
        shutil.copy(source_file, os.path.join(_input_img_detection_path, file_img) )

        while True:
            try:
                if lib.is_folder_empty(_input_img_detection_path):
                    items = os.listdir(_output_img_detection_path)
                    # Filter and list folders
                    list_file_labels = lib.list_files_in_folder( os.path.join(_output_img_detection_path, items[0]) )

                    label_vote_labels = []
                    label_vote_confs = []
                    for f1 in list_file_labels:
                        with open(os.path.join(_output_img_detection_path, items[0], f1), 'r') as file:
                            json_label = json.load(file)
                            label_vote_labels.append(float(json_label["label_id"]))
                            label_vote_confs.append(float(json_label["conf"]))

                    if len(label_vote_confs) > 0:
                        #best label
                        # best_label = lib.find_majority_element(label_votes)
                        conf_value = max(label_vote_confs)
                        conf_index = label_vote_confs.index(conf_value)
                        best_label = label_vote_labels[conf_index]
                        print("detect: ", file_img, "class: ", best_label, ", max confident: ", str(max(label_vote_confs)))
                        # print("detect: ", file_img, " label: ",label_vote_confs, " marjority vote: ",best_label)
                        y_pred.append(best_label)
                    else:
                        print("detect: ", file_img, "*** label: not detected ***")
                        y_pred.append(-1) #no result from yolo
                        print("*** cant detected: ", _input_img_detection_path)

                    print("clear folder")
                    lib.clear_folder(_output_img_detection_path)

                    break
            except:
                x = 0

    lib.evaluate_model(y_true, y_pred)





