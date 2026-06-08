import shutil
import os
from PIL import Image

import subprocess

import json

if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description="Select mode train or test")
    # # Add an argument
    # parser.add_argument('mode', type=str, choices=['preprocess', 'train', 'test'], help='Mode to select: train or test')
    # # Parse the arguments
    # args = parser.parse_args()

    #project_directory
    _output_directory = os.path.dirname(os.path.abspath(__file__))

    #-- algor1 --
    # yolo exp in runs folder
    _exp_weight_pts_map = ["exp", "exp2"]
    _exp_kfold_map = [0, 1]
    # _exp_weight_pts_map = ["exp0_algor1", "exp1_algor1", "exp2_algor1", "exp3_algor1", "exp4_algor1",
    #                        "exp0_algor2", "exp1_algor2", "exp2_algor2", "exp3_algor2",
    #                        "exp4_algor2"]  # k fold 0,1,2,3,4 respectively training
    # _exp_kfold_map = [0, 1, 2, 3, 4,
    #                   0, 1, 2, 3, 4]  # k fold 0,1,2,3,4 respectively as _exp_weight_pts_map training
    _output_k_fold_path = os.path.join(_output_directory, "output_k_fold")

    for i in range(len(_exp_weight_pts_map)):
        file_weight = os.path.join(_output_directory, "runs", "train", _exp_weight_pts_map[i], "weights", "best.pt")
        detect_strip_py = os.path.join(_output_directory, "detect_strip.py")
        detect_strip_bat_str = "python " + detect_strip_py + " --weights " + file_weight

        with open(os.path.join(_output_directory, 'output_k_fold', str(_exp_kfold_map[i]), 'detect.bat'), 'w') as file:
            file.write(detect_strip_bat_str)
        print("object detection k-fold["+str(_exp_kfold_map[i])+"] (cmd) > " + detect_strip_bat_str)

    #we can run above command for each. after that, go to the strip_test_scores.py
    print("we can run above command for each command (only 1 cross validation check for each time).\n"
          "after that, go to the strip_test_scores.py.\n"
          "change _exp_kfold_no = 0  # test for k-fold number 0 and see scores")
