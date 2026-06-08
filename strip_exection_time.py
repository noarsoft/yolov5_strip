import json
import os

json_files1 = ["exec_time_exp0_algor1.json", "exec_time_exp1_algor1.json", "exec_time_exp2_algor1.json", "exec_time_exp3_algor1.json", "exec_time_exp4_algor1.json"]
json_files2 = ["exec_time_exp0_algor2.json", "exec_time_exp1_algor2.json", "exec_time_exp2_algor2.json", "exec_time_exp3_algor2.json", "exec_time_exp4_algor2.json"]
json_files3 = ["exec_time_exp0_algor3.json", "exec_time_exp1_algor3.json", "exec_time_exp2_algor3.json", "exec_time_exp3_algor3.json", "exec_time_exp4_algor3.json"]

execution_times = []


json_filesx = []
json_filesx.append(json_files1)
json_filesx.append(json_files2)
json_filesx.append(json_files3)

for json_files in json_filesx:
    for file in json_files:
        file1 = os.path.join(f"C:\\yolov5_strip\\backup_iris_train_and_kfold_pca2\\exec_times", file)
        # file1 = os.path.join(f"C:\\yolov5_strip\\backup_iris_train_and_kfold2\\exec_times", file)
        # file1 = os.path.join(f"C:\\yolov5_strip\\backup_wine_train_and_kfold_pca2\\exec_times", file)
        # file1 = os.path.join(f"C:\\yolov5_strip\\backup_wine_train_and_kfold2\\exec_times", file)
        if os.path.exists(file1):
            with open(file1, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    execution_times.extend(data)
        else:
            print(f"File {file1} does not exist.")

    # Calculate the average execution time
    if execution_times:
        average_time = sum(execution_times) / len(execution_times)
        print(f"Average Execution Time: {average_time:.4f}")
    else:
        print("No execution times found.")