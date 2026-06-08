import os
import statistics
import time

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import lib
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# project_directory
_output_directory = os.path.dirname(os.path.abspath(__file__))
result, label_name = lib.load_k_float_encoder_student_performance(_output_directory)
# result, label_name = lib.load_k_float_encoder_iris(_output_directory)
# result, label_name = lib.load_k_float_encoder_wine(_output_directory)

print(result) # 5 listkfold inside 3 folders (train validate test)



_times = []


accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

for result_each in result:
    train_out = result_each[0]
    val_out = result_each[1]
    test_out = result_each[2]

    train_data = pd.read_csv(train_out)
    val_data = pd.read_csv(val_out)
    test_data = pd.read_csv(test_out)

    X_train = train_data.drop(columns=[label_name])
    y_train = train_data[label_name]

    X_val = val_data.drop(columns=[label_name])
    y_val = val_data[label_name]

    X_test = test_data.drop(columns=[label_name])
    y_test = test_data[label_name]


    # 3 points (A,A,B) pred A
    model = KNeighborsClassifier(n_neighbors=10) # nearest 3 points

    model.fit(X_train, y_train)


    # df = pd.DataFrame({
    #     'actual': [1, 0, 1, 1, 0],
    #     'predicted': [1, 0, 1, 0, 0]
    # })
    #
    # accuracy = accuracy_score(df['actual'], [1, 0, 1, 0, 0])
    # print("Accuracy:", accuracy)


    # y_val_pred = model.predict(X_val)
    # print(f'Validation Accuracy: {accuracy_score(y_val, y_val_pred)}')

    execution_times = []
    for _, row in X_test.iterrows():
        input_data = row.values.reshape(1, -1)  # Reshape row into a 2D array (1 sample)

        start_time = time.time()  # Start timing
        model.predict(input_data)  # Predict for the single row
        end_time = time.time()  # End timing

        execution_times.append(end_time - start_time)



    y_test_pred = model.predict(X_test)


    accuracy, precision, recall, f1, cm = lib.evaluate_model(y_test, y_test_pred)
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

average_time = sum(execution_times) / len(execution_times)
print("average_time ",average_time)

mean_accuracy = sum(accuracy_list) / len(accuracy_list)
_,std_accuracy = lib.calculate_std(accuracy_list)

mean_precision = sum(precision_list) / len(precision_list)
_,std_precision = lib.calculate_std(precision_list)

mean_recall = sum(recall_list) / len(recall_list)
_,std_recall = lib.calculate_std(recall_list)

mean_f1 = sum(f1_list) / len(f1_list)
_,std_f1 = lib.calculate_std(f1_list)

print("Metrics Over All Folds:")
print(f"Mean Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}")
print(f"Mean Precision: {mean_precision:.2f} ± {std_precision:.2f}")
print(f"Mean Recall: {mean_recall:.2f} ± {std_recall:.2f}")
print(f"Mean F1-Score: {mean_f1:.2f} ± {std_f1:.2f}")




    # cm = confusion_matrix(y_test, y_test_pred)
    # print(f'Confusion Matrix:\n{cm}')
    #
    # # Plot Confusion Matrix
    # # plt.figure(figsize=(8, 6))
    # # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    # # plt.title('Confusion Matrix')
    # # plt.xlabel('Predicted')
    # # plt.ylabel('Actual')
    # # plt.show()
    #
    #
    # print(f'Test Accuracy: {accuracy_score(y_test, y_test_pred)}')
    #
    # #precision
    # #manage imbalance data by weighted data numbers of the class
    # precision = precision_score(y_test, y_test_pred, average='weighted')
    # print(f'Precision: {precision:.2f}')
    #
    # #recall
    # recall = recall_score(y_test, y_test_pred, average='weighted')
    # print(f'Recall: {recall:.2f}')
    #
    # #f-socre
    # f1 = f1_score(y_test, y_test_pred, average='weighted')
    # print(f'F1 Score: {f1:.2f}')