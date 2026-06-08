import os
import time
import warnings
import statistics

import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import lib

warnings.filterwarnings("ignore", message="X does not have valid feature names")



# project_directory
_output_directory = os.path.dirname(os.path.abspath(__file__))

result, label_name = lib.load_k_float_encoder_student_performance(_output_directory)
# result, label_name = lib.load_k_float_encoder_nutrition(_output_directory)
# result, label_name = lib.load_k_float_encoder_breast_cancer(_output_directory)
# result, label_name = lib.load_k_float_encoder_wine(_output_directory)
# result, label_name = lib.load_k_float_encoder_adult(_output_directory)
# result, label_name = lib.load_k_float_encoder_electric(_output_directory)


# =========================
# Store results
# =========================
acc_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
train_times = []


# =========================
# Run each fold
# =========================
for fold_idx, paths in enumerate(result):
    print("=" * 60)
    print(f"Fold {fold_idx}")
    print("=" * 60)

    train_csv = paths[0]
    val_csv = paths[1]
    test_csv = paths[2]

    # -------------------------
    # Read CSV
    # -------------------------
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # -------------------------
    # Split X, y
    # -------------------------
    # label_name คือชื่อ column label จาก lib
    X_train = train_df.drop(columns=[label_name])
    y_train = train_df[label_name]

    X_val = val_df.drop(columns=[label_name])
    y_val = val_df[label_name]

    X_test = test_df.drop(columns=[label_name])
    y_test = test_df[label_name]

    # -------------------------
    # Encode string labels to number
    # เช่น Setosa -> 0
    # -------------------------
    le = LabelEncoder()

    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    print("Classes:", list(le.classes_))

    # -------------------------
    # Create XGBoost model
    # -------------------------

    num_classes = len(le.classes_)

    if num_classes == 2:
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42
        )
    else:
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            random_state=42
        )

    # -------------------------
    # Train
    # -------------------------
    start_time = time.time()

    model.fit(X_train, y_train_enc)

    end_time = time.time()
    train_time = end_time - start_time
    train_times.append(train_time)

    # -------------------------
    # Predict
    # -------------------------
    y_pred_enc = model.predict(X_test)

    # ถ้าอยากได้ชื่อ class กลับมา
    y_pred = le.inverse_transform(y_pred_enc)

    # -------------------------
    # Metrics
    # -------------------------
    acc = accuracy_score(y_test_enc, y_pred_enc)
    precision = precision_score(y_test_enc, y_pred_enc, average="macro", zero_division=0)
    recall = recall_score(y_test_enc, y_pred_enc, average="macro", zero_division=0)
    f1 = f1_score(y_test_enc, y_pred_enc, average="macro", zero_division=0)

    acc_scores.append(acc)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    print(f"Train time: {train_time:.4f} sec")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test_enc,
        y_pred_enc,
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test_enc, y_pred_enc))


# =========================
# Summary
# =========================
print("\n" + "=" * 60)
print("5-Fold Summary")
print("=" * 60)

print(f"Accuracy : {statistics.mean(acc_scores):.4f} ± {statistics.stdev(acc_scores):.4f}")
print(f"Precision: {statistics.mean(precision_scores):.4f} ± {statistics.stdev(precision_scores):.4f}")
print(f"Recall   : {statistics.mean(recall_scores):.4f} ± {statistics.stdev(recall_scores):.4f}")
print(f"F1-score : {statistics.mean(f1_scores):.4f} ± {statistics.stdev(f1_scores):.4f}")
print(f"Train time avg: {statistics.mean(train_times):.4f} sec")