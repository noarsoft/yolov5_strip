import matplotlib.pyplot as plt
import numpy as np

x = [{
    "Algor1.+PCA iris": 0.0504,
    "Algor2.+PCA iris": 0.0498,
    "Algor3.+PCA iris": 0.0503,
    "Algor1. iris": 0.0513,
    "Algor2. iris": 0.0500,
    "Algor3. iris": 0.0501,
    "knn (k=3) iris": 0.00039997895558675133,
    "knn (k=7) iris": 0.0004667202631632487,
    "knn (k=10) iris": 0.0005002737045288086,
    "SVM iris": 0.0001666704813639323,
    "Decision Tree iris": 0.00010002454121907553,
}, {
    "Algor1. wine": 0.0468,
    "Algor2. wine": 0.0467,
    "Algor3. wine": 0.0462,
    "Algor1.+PCA wine": 0.0481,
    "Algor2.+PCA wine": 0.0478,
    "Algor3.+PCA wine": 0.0473,
    "knn (k=3) wine": 0.0004725592476981027,
    "knn (k=7) wine": 0.00042858123779296873,
    "knn (k=10) wine": 0.00042704854692731585,
    "SVM wine": 0.00017136165073939732,
    "Decision Tree wine": 0.000008575575692313059,
}]

data_iris = x[0]
data_wine = x[1]

labels_iris = list(data_iris.keys())
values_iris = list(data_iris.values())

labels_wine = list(data_wine.keys())
values_wine = list(data_wine.values())


x_iris = np.arange(len(labels_iris))
x_wine = np.arange(len(labels_wine))

plt.figure(figsize=(14, 10))

plt.subplot(2, 1, 1)
plt.bar(x_iris, values_iris, color='skyblue')
plt.title("Average Execution Times of Test Set: Iris Dataset")
plt.ylabel("Execution Time (seconds)")
plt.xticks(x_iris, labels_iris, rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.subplot(2, 1, 2)
plt.bar(x_wine, values_wine, color='lightcoral')
plt.title("Average Execution Times of Test Set: Wine Dataset")
plt.ylabel("Execution Time (seconds)")
plt.xticks(x_wine, labels_wine, rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
