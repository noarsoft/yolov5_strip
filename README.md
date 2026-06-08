# NISE: Numerical-to-Image Strip Encoding for YOLO-based Classification

**NISE** is a numerical-to-image transformation approach for converting tabular numerical data into image representations that can be used with YOLO-based object detection models.

This repository contains the source code for the **NISEv1 / YOLOv5 Strip Encoding** experiment.

## Project Website

More information is available on the project website:

https://piyavach.com

## Source Code

The source code is available on GitHub:

https://github.com/noarsoft/yolov5_strip

## Download Weights and Training Files

The trained weights, datasets, and training-related files are available on Google Drive:

https://drive.google.com/drive/folders/1oE_P17nQUwuVlinQfUSzm9dftsE3Z5tE?usp=sharing

## Demo Video

YouTube demonstration:

https://youtu.be/zmwJagc-FJ0

## Published Paper

Paper available on Springer:

https://link.springer.com/chapter/10.1007/978-981-96-8889-0_40

If you find this work helpful, please consider citing our paper.

## Authors

* Piyavach K.
* Waranya M.

Proposed in **IEA/AIE 2025**.

## Overview

This work explores a transformation method that converts numerical data into image strips. The generated images are then used as input for YOLOv5 to perform feature-based classification.

The main purpose of this first version is to demonstrate the feasibility of transforming numerical datasets into image-based representations for YOLO-based learning.

Performance speed was not the primary focus in this version. The processing speed depends largely on the YOLO model and training configuration used.

## Supported Datasets

The experiment uses datasets from the UCI Machine Learning Repository, including:

* Iris
* Wine
* Student Performance
* Breast Cancer Wisconsin

## Main Files

Important files in this project include:

```text
strip_pre.py        Main preprocessing script for generating strip images
lib.py              Core helper functions for image generation and processing
train.py            YOLOv5 training script
val.py              YOLOv5 validation script
detect_strip.py     Detection / prediction script for strip images
requirements.txt    Python package list
```

## How to Run

Start by running:

```bash
python strip_pre.py
```

The script contains step-by-step comments for generating NISE image datasets.

The generated image folders can then be used for YOLOv5 training, validation, and testing.

## Generated Output Folders

The preprocessing script may generate folders such as:

```text
output_k_fold/
output_k_fold2/
output_k_fold3/
out_strip/
out_strip_label/
out_strip_label2/
out_strip_label3/
detect_out_crop/
```

These folders are generated outputs and are not included directly in this GitHub repository.

Large files such as trained weights, generated datasets, and training outputs are provided separately through Google Drive.

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is incomplete, the main libraries used include:

```text
opencv-python
numpy
pillow
scikit-learn
pandas
matplotlib
```

## Notes

This project was originally developed and tested on Windows.
With small path-related modifications, it should also be able to run on Linux.

Large files such as weights, generated images, datasets, and training outputs are intentionally excluded from GitHub to keep the repository lightweight.

Please download those files from Google Drive when needed.

## Future Work

Future versions of NISE aim to improve:

* Training speed
* Dataset scalability
* Model efficiency
* Support for larger datasets
* Alternative base models such as MobileNet
* Easier deployment and usage

NISEv2 is currently planned as the next version of this project.

## License

This repository is provided for research and educational purposes.

## Citation

If you use this project or find it useful, please cite our published paper:

```text
Piyavach K. and Waranya M.
Extending YOLO for Feature-Based Classification via Numerical-to-Image Transformation.
IEA/AIE 2025.
Springer.
```

## References

1. Africa, A.D., Velasco, J.: Development of a urine strip analyzer using artificial neural network using an android phone. ARPN Journal of Engineering and Applied Sciences 12, 1706–1713 (03 2017)

2. Bochkovskiy, A., Wang, C.Y., Liao, H.Y.M.: Yolov4: Optimal speed and accuracy of object detection. ArXiv abs/2004.10934 (2020), https://api.semanticscholar.org/CorpusID:216080778

3. Cortes, C., Vapnik, V.: Support-vector networks. Machine learning 20(3), 273–297 (1995)

4. Cortez, Paulo Cerdeira A. Almeida F., M.T., J., R.: Wine Quality. UCI Machine Learning Repository (2009)

5. Cover, T., Hart, P.: Nearest neighbor pattern classification. IEEE Transactions on Information Theory 13(1), 21–27 (1967)

6. El Ghmary, M., Ouassine, Y., Ouacha, A.: License Plate Character Recognition System Using YOLOv5, pp. 588–594 (03 2023)

7. Fisher, R.A.: Iris. UCI Machine Learning Repository (1936)

8. Jocher, G.: Ultralytics yolov5 (2020), https://github.com/ultralytics/yolov5

9. Kim, S.C., Cho, Y.S.: Predictive system implementation to improve the accuracy of urine self-diagnosis with smartphones: Application of a confusion matrix-based learning model through rgb semiquantitative analysis. Sensors 22, 5445 (07 2022)

10. Mishra, S., Sarkar, U., Taraphder, S., Datta, S., Swain, D., Saikhom, R., Panda, S., Laishram, M.: Principal component analysis. International Journal of Livestock Research p. 1 (01 2017)

11. Ng, W.K., Choi, S., Ravishankar, C.: Lossless and lossy data compression (06 1998)

12. Pei, C.: Research on the influencing factors of student performance. Theoretical and Natural Science 51, 26–33 (09 2024). https://doi.org/10.54254/2753-8818/51/2024CH0131

13. Ragab, M., Jadid Abdulkadir, S., Muneer, A., Alqushaibi, A., Sumiea, E., Qureshi, R., Al-Selwi, S., Alhussian, H.: A comprehensive systematic review of yolo for medical object detection (2018 to 2023). IEEE Access PP, 1–1 (01 2024)

14. Rajpurkar, P., Hannun, A.Y., Haghpanahi, M., Bourn, C., Ng, A.Y.: Cardiologist-level arrhythmia detection with convolutional neural networks (2017), https://arxiv.org/abs/1707.01836

15. Raza, A., Shaikh, M., Siddiqui, O., Ali, A., Khan, A.: Enhancing agricultural pest management with yolo v5: A detection and classification approach. UMT Artificial Intelligence Review 3 (12 2023)

16. Redmon, J., Divvala, S., Girshick, R., Farhadi, A.: You only look once: Unified, real-time object detection. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 779–788 (2016)

17. Redmon, J., Farhadi, A.: Yolov3: An incremental improvement. ArXiv abs/1804.02767 (2018), https://api.semanticscholar.org/CorpusID:4714433

18. Wolberg, William, M.O.S.N., Street, W.: Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository (1993), DOI: https://doi.org/10.24432/C5DW2B

19. Wu, X., Kumar, V., Quinlan, J.R., Ghosh, J., Yang, Q., Motoda, H., McLachlan, G.J., Ng, A., Liu, B., Philip, S.Y., et al.: Top 10 algorithms in data mining. Knowledge and information systems 14(1), 1–37 (2008)

20. Zhang, Y., Guo, Z., Wu, J., Tian, Y., Tang, H., Guo, X.: Real-time vehicle detection based on improved yolo v5. Sustainability 14, 12274 (09 2022)
