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
