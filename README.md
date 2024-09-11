# Nutrient Estimation from Hyperspectral Images

This repository contains code for estimating nutrient concentrations from hyperspectral images using machine learning models.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Running Preprocessing Scripts](#running-preprocessing-scripts)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)

## Project Overview

This project involves estimating nutrient concentrations in plants using hyperspectral images. The workflow includes preprocessing hyperspectral data, training machine learning models, and evaluating the models' performance.

## Installation

To set up the project, follow these steps:

1. **Clone the repository:**
```sh
   git clone https://github.com/terrapretty/nutrient_estimation.git
   cd nutrient_estimation
```

2. **Create and activate the Conda environment:**
```sh
    conda create -n hs_nutrient python=3.9
    conda activate hs_nutrient
```   
3. **Install the required packages:**
```sh
    pip install -r src/requirements.txt
```
## Usage
Running Preprocessing Scripts to Establish Baseline & Verify Background Removal

1. **Resize Raw Hypercubes**
```sh
    python src/resize_hs.py --input /path/to/bil/images --output /path/to/output --calib /path/to/calibration.csv
```
## Training the DL Model

To train the model using the preprocessed data, run:
```sh
    python src/train.py
```
Ensure you have set the correct paths in the script to your training and validation data.

## Evaluating the Model
After training, you can evaluate the model's performance using the evaluation scripts provided.

