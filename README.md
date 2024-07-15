# Nutrient Estimation from Hyperspectral Images

This repository contains code for estimating nutrient concentrations from hyperspectral images using machine learning models.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Running Preprocessing Scripts](#running-preprocessing-scripts)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
- [Workflow](#workflow)

## Project Overview

This project involves estimating nutrient concentrations in plants using hyperspectral images. The workflow includes preprocessing hyperspectral data, training machine learning models, and evaluating the models' performance.

## Installation

To set up the project, follow these steps:

1. **Clone the repository:**

   git clone https://github.com/terrapretty/nutrient_estimation.git
   cd nutrient_estimation

2. **Create and activate the Conda environment:**

    conda create -n hs_nutrient python=3.9
    conda activate hs_nutrient
    
3. **Install the required packages:**

    pip install -r src/requirements.txt

## Usage
Running Preprocessing Scripts to Establish Baseline & Verify Background Removal

1. **Without Background Removal:**

    python src/preprocess_no_bg_removal.py --input /path/to/bil/images --output /path/to/output --calib /path/to/calibration.csv

2. **With Background Removal:**

    python src/run_plsr_with_bg_removal.py --input /path/to/bil/images --output /path/to/output --calib /path/to/calibration.csv

## Training the DL Model

To train the model using the preprocessed data, run:

    python src/train.py

Ensure you have set the correct paths in the script to your training and validation data.

## Evaluating the Model
After training, you can evaluate the model's performance using the evaluation scripts provided.

## Workflow

The workflow for this project involves the following steps:

1. **Obtain Baseline Results with PLSR using Unprocessed Hyperspectral Images:**

Preprocess the images without background removal and run PLSR.

    python src/preprocess_no_bg_removal.py --input /path/to/bil/images --output /path/to/output --calib /path/to/calibration.csv

2. **Remove Background and Rerun PLSR:**

Preprocess the images with background removal and rerun PLSR.

    python src/run_plsr_with_bg_removal.py --input /path/to/bil/images --output /path/to/output --calib /path/to/calibration.csv

3. **Compare Results of Steps 1 and 2:**

Evaluate and compare the results obtained with and without background removal. If background removal helps the baseline model, train the CNN model with preprocessing:

4. **Integrate background removal into the CNN model training pipeline and run the training script.**

    python src/train_with_bg_removal.py

Ensure all the paths are correctly set in the scripts before running them.