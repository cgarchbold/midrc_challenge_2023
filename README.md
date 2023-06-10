# mRALE Mastermind Challenge - Implementation
This repository contains the implementation of our model for the mRALE Mastermind Challenge, which focuses on predicting COVID severity on portable chest radiographs (CXRs).

## Challenge Summary
The mRALE Mastermind Challenge aimed to develop an AI/machine learning model capable of predicting COVID severity based on portable chest radiographs. The participants were provided with a dataset of CXRs and tasked with training a model that could accurately classify the severity of COVID in these images.

## Repository Structure

`\models`
-Contains code for various cnn models

## Install the dependencies:

`conda env create --file environment.yml`

`conda activate mrale`

## Train the model:

Modify the config in config.py to set up experiment

`python train.py`

Trained models are saved to `\experiments\(experiment_name)\saved_models`
Train plots are saved to `\experiments\(experiment_name)\plots`

## Evaluate the model:

`python test.py`

Results are saved to `\experiments\(experiment_name)\`