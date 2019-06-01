#!/bin/bash

py -m venv env
.\env\Scripts\activate

pip install --user -r requirements.txt


cd ./codes
python etl.py
start https://mybinder.org/v2/gh/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/master?filepath=%2FEDA%2FEDA.ipynb
python featureEngineering.py --ask_user 0 --verbose 1
python train.py --clean_start 0 --ask_user 0 --verbose 1 --plot 1

deactivate
