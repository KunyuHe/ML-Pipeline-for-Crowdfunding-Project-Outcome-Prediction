#!/bin/bash

pip install --user -r requirements.txt

cd ./codes
python etl.py
start 
python featureEngineering.py
python train.py
