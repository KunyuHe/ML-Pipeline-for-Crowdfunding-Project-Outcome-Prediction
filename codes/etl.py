"""
Summary:     A collections of functions for ETL.

Description: This module is used for loading the data, loading and translating
             data types from the data dictionary, and saving them to the
             output directory "../clean_data/".
Author:      Kunyu He, CAPP'20
"""

import pandas as pd

from datetime import timedelta


INPUT_DIR = "../data/"
OUTPUT_DIR = "../data/"

DATA_FILE = "projects_2012_2013.csv"
OUTPUT_FILE = "projects.csv"
TARGET = 'fully_funded'
LENGTH = 60


#----------------------------------------------------------------------------#
if __name__ == "__main__":

    data = pd.read_csv(INPUT_DIR + DATA_FILE)

    data['date_posted'] = pd.to_datetime(data['date_posted'])
    data['datefullyfunded'] = pd.to_datetime(data['datefullyfunded'])
    data[TARGET] = ((data['datefullyfunded'] - data['date_posted']) >
                     timedelta(days=LENGTH)).astype(float)

    data.dropna(subset=['grade_level'], inplace=True)
    data.to_csv(OUTPUT_DIR + OUTPUT_FILE, index=False)

    print(("ETL process finished. Generated target for prediction as '{}' and"
           " data stored as '{}' under directory '{}'.".format(TARGET,
                                                               OUTPUT_FILE,
                                                               OUTPUT_DIR)))
