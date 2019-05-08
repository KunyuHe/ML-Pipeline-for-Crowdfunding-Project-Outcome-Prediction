"""
Summary:     A collections of functions to generate features.

Description:
Author:      Kunyu He, CAPP'20
"""

import os
import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from viz import read_data


pd.set_option('mode.chained_assignment', None)

INPUT_DIR = "../data/"
OUTPUT_DIR = "../processed_data/"

DATA_FILE = "projects.csv"

MAX_OUTLIERS = {'students_reached': 3,
                'total_price_including_optional_support': 1}

TO_FILL_NA = {'primary_focus_subject': "Other",
              'primary_focus_area': "Other",
              'secondary_focus_subject': "None",
              'secondary_focus_area': "None",
              'resource_type': "Other"}

TO_COMBINE = {'teacher_prefix': {'female': ['Mrs.', 'Ms.'],
                                 'male': ['Mr.', 'Dr.']},
              'resource_type': {'Other': ["Trips", "Visitors"]}}

TO_BINARIES = {'school_charter': 'auto',
               'school_magnet': 'auto',
               'poverty_level': [['highest poverty', 'high poverty',
                                  'moderate poverty', 'low poverty']],
               'grade_level': [['Grades PreK-2', 'Grades 3-5',
                                'Grades 6-8', 'Grades 9-12']],
               'eligible_double_your_impact_match': 'auto',
               'teacher_prefix': [['male', 'female']],
               'secondary_focus_area': [["None", "Yes"]],
               'secondary_focus_subject': [["None", "Yes"]]}

TO_EXTRACT_DATE_TIME = {'date_posted': ["month"]}

TO_ONE_HOT = ['primary_focus_subject', 'primary_focus_area', 'resource_type']

TARGET = 'fully_funded'
TO_DROP = ['projectid', 'teacher_acctid', 'schoolid', 'school_ncesid',
           'school_city', 'school_state', 'school_district', 'school_county',
           'datefullyfunded', 'school_metro', 'fully_funded']

TO_DESCRETIZE = {'school_longitude': 5}
RIGHT_INCLUSIVE = {'school_longitude': True}

SCALERS = [StandardScaler, MinMaxScaler]


#----------------------------------------------------------------------------#
def drop_max_outliers(data, drop_vars):
    """
    Takes a data set and a dictionary mapping names of variables whose large
    extremes need to be dropped to number of outliers to drop, drops those
    values.

    Inputs:
        - data (DataFrame): data matrix to modify.
        - drop_vars {string: int}: mapping names of variables to number of
            large outliers to drop.

    Returns:
        (DataFrame) the modified data set.

    """
    for col_name, n in drop_vars.items():
        data.drop(data.nlargest(n, col_name, keep='all').index,
                  axis=0, inplace=True)

    return data


def fill_na(data, to_fill):
    """
    """
    for var, fill in to_fill.items():
        data[var].fillna(value=fill, inplace=True)
        print("\tFilled missing values in '{}' with '{}'.".format(\
              var, fill))

        if fill == "None":
            TO_COMBINE[var] = {"Yes": [col for col in list(data[var].unique())
                                       if col != "None"]}

    return data


def to_combine(data, to_combine_vars):
    """
    """
    for var, dict_combine in to_combine_vars.items():
        print("\tCombinations of levels on '{}':".format(var))
        for combined, lst_combine in dict_combine.items():
            data.loc[data[var].isin(lst_combine), var] = combined

    return data


def to_binary(data, to_bin_vars):
    """
    Takes a data set and a dict of variables that needed to be trasformed to
    binaries, performs the transformations and returns the modified data set.

    Inputs:
        - data (DataFrame): data matrix to modify.
        - to_bin_vars ({string: string/[[]]}): mapping names of variables that
            needed to be trasformed to binaries to 'auto' or unique levels.

    Returns:
        (DataFrame): the modified data matrix.

    """
    for var, cats in to_bin_vars.items():
        enc = OrdinalEncoder(categories=cats)
        data[var] = enc.fit_transform(np.array(data[var]).reshape(-1, 1))

    return data


def extract_date(data, to_extract):
    """
    """
    for var, lst_extract in to_extract.items():
        can_extract = {"year": data[var].dt.year, "month": data[var].dt.month,
                       "weekday": data[var].dt.weekday, "day": data[var].dt.day}

        for extract in lst_extract:
            new_col = var + "_" + extract
            data[new_col] = can_extract[extract]
            TO_ONE_HOT.append(new_col)
            print("\tExtracted {} from '{}' into '{}'.".format(extract, var,
                                                               new_col))

    return data


def one_hot(data, cat_vars):
    """
    Takes a data set and a list of categorical variables, creates binary/dummy
    variables from them, drops the original and inserts the dummies back to the
    data set.

    Inputs:
        - data (DataFrame): data matrix to modify.
        - cat_vars (list of strings): categorical variables to apply one-hot
            encoding.

    Returns:
        (DataFrame): the modified data matrix.

    """
    for var in cat_vars:
        dummies = pd.get_dummies(data[var], prefix=var)
        data.drop(var, axis=1, inplace=True)
        data = pd.concat([data, dummies], axis=1)

    return data


def time_train_test_split(data, colname, freq=None):
    """
    Creates temporal train/test splits. This function is adapted from
    https://github.com/rayidghani/magicloops.

    Inputs:
        data: (DataFrame) the data used for split
        time_colname: (string) the name of the column indicating time
        freq (string): the time gap for the testing, and it should be
            specified as '1M', '3M', etc.
    Returns:
        lst_train: (list) list of DataFrames used for training
        lst_test: (list) list of DataFrames used for testing
        lst_gap: (list )list of startpoints used for split
    """
    lst_gap = []
    lst_train = []
    lst_test = []
    lst_starts = pd.date_range(start=data[colname].min(),
                               end=data[colname].max(),
                               freq=freq)

    for i, start in enumerate(lst_starts[:-1]):
        cut = start + pd.DateOffset(1)
        train = data.loc[data[colname] <= start].drop(colname, axis=1)
        test = data.loc[(data[colname] > cut) &
                        (data[colname] < lst_starts[i + 1])]\
                   .drop(colname, axis=1)

        lst_train.append(train)
        lst_test.append(test)
        lst_gap.append(start)

    return lst_train, lst_test, lst_gap


def ask():
    """
    """
    scaler_index = int(input(("Up till now we support:\n"
                              "\t1. StandardScaler\n"
                              "\t2. MinMaxScaler\n"
                              "Please input a scaler index (1 or 2):\n")))

    imputer_index = int(input(("\nUp till now we support:\n"
                               "\t1. Imputing with column mean\n"
                               "\t2. Imputing with column median\n"
                               "Please input the index (1 or 2) of your"
                               " imputation method:\n")))

    return imputer_index, scaler_index


def split(data):
    """
    Drop rows with missing labels, split the features and targert.

    Inputs:
        - data (DataFrame): the data matrix.

    Outputs:
        (X_train, X_test, y_train, y_test)

    """
    data.dropna(axis=0, subset=[TARGET], inplace=True)
    y = data[TARGET]
    data.drop(TO_DROP, axis=1, inplace=True)
    X = data

    return X, y


def impute(X_train, X_test, imputer_index):
    """
    Take the data matrix, and impute the missing features with a customized
    column feature, then set the data types as preferred from the data
    dictionary.

    Inputs:
        - X_train (arry): training features.
        - X_test (array): testing features.
        - data_types (dict): mapping column name
        - ask (bool): whether to ask for imputer index from the user.

    Returns:
        (DataFrame, DataFrame) the modified training features and test
            features.

    """
    if imputer_index == 1:
        X_train = X_train.fillna(X_train.mean())
        X_test = X_test.fillna(X_test.mean())
    else:
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_test.median())

    return X_train, X_test


def discretize(X_train, X_test):
    """
    Discretizes a continuous variable into a specific number of bins.

    Inputs:
        - X_train (arry): training features.
        - X_test (array): testing features.
        - cont_vars ({string: int}): mapping column names to number of bins to
            cut the columns in.
        - right_inclusive ({string: bool}): mapping column names to whether to
            cut the series into right inclusive of not.

    Returns:
        (DataFrame) the modified data set.

    """
    for col_name, n in TO_DESCRETIZE.items():
        X_train[col_name] = pd.cut(X_train[col_name], n,
                                   right=RIGHT_INCLUSIVE[col_name]).cat.codes
        X_test[col_name] = pd.cut(X_test[col_name], n,
                                  right=RIGHT_INCLUSIVE[col_name]).cat.codes
        print("\tDiscretized '{}' into {} bins.".format(col_name, n))

    return X_train, X_test


def scale(X_train, X_test, scaler_index):
    """
    Asks user for the scaler to use, or uses default standard scaler. Fits it
    on the training data and scales the test data with it if test data is
    provided.

    Inputs:
        - X_train (arry): training features.
        - X_test (array): testing features.

    Returns:
        (array, array): train and test data after scaling.

    """
    with open(OUTPUT_DIR + 'col_names.pickle', 'wb') as handle:
        pickle.dump(X_train.columns, handle)

    scaler = SCALERS[scaler_index - 1]()
    X_train = scaler.fit_transform(X_train.values.astype(float))
    X_test = scaler.transform(X_test.values.astype(float))

    return X_train, X_test


def save_data(X_train, X_test, y_train, y_test, i):
    """
    Saves traning and testing data as numpy arrays in the output directory.

    Inputs:
        - X_train (array): training features.
        - X_test (array): testing features.
        - y_train (array): training target.
        - y_test (array): testing target.

    Returns:
        None

    """
    if "processed_data" not in os.listdir("../"):
        os.mkdir("processed_data")

    np.savez(OUTPUT_DIR + 'X{}.npz'.format(i), train=X_train, test=X_test)
    np.savez(OUTPUT_DIR + 'y{}.npz'.format(i), train=y_train.values.astype(float),
                                               test=y_test.values.astype(float))

    print(("Saved the resulting NumPy matrices to directory {}. Features are"
           " in 'X{}.npz' and target is in 'y{}.npz'. Column names are saved as"
           " 'col_names.pickle'.").format(OUTPUT_DIR, i, i))


def process():
    """
    Reads the data set, drops rows with large extremes, converts some
    variables into binaries, combines some binaries into categoricals or
    ordinals, and applies one-hot encoding on categoricals. Then splits the
    data set in to training and test, impute missing values separately, and
    descretizes some continuous variables into ordinals. Then save the data as
    NumPy arrays to the output directory.

    Inputs:
        - data (DataFrame): data matrix to modify.

    Returns:
        (DataFrame): the processed data matrix.

    """
    # load data
    data = read_data()
    print("Finished reading cleaned data.\n")

    # drop rows with large extremes
    data = drop_max_outliers(data, MAX_OUTLIERS)
    print("Finished dropping extreme large values:")
    for col_name, n in MAX_OUTLIERS.items():
        print("\tDropped {} observations with extreme large values on '{}'.".\
              format(n, col_name))

    # fill in missing values
    print("\nFinished filling in missing data.")
    data = fill_na(data, TO_FILL_NA)

    # combine levels of some categoricals
    print("\nFinished combining levels of some categoricals.")
    data = to_combine(data, TO_COMBINE)

    # convert some variables into binaries
    data = to_binary(data, TO_BINARIES)
    print("\nFinished transforming the following variables: {}.\n".\
          format(list(TO_BINARIES.keys())))

    # apply one-hot encoding on categoricals
    data = one_hot(data, TO_ONE_HOT)
    print("Finished one-hot encoding the following categorical variables: {}\n".\
          format(TO_ONE_HOT))

    # split the data into training and test sets
    trains, tests, _ = time_train_test_split(data, 'date_posted', freq='6M')
    imputer_index, scaler_index = ask()

    for i in range(len(trains)):
        train, test = trains[i], tests[i]
        X_train, y_train = split(train)
        X_test, y_test = split(test)

        # do imputation to fill in the missing values
        X_train, X_test = impute(X_train, X_test, imputer_index)
        print(("\n\n**-------------------------------------------------------------**\n"
              "Finished imputing missing values with feature {} for window {}.\n").\
              format(["means", "medians"][imputer_index - 1], i))

        # discretize some continuous features into ordinals
        print("Finished discretizing some continuous variables for window {}:".\
              format(i))
        X_train, X_test = discretize(X_train, X_test)

        # scale training and test data
        X_train, X_test = scale(X_train, X_test, scaler_index)
        print("Finished extracting the target and scaling the features for window {}.\n".\
              format(i))

        save_data(X_train, X_test, y_train, y_test, i)


#----------------------------------------------------------------------------#
if __name__ == "__main__":

    process()
