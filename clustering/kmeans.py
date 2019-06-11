"""
Title:       Build a preprocessing pipeline that helps user preprocess training
             and test data from the corresponding CSV input files.

Description: Fill in missing values, discretize continuous variables, generate
             new features, deal with categorical variables with multiple levels,
             scale data, and save preprocessed data.

Author:      Kunyu He, CAPP'20, The University of Chicago

"""

import logging
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans
from collections import OrderedDict
from sklearn.externals import joblib

INPUT_DIR = "../data/"
OUTPUT_DIR = "./models/"

LOG_DIR = "../logs/clustering/"

# logging
logger= logging.getLogger('clustering')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
logger.addHandler(ch)
fh = logging.FileHandler(LOG_DIR + time.strftime("%Y%m%d-%H%M%S") + '.log')
logger.addHandler(fh)

#----------------------------------------------------------------------------#
class KMeansPipeline:
    """

    """

    SEED = 123

    def __init__(self, X, data, Ks, output_dir=OUTPUT_DIR):
        """

        :param data:
        :param scaler:
        :param Ks:
        """
        self.X = X
        self.data = data
        self.Ks = Ks
        self.output_dir = output_dir

        self.distortions = OrderedDict()

    def cluster(self, k):
        """

        :param k:
        :return:
        """
        model = KMeans(n_clusters=k, random_state=self.SEED)
        file_name = self.output_dir + "KMeans_at_{}".format(k) + ".joblib"

        if not os.path.isfile(file_name):
            model.fit(self.X, file_name)
        else:
            model = joblib.load(file_name)

        if k not in self.Ks:
            self.Ks.append(k)
            self.Ks.sort()

        return model

    def find_best_k(self):
        """

        :return:
        """
        for k in self.Ks:
            model = self.cluster(k)
            self.distortions[k] = sum(np.min(cdist(self.X, model.cluster_centers_,
                                                   'euclidean'), axis=1)) / self.X.shape[0]

        plt.plot(self.distortions.keys(), self.distortions.values(), 'ko-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.show()

    def apply_cluster(self, k):
        """

        :param k:
        :return:
        """
        model = self.cluster(k)
        self.data["Cluster"] = pd.Series(model.labels_)

        return self

    def merge_clusters(self, to_merge, merged):
        """

        :param to_merge:
        :return:
        """
        for k in to_merge:
            self.data.Cluster[self.data.Cluster == k] = merged

        return self


if __name__ == "__main__":
    pass
