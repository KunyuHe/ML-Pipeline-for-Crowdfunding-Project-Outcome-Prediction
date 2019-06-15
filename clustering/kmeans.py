"""
Title:       Build a preprocessing pipeline that helps user preprocess training
             and test data from the corresponding CSV input files.

Description: Fill in missing values, discretize continuous variables, generate
             new features, deal with categorical variables with multiple levels,
             scale data, and save preprocessed data.

Author:      Kunyu He, CAPP'20, The University of Chicago

"""

import sys
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
from sklearn.tree import DecisionTreeClassifier

sys.path.append('../codes/')
from featureEngineering import create_dirs, FeaturePipeLine
from trainviz import plot_feature_importances, plot_decision_tree

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
        Construct a KMeansPipeline object.

        Parameters:
            - X (NumPy ndarray): feature matrix, can be None
            - data (Pandas DataFrame): data set to apply clustering on
            - Ks (list of ints): potential number of clusters to try
            - output_dir (string): relative path to save models of this pipeline

        """
        self.X = X
        self.data = data
        self.Ks = Ks
        self.output_dir = output_dir

        self.distortions = OrderedDict()

    def preprocess(self, batch_name):
        """
        For KMeansPipeline object without a feature matrix, preprocess the data
        and obtain the feature matrix.

        Inputs:
            - batch_name (string): for naming the directory to save the scaler

        """
        if not self.X:
            train_pipe = FeaturePipeLine(batch_name, self.data.copy(deep=True),
                                         ask_user=False, test=False)
            train_pipe.to_combine().one_hot()
            train_pipe.X = train_pipe.data.copy(deep=True)
            train_pipe.scale()

            self.X = train_pipe.X.copy()

    def cluster(self, k):
        """
        Cluster the feature matrix into k clusters.

        Inputs:
            - k (int): number of clusters

        Returns:
            KMeans model object

        """
        model = KMeans(n_clusters=k, random_state=self.SEED)
        file_name = self.output_dir + "KMeans_at_{}".format(k) + ".joblib"

        if not os.path.isfile(file_name):
            model.fit(self.X, file_name)
            joblib.dump(model, file_name)
        else:
            model = joblib.load(file_name)

        if k not in self.Ks:
            self.Ks.append(k)
            self.Ks.sort()

        return model

    def find_best_k(self):
        """
        Find the best number of clusters with elbow plot.

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
        Add labels from the KMeans model to the data we are clustering on.

        Inputs:
            - k (int): number of clusters to apply

        Returns:
            (KMeansPipeline) with the `data` attribute clustered

        """
        model = self.cluster(k)
        self.data["Cluster"] = pd.Series(model.labels_)

        return self

    def merge_clusters(self, to_merge, merged):
        """
        Merge many clusters into one.

        Inputs:
            - to_merge (list of ints): the clusters to merge
            - merged (int): the cluster to merge into

        Returns:
            (KMeansPipeline) with clusters in the `data` attribute merged

        """
        for k in to_merge:
            self.data.Cluster[self.data.Cluster == k] = merged

        return self

    def split_cluster(self, to_split, sub_Ks, sub_path):
        """
        Split a cluster further into many clusters.

        Inputs:
            - to_split (int): to indicate the cluster to further split on
            - sub_Ks (list of ints): number of clusters to try and find the best
                one
            - sub_path (string): relative path to save models on the sub-cluster

        Returns:
            (KMeansPipeline) an object built on preprocessed data of a cluster
                to further split on

        """
        sub_data = self.get_sub_data([to_split])
        sub_pipe = KMeansPipeline(None, sub_data , sub_Ks, output_dir=sub_path)
        sub_pipe.preprocess("Cluster 47")

        create_dirs(sub_path)
        sub_pipe.find_best_k()
        return sub_pipe

    def get_sub_data(self, clusters):
        """
        Slice the data set by choosing a set of clusters.

        Inputs:
            - clusters (list of ints): clusters to take from the data set

        Returns:
            (Pandas DataFrame) a sliced dataframe

        """
        sub_data = self.data[self.data.Cluster.isin(clusters)].copy(deep=True)

        return sub_data.drop("Cluster", axis=1)

    def get_sub_features(self, clusters):
        """

        """
        row_index = self.get_sub_data(clusters).index

        return self.X[row_index]

    def describe_cluster(self, cluster):
        """
        Print the summary statistics of the assigned cluster.

        Inputs:
            - cluster (int): the cluster to describe

        Returns:
            (Pandas DataFrame) summary statistics of the cluster

        """
        sub_data = self.get_sub_data([cluster])

        return sub_data.describe()

    def find_distinctive_features(self, cluster, depth):
        """

        :param cluster:
        :return:
        """
        X_train = self.X
        y_train = np.where(self.data.Cluster == cluster, 1, 0)
        clf = DecisionTreeClassifier(random_state=self.SEED, max_depth=depth)

        clf.fit(X_train, y_train)
        importances = clf.feature_importances_

        cluster_pipe = FeaturePipeLine("", self.data.copy(deep=True),
                                       ask_user=False, test=False)
        cluster_pipe.to_combine().one_hot()
        col_names = cluster_pipe.data.columns

        plot_feature_importances(importances, col_names, "", top_n=depth,
                                 title=("Cluster %s against All Others" % cluster))

        col_names = [name for name in col_names if "school_city" not in name]
        plot_decision_tree(clf, col_names, "Cluster", "./")


if __name__ == "__main__":
    pass
