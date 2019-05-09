"""
Title:       recommended_model.py
Description:
Author:      Kunyu He, CAPP'20
"""


import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

from train import load_features, precision_at_k, clf_predict_proba


METRICS = [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]

THRESHOLDS = [0.01, 0.02, 0.05]

DEFAULT_ARGS = {'n_estimators': 1000, 'random_state': 123, 'max_features': 15,
                'oob_score': True, 'n_jobs': -1}

BEST_THRESHOLD = 0.7

BATCH = 0


#----------------------------------------------------------------------------#
if __name__ == "__main__":

    while BATCH < 3:
        X_train, X_test, y_train, y_test = load_features(BATCH)
        clf = RandomForestClassifier(**DEFAULT_ARGS)

        clf.fit(X_train, y_train)
        predicted_prob = clf_predict_proba(clf, X_test)
        predicted_labels = np.where(predicted_prob > BEST_THRESHOLD, 1, 0)
        print([metric(y_test, predicted_labels) for metric in METRICS])

        pred_prob_sorted, y_testsorted = zip(*sorted(zip(predicted_prob, y_test),
                                                     reverse=True))
        print([precision_at_k(y_testsorted, pred_prob_sorted, threshold) for threshold in THRESHOLDS])
        print("**-------------------------------------------------------------**\n\n")

        BATCH += 1
