"""
Summary:     A collections of functions for visualization.

Description: contains a function that reads data and data types, and many
             other functions for visualization
Author:      Kunyu He, CAPP'20
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve


INPUT_DIR = "../data/"
OUTPUT_DIR = "../log/images/"

TITLE = FontProperties(family='serif', size=14, weight="semibold")
AXIS = FontProperties(family='serif', size=12)
TICKS = FontProperties(family='serif', size=10)

POSITIVE = 1
N_CLASSES = 2


#----------------------------------------------------------------------------#
def plot_predicted_scores(cv_scores, batch, title=""):
    """
    Plot the cross-validation predicted scores on the training set with a
    histogram.

    Inputs:
        - cv_scores (array of floats): cross-validation predicted scores

    Returns:
        None
    """
    fig, ax = plt.subplots()

    ax.hist(cv_scores, 10, edgecolor='black')

    ax.set_xlabel('Cross-Validation Predicted Scores', fontproperties=AXIS)
    ax.set_ylabel('Probability density', fontproperties=AXIS)
    ax.set_title('Frequency Distribution of Predicted Scores\n' + title,
                 fontproperties=AXIS)

    figname = OUTPUT_DIR + str(batch) + "/decision thresholds/" + "{}.png".format(title)
    fig.savefig(figname, dpi=400)


def plot_precision_recall(y_true, y_score, baseline, batch, title=""):
    """
    Generates plots for precision and recall curve. This function is
    adapted from https://github.com/rayidghani/magicloops.

    Inputs:
        y_true: (Series) the Series of true target values
        y_score: (Series) the Series of scores for the model
        title: (string) the name of the model

    Returns:
        None
    """
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)

    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score >= value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)

    pct_above_per_thresh = np.array(pct_above_per_thresh)

    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('Percent of Population', fontproperties=AXIS)
    ax1.set_ylabel('Precision', fontproperties=AXIS, color='b')

    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('Recall', fontproperties=AXIS, color='r')

    plt.axhline(baseline, ls='--', color='black')

    ax1.set_ylim([0, 1.05])
    ax2.set_ylim([0, 1.05])
    ax1.set_xlim([0, 1])

    figname = OUTPUT_DIR + str(batch) + "/precision recall/" + "{}.png".format(title)
    plt.title("Precision, Recall, and Percent of Population\n" + title,
              fontproperties=AXIS)
    fig.savefig(figname, dpi=400)


def plot_auc_roc(clf, data, batch, title=""):
    """
    """
    X_train, X_test, y_train, y_test = data
    y_train = label_binarize(y_train, classes=[0, 1, 2])
    y_test = label_binarize(y_test, classes=[0, 1, 2])

    classifier = OneVsRestClassifier(clf)
    if hasattr(classifier, "decision_function"):
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    else:
        y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

    # Compute ROC curve and ROC area for binary classes
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(N_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    fig, _ = plt.subplots()
    plt.plot(fpr[POSITIVE], tpr[POSITIVE], color='darkorange', lw=1.5,
             label='ROC curve (area = {:.4f})'.format(roc_auc[POSITIVE]))
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate', fontproperties=AXIS)
    plt.ylabel('True Positive Rate', fontproperties=AXIS)
    plt.legend(loc="lower right")

    figname = OUTPUT_DIR + str(batch) + "/roc/" + "{}.png".format(title)
    plt.title('Receiver Operating Characteristic Curve\n' + title,
              fontproperties=AXIS)
    fig.savefig(figname, dpi=400)


def plot_feature_importances(importances, col_names, batch, n=5, title=""):
    """
    Plot the feature importances of the decision tree. This credit to the
    University of Michigan.

    Inputs:
        clf: the model
        feature_names: (list) the list of strings to store feature names

    Returns:
        None
    """
    indices = np.argsort(importances)[::-1][:n]
    labels = col_names[indices][::-1]

    fig, _ = plt.subplots(figsize=[12, 8])
    plt.barh(range(n), sorted(importances, reverse=True)[:n][::-1], color='g',
             alpha=0.4, edgecolor=['black']*n)

    plt.xlabel("Feature Importance", fontproperties=AXIS)
    plt.ylabel("Feature Name", fontproperties=AXIS)
    plt.yticks(np.arange(n), labels, fontproperties=AXIS)

    figname = OUTPUT_DIR + str(batch) + "/feature importance/" + "{}.png".format(title)
    plt.title("Feature Importance: Top {}\n".format(n) + title,
              fontproperties=AXIS)
    fig.savefig(figname, dpi=400)
