# ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction

[![Maintainability](https://api.codeclimate.com/v1/badges/990caf02345d986dbba9/maintainability)](https://codeclimate.com/github/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/maintainability) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/80262728540b41d3b6aca75031a74a1e)](https://www.codacy.com/app/kunyuhe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction&amp;utm_campaign=Badge_Grade)



## 0. To Reproduce My Results

Change you working directory to the folder where you want the project. Clone the repository to your local with:

```console
$ git clone https://github.com/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction.git
```

Then, run one of the following:

### 0.1 Windows

```console
$ chmod u+x run.sh
$ run.sh
```

### 0.2 Unix/Linux

```console
$ chmod +x script.sh
$ ./run.sh
```

## 1. Introduction

The task is to build a pipeline that predicts whether a crowdfunding project on will not get fully funded within 60 days of posting, based on data from `DonorChoose.org`.

The pipeline has six components:

1.  Read Data
2.  Explore Data
3.  Generate Features/Predictors
4.  Build Classifier
5.  Evaluate Classifier

The pipeline currently supports **seven classification algorithms**:

| Imputation       | Scalers         | Classification Algorithms | Metrics   |
| ------------- | --------------- | ------------------------- | --------- |
| Column Mean   | Standard Scaler | KNN                       | Accuracy  |
| Column Median | Mini-max Scaler | Logistic Regression       | Precision |
|               |                 | Decision Tree             | Recall    |
|               |                 | Linear SVC                | F-1 Score |
|               |                 | Bagging                   | AUC ROC   |
|               |                 | Boosting                  |           |
|               |                 | Random Forest             |           |

