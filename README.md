# ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/80262728540b41d3b6aca75031a74a1e)](https://www.codacy.com/app/kunyuhe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction&amp;utm_campaign=Badge_Grade) [![Maintainability](https://api.codeclimate.com/v1/badges/990caf02345d986dbba9/maintainability)](https://codeclimate.com/github/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/maintainability) [![Documentation Status](https://readthedocs.org/projects/pydocstyle/badge/?version=stable)](http://www.pydocstyle.org/en/stable/?badge=stable) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/master?filepath=%2FEDA%2FEDA.ipynb)



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



The pipeline currently supports **seven classification algorithms and five evaluation metrics**, it also implements **two imputation methods and two different scalers**. User can run any configuration of his/her choice. The implemented configurations are listed below.



| Imputation Methods | Scalers         | Classification Algorithms | Metrics   |
| ------------- | --------------- | ------------------------- | --------- |
| Column Mean   | Standard Scaler | KNN                       | Accuracy  |
| Column Median | Mini-max Scaler | Logistic Regression       | Precision |
|               |                 | Decision Tree             | Recall    |
|               |                 | Linear SVC                | F-1 Score |
|               |                 | Bagging                   | AUC ROC   |
|               |                 | Boosting                  |           |
|               |                 | Random Forest             |           |

*(Customizable Configurations)*



Upon running the program, use `keyboard input` to specify your configuration when notified to. You can also choose to run all possible configurations by sequence *(but it takes really long)*.

When the program runs, messages would be printed to console to give you a sense about what it is doing. Also at some points, there would be prompted plots reporting one of the following:

*   Distribution of the Predicted Probabilities
*   Precision, Recall Curve and Percent of Polpulation
*   Receiver Operating Characteristic Curve
*   Feature Importance *(Top 5)* Bar Plots *(if applicable)*



They would stay for 3 seconds and close automatically. Then they would be saved to `/log/images/`. **Please do not close them manually, or your progress would be killed**.



## 2. Get Data

*   Output Directory: `./data/ `   *(All paths hereby would be relative to the `/codes/` directory)*



Data can be manually downloaded from [this link](https://canvas.uchicago.edu/courses/20751/files/2388413/download?download_frd=1) on the `University of Chicago Canvas` as given. It is stored in the `./data/` directory as `projects_2012_2013.csv`.

The data is a CSV file that has one row for each project posted with a column for `date_posted` *(the date the project was posted)* and a column for `datefullyfunded` *(the date the project was fully funded)*. During  The data spans Jan 1, 2012 to Dec 31, 2013. Description of the variables can be found [here](<https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose/data>).



## 3. ETL

*   Input Directory: `../data/`
*   Output Directory: `../data/`
*   Code Script: [etl.py](https://github.com/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/blob/master/codes/etl.py)



As data comes as CSV, I used `Pandas` to read it into Python. Two columns `date_posted` and `datefullyfunded` are converted into `datatime` and the target `fully_funded` is generated based on whether at most 59 days had passed between those two dates. We label the observation "positive" if it gets fully funded within 60 days, and "negative" when it fails to do so.



## 4. EDA

*   Input Directory: `../data/`
*   Output Directory: `../EDA/images/`
*   Notebook: [EDA.ipynb](https://mybinder.org/v2/gh/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/master?filepath=%2FEDA%2FEDA.ipynb)
*   Code Script: [viz.py](https://github.com/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/blob/master/codes/viz.py)



*Try the interactive Jupyter Notebook supported by `binder` if you click on the badge above*!

There are 124973 observations, 88965 *(71%)* of which successfully got fully funded within 60 days of posting. 



### 4.1 Categorical Variables

![](https://github.com/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/blob/master/EDA/images/figure-3.png)

As the bar plots above indicate, nearly half of the schools are categorized as in areas of `highest poverty`. We transformed these two variables into ordinal in the *Feature Engineering* phase.



### 4.2 Correlation Among Variables

![](https://github.com/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/blob/master/EDA/images/figure-5.png)

As shown in the correlation triangle, our target, `fully_funded`, is relatively positively correlated with `school_latitude` and `eligible_double_your_impact_match`, and it's negatively correlated with `total_price_including_optional_support` as expected.

There are also some strong positive correlation between predictors, including that between `student_reached` and `grade_level`, `student_reached` and `total_price_including_optional_support` as expected, and `school_latitude` and `school_longitude`.



## 5. Feature Engineering

*   Input Directory: `../data/`
*   Output Directory: `../processed_data/`
*   Code Script: [featureEngineering.py](https://github.com/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/blob/master/codes/featureEngineering.py)



### 5.1 Observations: Drop Outliers and Impute Missing Values

I first dropped 3 outliers with extremely large values in `students_reached` and 1 in `total_price_including_optional_support`.

For non-numerical variables, I imputed the missing values based on the context. For example, I assumed that missing values in`primary_focus_subject`,`primary_focus_area`, and `resource_type` with `"Other"`. As there are not many values missing, this should be safe.  Users can modify `TO_FILL_NA` in the code script to change the way of imputation.

For numerical variables, users can specify their way to fill in with either column mean or median as indicated when the program runs.



### 5.2 Features: Generate New Features and Select Features

Based on findings of the EDA process, I modified some existing categorical variables to reduce number of levels they have. For example, I changed `teacher_prefix` into `gender` by assigning teachers with prefix `"Mrs."` and `"Ms."` with `"feamale"` and `"Mr."` with `"male"`. Here I turned it into a dummy. Users can change this behavior by modifying  `TO_COMBINE` in the code script.

Then I transformed all the binary dummies to 0-1 representation, and ordinal variables like poverty and grade level to integer variables. I also generated a new variable named `posted_month` from original `date_posted`.

For feature selection, I dropped some non-numerical variables because they contain too many missing values, including `school_metro` *(whether the school is in urban, suburban or rural area)*, which is really hard and dangerous to impute. I also dropped multinomial variables with too many unique levels so that after *one-hot encoding* I wouldn't end up with a large sparse matrix.



### 5.3 Train Test Split

In order to achieve best future predictions, we need to find the best model based on historical data. Hence in this practice, **we do not have true actual data** because they are not yet available. The whole data set would be our training set. However, for evaluation and optimization purposes, we would like to split it into training and validation sets. The `time_train_test_split()` function in the `featureEngineering` module implements this.

As the data spans over time and we are predicting whether a project would get fully funded within 60 days *(nearly two months)* of posting, I applied temporal train test split technique with test sets spanning 6 months. In other words, **I created training and test sets over time, and the test sets are six months long and the training sets are all the data before each test set**.

Therefore, there would be three pairs of training, test sets:

| Training                | Testing                 |
| ----------------------- | ----------------------- |
| 2012.01.01 - 2012.01.31 | 2012.02.02 - 2012.07.30 |
| 2012.01.01 - 2012.07.30 | 2012.08.02 - 2013.01.30 |
| 2012.01.01 - 2013.01.30 | 2013.02.02 - 2013.07.30 |

*(Time Span of Training and Test Sets)*



Obviously we wasted data from `2013-08-01` to `2013-10-30`, as we would not be able to know whether projects posted after that would be fully funded by looking at historical data of year 2012 and 2013. However, as we also have information from `"future"` on when those projects got fully funded, we wasted data from `2013-11-01` to `2013-12-31`, either.

This is a huge mistake and need to be improved in future patches. Also, we lost data on every first day of the first month in the test sets due to code issues. Fix those in later patches, too.

 

## 6. Build and Evaluate Classifier

*   Input Directory: `../processed_data/`
*   Output Directory: `../log/`
*   Code Script: [train.py](https://github.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/blob/master/train.py)



### 6.1 Benchmark: Default Decision Tree

For the training and evaluation part, I built the benchmark with a default *scikit-learn* `DecisionTreeClassifier`. Test metrics of the default decision tree is reported below:

| Metrics                                  | 2012.02 - 2012.07 | 2012.08 - 2013.01 | 2013.02 - 2013.07 |
| ---------------------------------------- | ----------------- | ----------------- | ----------------- |
| Accracy                                  | 0.6012            | 0.6166            |                   |
| Precision *( at decision threshold 0.5)* | 0.7031            | 0.7660            |                   |
| Recall *( at decision threshold 0.5)*    | 0.7170            | 0.6960            |                   |
| F-1 Score *( at decision threshold 0.5)* | 0.7100            | 0.7293            |                   |
| AUC ROC                                  | 0.5355            | 0.5241            |                   |

*(Benchmark Test Performances Across Test Periods)*



It seems that although for the first split our training set spans only one month, test performance of the benchmark model is not bad at all. Pattern in the data seems to be quite consistent.



### 6.2 Model Tuning

#### 6.2.1 Find the Best Decision Threshold

Users can build classifiers of their choices, and the first step after that, would be going through the automate process of finding the best decision threshold *(a test observation would be labeled `"positive"` when its predicted probability exceeds the decision threshold in a binary classification case)* within a predefined grid of thresholds *(`THRESHOLDS` in the code script)*.

While searching for the best decision threshold, the classifier is built with default parameters. Users can change this behavior by modifying `DEFAULT_ARGS` in the code script.

With 10-fold cross validation by default, the classifier would return a predicted probability for each observation in the validation set. It then goes down the threshold grid and label observations in the validation set. A score *(depending on the metrics of choice)* would be calculated on each validation set, and averaged as the final score of the threshold. Finally, we pick the threshold with the highest cross validation score. The process is illustrated below.



### 6.3 Model Evaluations

#### 6.3.1 Tables of Performance Metrics on All the Test Sets

After finding the best set of hyper-parameters of a specific model according to a specific metrics on the training set, the so far "best" model would be validated on the test set. Performances of all the "best" models on any of the three test sets would be recorded in `CSV` format under `../log/evaluations/` directory. There are three folders under the directory, each stores the `performances.csv` table recording performance metrics listed above. Take `test set 1`, which spans Aug. 2012 - Jan. 2013 as an example.

In terms of **Precision**, the best models on `test set 1` are listed below:

| Type                | Threshold | Hyperparameters                                      | Default Parameters                     | Accuracy | Precision | Recall | F1 Score | AUC ROC Score |
| ------------------- | --------- | ---------------------------------------------------- | -------------------------------------- | -------- | --------- | ------ | -------- | ------------- |
| Logistic Regression | 0.02      | {'penalty': 'l1', 'solver': 'liblinear',   'C': 0.1} | {'random_state': 123}                  | 0.7421   | 0.7423    | 0.9990 | 0.8520   | 0.5007        |
| Linear SVM          | 0.45      | {'penalty': 'l2', 'C': 0.01}                         | {'random_state': 123, 'max_iter': 200} | 0.7421   | 0.7421    | 1.0000 | 0.8520   | 0.5001        |

*(Best Models on Test Set 1 -- Accuracy)*


$$Accuracy = \frac{TP+TN}{TP+FP+TN+FN}$$

The two models have the same accuracy. It means they can correctly labels projects that would get fully funded within 60 days of posting or not correctly about 74.2% of the time, and they are as good as the other in terms of accuracy. The other performances metrics are nearly the same, too. In order to decide which one is better, we need to look further into their precision against percentage of population.

![](https://github.com/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/blob/master/log/images/best-models/(Linear%20SVM%20--%20Accuracy).png)

![](https://github.com/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/blob/master/log/images/1/precision recall/\(Logistic Regression -- Accuracy\).png)



