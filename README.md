# ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/80262728540b41d3b6aca75031a74a1e)](https://www.codacy.com/app/kunyuhe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction&amp;utm_campaign=Badge_Grade)
[![Maintainability](https://api.codeclimate.com/v1/badges/990caf02345d986dbba9/maintainability)](https://codeclimate.com/github/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/maintainability)
[![Documentation Status](https://readthedocs.org/projects/pydocstyle/badge/?version=stable)](http://www.pydocstyle.org/en/stable/?badge=stable)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/master?filepath=%2FEDA%2FEDA.ipynb)



## Introduction

The task is to build a pipeline that predicts whether a crowdfunding project on will not get fully funded within 60 days of posting, based on data from `DonorChoose.org`.



The pipeline has five components:

1. Read Data
2. Explore Data
3. Feature Engineering
4. Build Classifier
5. Evaluate Classifier



There are three major advantages of this pipeline compared to its counterparts or predecessors that I would like to highlight in the next section.



## Four Major Advantages

### 1. Supports a Variety of Classification Algorithms and Evaluation Metrics

The pipeline currently supports **ten classification algorithms and eight evaluation metrics** in the training and evaluation phase. It also implements **two different scalers** in the feature engineering phase. Users can run any configuration of their choice, or by default scan through all the potential combinations of them. The implemented configurations for `train.py` are listed below.



| Classification Algorithms | Metrics                                      |
| ------------------------- | -------------------------------------------- |
| Logistic Regression       | Accuracy                                     |
| Decision Tree             | Precision                                    |
| Random Forest             | Recall                                       |
| Bagging                   | F-1 Score                                    |
| Adaptive Boosting         | AUC ROC                                      |
| Gradient Boosting         | Scalers                                      |
| Extra Tree                | Standard Scaler                              |
| Linear SVM                | Precision @ [1%, 2%, 5%, 10%, 20%, 30%, 50%] |
| KNN                       |                                              |
| Naïve Bayes               |                                              |

*(Customizable Configurations for `train.py`)*



### 2. OOP Design

The preprocessing pipeline and modeling pipeline are both implemented as a Python class. Scalers, models and metrics the pipelines support are listed as class variables, and a specific configuration is listed in instance variables. This design makes later improvements, like adding other classification algorithms and including extra metrics for evaluation, much easier. The code is highly modular, relatively maintainable and easy to extend.



### 3. Warm Start for Recovery from Unexpected Interruptions

One of the caveats of its predecessor is that if you users would like to run all possible combinations of configurations, if there is any unexpected interruption, the program need to start from scratch to fit and evaluate models (although the past evaluations are maintained), which can be really time consuming. Besides, under the same model configuration, system in the past would fit, tune, and evaluate the same model with same sets of hyperparameters all over again. Past design is neither robust to interruptions, nor efficient in running time.

For this version, both issues are resolved to some extend. The solution is basically to save the cross-validation predicted probabilities for each specific pair of model and hyperparameter set in the tuning phase. If you we the same seed for the random generator, the cross-validation iterator would split the data in the same way as is, and the same pair of model and hyperparameter set would always generate the same cross-validation predicted probabilities. With these probabilities recorded, the pipeline can retrieve those from last training after any interruption or intended early stopping. Across decision thresholds and evaluation metrics, as the predicted probabilities are saved, the time consumption is merely retrieving the records. The efficiency of the system improves significantly.

However, this design still need to be improved. The most apparent caveat is that the system track hyperparameter sets with serial index (*labeled from 0 to the number of sets*) if the users make any change to the hyperparameter grid, same index could point to different sets of hyperparameters and simply retrieving the corresponding record of predicted probabilities is not reasonable. One potential solution is to use a hash function to link the sets and records so that the system would save to and retrieve from different locations if the set of hyperparameter changes.



### 4. Automate Running and Logging

When the program runs, if users specify that the pipeline shall not ask for configurations, then messages would be printed to console to give you a sense about what is going on. The same message would also be written to log files stamped with starting time under the directory `./logs/train/logs`. Users can also specify the level of verbosity of the pipeline.

The fine-tuned models would be evaluated on the test and plots for visualized evaluations would prompt, reporting one of the following:



- Distribution of the Predicted Probabilities
- Precision, Recall at Percentages of Population
- Receiver Operating Characteristic Curve
- Feature Importance *(Top 5)* Bar Plots *(if applicable)*



They would stay for 3 seconds and close automatically. Then they would be saved to `/logs/train/images/`. The training and test time, all the evaluation metrics of the best models would be saved under `/evaluations/` by name of the metrics the models were optimized on.



## To Reproduce My Results

Change you working directory to the folder where you want the project. Clone the repository to your local with:

```console
$ git clone https://github.com/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction.git
```



**NOTE**: the cloning process might be slower than expected as the repo contains cross-validation predicted probabilities to give users a warm start.



Then, run one of the following:

- Windows

```console
$ chmod u+x run.sh
$ run.sh
```

- Linux

```console
$ chmod +x script.sh
$ ./run.sh
```



**NOTE**: users can alter input parameters in the shell script to change the behavior of the scripts for Feature Engineering (`featureEngineering.py`) and Training (`train.py`). You can use:

```console
$ python featureEngineering.py -h
$ python train.py -h
```

to check what's user inputs are available for each script. **Remember that `--start_clean=0` is mandatory for a warm start.** However, if you changed the random seed or the hyperparameter grid, please change it to `--start_clean=1` to obtain the best classifiers under the modified context. 



## 0. Get Data

> - Output Directory: `../data/ `   *(All paths hereby would be relative to the `/codes/` directory)*

Data can be manually downloaded from [this link](https://canvas.uchicago.edu/courses/20751/files/2388413/download?download_frd=1) on the `University of Chicago Canvas` as given. It is stored in the `../data/` directory as `projects_2012_2013.csv`. For external users, as I've uploaded the data, it would be there after cloning the repo. 

The data is a CSV file that has one row for each project posted with a column for `date_posted` *(the date the project was posted)* and a column for `datefullyfunded` *(the date the project was fully funded)*. The data spans Jan 1, 2012 to Dec 31, 2013. Description of the variables in more detail can be found [here](<https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose/data>).



## 1. ETL

> - Input Directory: `../data/`
>
> - Output Directory: `../data/`
>
> - Code Script: [etl.py](https://github.com/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/blob/master/codes/etl.py)
> - Test Script: *in progress*



As data comes as CSV, I used `Pandas` to read it. Two columns `date_posted` and `datefullyfunded` are converted into `datatime` and the target `fully_funded` is generated based on whether at most 59 days had passed between those two dates. We label the observation `"positive"` *(or as 1)*  if it **fails to get fully funded within 60 days**, and `"negative"` *(or as 0)* when it succeed to do so.



## 2. EDA

> - Input Directory: `../data/`
> - Output Directory: `../EDA/images/`
> - Notebook: [EDA.ipynb](https://mybinder.org/v2/gh/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/master?filepath=%2FEDA%2FEDA.ipynb)
> - Code Script: [viz.py](https://github.com/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/blob/master/codes/viz.py)
> - Test Script: *in progress*



*Try the interactive Jupyter Notebook supported by `binder` if you click on the badge above*!

There are 124973 observations, nearly 29% of which didn't get fully funded within 60 days of posting. 



### 2.1 Categorical Variables

![](https://github.com/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/blob/master/EDA/images/figure-3.png)

As the bar plots above indicate, nearly half of the schools are categorized as in areas of `highest poverty`. We transformed these two variables into ordinal in the *Feature Engineering* phase.



### 2.2 Correlation Among Variables

![](https://github.com/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/blob/master/EDA/images/figure-5.png)

As shown in the correlation triangle, our target, `fully_funded`, is relatively positively correlated with `school_latitude` and `eligible_double_your_impact_match`, and it's negatively correlated with `total_price_including_optional_support` as expected.

There are also some strong positive correlation between predictors, including that between `student_reached` and `grade_level`, `student_reached` and `total_price_including_optional_support` as expected, and `school_latitude` and `school_longitude`.



## 3. Feature Engineering

> - Input Directory: `../data/`
> - Output Directory: `../processed_data/`
> - Logging Directory = `../logs/featureEngineering/`
> - Code Script: [featureEngineering.py](https://github.com/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/blob/master/codes/featureEngineering.py)
> - Test Script: *in progress*



### 3.1 Temporal Train Test Split

In order to achieve best future predictions, we need to find the best model based on historical data. In this practice, **we do not have true actual data** because they are not yet available. I performed feature engineering and modeling in the same manner that I split the data into several training - test pairs that each **test set has a fixed time window of 4 months**, and **the corresponding training set is all observations available 60 days before the start date of the test window**.

**NOTE**: Here I left a gap of 60 days between the training set and test set in each pair so that the last observation in the training set is not using any information "from the future", as for each posted project we need at least 60 days to know whether it actually failed to get fully funded within 60 days.

The data spans Jan 1, 2012 to Dec 31, 2013. Each training - test pair is called a `Batch` and we have 4 batches in total. In order to let `Batch 0` have enough training data, the first test set starts from July of 2012. Test set of `Batch 1` starts from the end of that in `Batch 0`. **The 60-days gaps I left untouched in each batch were actually included in training sets in later batches**. However, the last 60 days in our data will not be used in any batch, as we will not know their results in theory and cannot label them. The `temporal_split(*args)` function in the `featureEngineering` module implemented the split.

The following table shows the time spans of the four batches:

| Batch   | Training                 | n_train | Testing                  | n_test | m (features count) |
| ------- | ------------------------ | ------- | ------------------------ | ------ | ------------------ |
| Batch 0 | 2012-01-01 to 2012-04-30 | 21179   | 2012-06-30 to 2012-10-28 | 22986  | 78                 |
| Batch 1 | 2012-01-01 to 2012-08-29 | 34227   | 2012-10-29 to 2013-02-26 | 18009  | 83                 |
| Batch 2 | 2012-01-01 to 2012-12-28 | 59061   | 2013-02-27 to 2013-06-27 | 13336  | 88                 |
| Batch 3 | 2012-01-01 to 2013-04-28 | 74109   | 2013-06-28 to 2013-10-26 | 30021  | 89                 |

*(Time Span of Training and Test Sets)*



**NOTE**: besides the last 60 days of the whole data set, I left 2013-10-27 to 2013-11-01 unused. This is because I implemented a fixed time window for test sets in each batch. This caveat can be fixed if I use variable time window for test set in `Batch 3`.

In the `featureEngineering` module, the temporal split of data is performed first, and two preprocessing pipelines are constructed with the training and test sets for each batch. The preprocessing is thus independent between training and test sets.



### 3.2 Impute Missing Values

For non-numerical variables, I imputed the missing values in a variety of ways. For variables with value missing because they actually should not have any value in some rows, I filled in with `"MISC"`. For those with only a few rows that have missing values, I filled in with either a new level `"Other"`, or the mode of the column. Users can modify class variable `TO_FILL_NA`to change its behavior.

For numerical variables, I filled in missing values in `students_reached` with column median. Users can specify their way to fill in by modifying the function `con_fill_na()`.



### 3.3 Feature Generation

Based on findings of the EDA process, I modified some existing categorical variables to reduce number of levels they have. For example, I manually changed `teacher_prefix` by assigning teachers with prefix `"Dr."` to `"Mr."` (*no offense to female*). For some multinomial variables I combined levels that has less than 5% of the total into a new (or in some cases existing) level automatically. Users can change this behavior by modifying class variable `TO_COMBINE`.

To reduce the effects of outliers, besides scaling I discretized `students_reached` and `total_price_including_optional_support`. I also generated new variables by extracting "year" and "month of year" from the original variable `date_posted`. I then applied one-hot-encoding to all the multi-level categorical predictors.

**NOTE**: one troublesome scenario that's common after applying one-hot-encoding is that some levels appear in training set are not presented in the test sets, which leads to different number of features in training and test feature matrices. For these variables, insert a column with all zeros at the same column index in the test set. Another scenario is that some levels are in test but not in training. For them, just drop them from the test set.



## 6. Build and Evaluate Classifier

*   Input Directory: `../processed_data/`
*   Output Directory: `../log/`
*   Code Script: [train.py](https://github.com/KunyuHe/ML-Pipeline-on-Financial-Distress-Data/blob/master/train.py)



### 6.1 Benchmark: Default Decision Tree

For the training and evaluation part, I built the benchmark with a default *scikit-learn* `DecisionTreeClassifier`. Test metrics of the default decision tree is reported below:

| Metrics                                  | 2012.02 - 2012.07 | 2012.08 - 2013.01 | 2013.02 - 2013.07 |
| ---------------------------------------- | ----------------- | ----------------- | ----------------- |
| Accracy                                  | 0.6012            | 0.6166            | 0.6105            |
| Precision *( at decision threshold 0.5)* | 0.7031            | 0.7660            | 0.7104            |
| Recall *( at decision threshold 0.5)*    | 0.7170            | 0.6960            | 0.7213            |
| F-1 Score *( at decision threshold 0.5)* | 0.7100            | 0.7293            | 0.7518            |
| AUC ROC                                  | 0.5355            | 0.5241            | 0.5482            |

*(Benchmark Test Performances Across Test Periods)*



It seems that although for the first split our training set spans only one month, test performance of the benchmark model is not bad at all. Pattern in the data seems to be quite consistent.



### 6.2 Model Tuning

#### 6.2.1 Find the Best Decision Threshold

Users can build classifiers of their choices, and the first step after that, would be going through the automate process of finding the best decision threshold *(a test observation would be labeled `"positive"` when its predicted probability exceeds the decision threshold in a binary classification case)* within a predefined grid of thresholds *(`THRESHOLDS` in the code script)*.

While searching for the best decision threshold, the classifier is built with default parameters. Users can change this behavior by modifying `DEFAULT_ARGS` in the code script.

With 10-fold cross validation by default, the classifier would return a predicted probability for each observation in the validation set. It then goes down the threshold grid and label observations in the validation set. A score *(depending on the metrics of choice)* would be calculated on each validation set, and averaged as the final score of the threshold. Finally, we pick the threshold with the highest cross validation score. The process is illustrated below.



### 6.3 Model Evaluations

After finding the best set of hyper-parameters of a specific model according to a specific metrics on the training set, the so far "best" model would be validated on the test set. Performances of all the "best" models on any of the three test sets would be recorded in `CSV` format under `../log/evaluations/` directory. There are three folders under the directory, each stores the `performances.csv` table recording performance metrics listed above. Take `test set 1`, which spans Aug. 2012 - Jan. 2013 as an example.



##### Accuracy

In terms of accuracy, the best models on `test set 1` are listed below:

| Type                | Threshold | Hyperparameters                                      | Default Parameters                     | Accuracy | Precision | Recall | F1 Score | AUC ROC Score |
| ------------------- | --------- | ---------------------------------------------------- | -------------------------------------- | -------- | --------- | ------ | -------- | ------------- |
| Logistic Regression | 0.02      | {'penalty': 'l1', 'solver': 'liblinear',   'C': 0.1} | {'random_state': 123}                  | 0.7421   | 0.7423    | 0.9990 | 0.8520   | 0.5007        |
| Linear SVM          | 0.45      | {'penalty': 'l2', 'C': 0.01}                         | {'random_state': 123, 'max_iter': 200} | 0.7421   | 0.7421    | 1.0000 | 0.8520   | 0.5001        |

*(Best Models on Test Set 1 -- Accuracy)*

![](http://www.sciweavers.org/upload/Tex2Img_1557358794/eqn.png)

The two models have the same accuracy. It means they can correctly labels projects that would get fully funded within 60 days of posting or not correctly about 74.2% of the time, and they are as good as the other in terms of accuracy. The other performances metrics are nearly the same, too. In order to decide which one is better, we need to look further into their precision against percentage of population.

![](https://github.com/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/blob/master/log/images/best-models/(Linear%20SVM%20--%20Accuracy).png)

![](https://github.com/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/blob/master/log/images/best-models/(Logistic%20Regression%20--%20Accuracy).png)

Both models beat the baseline, which is the fraction of fully funded projects with a collecting time less than 60 days in the test set, 74.2%. This actually means that on a sample that is large enough, **these models are not much better than randomly guessing** in terms of accuracy. This is disappointing.

However, the bright side of these models are that they both beat the benchmark in terms of precision across proportions of population nearly all the time. **This means that as long as our resources are limited, these models can help identify the projects that are at risk of failing to get fully funded within 60 days of posting with higher precision.**



**Precision**

| Type       | Threshold | Hyperparameters                          | Default Parameters                                           | Accuracy | Precision | Recall   | F1 Score | AUC ROC Score |
| ---------- | --------- | ---------------------------------------- | ------------------------------------------------------------ | -------- | --------- | -------- | -------- | ------------- |
| Bagging    | 0.98      | {'max_samples': 0.5, 'max_features': 39} | {'n_estimators': 50, 'oob_score': True,   'n_jobs': -1, 'random_state': 123} | 0.259072 | 1         | 0.001531 | 0.003057 | 0.500765      |
| Linear SVM | 0.99      | {'penalty': 'l2', 'C': 0.001}            | {'random_state': 123, 'max_iter': 200}                       | 0.257996 | 1         | 8.06E-05 | 0.000161 | 0.50004       |

*(Best Models on Test Set 1 -- Precision)*



![](http://www.sciweavers.org/upload/Tex2Img_1557359319/eqn.png)

Precision measures that among all the projects that we predicted would get fully funded within 60 days of posting, the fraction of those who actually meets the criterion. Hence, as we raise the decision threshold and become really selective to label any observation as `"positive"`, we would get high precision as we might only categorize one or two to be `"positive"`. **In practice, both models are useless, unless we can only help one or two crowdfunding projects** so no further analysis is needed.



**Recall**

![](http://www.sciweavers.org/upload/Tex2Img_1557361824/eqn.png)

Recall measures that among all the projects that actually got fully funded within 60 days of posting, the fraction of those that our model predicts to get fully funded. This measures the ability of our model to cover all the `"positive"` observations.

For recall, the story is the same. As long as we **do not have unlimited resources to help all the crowdfunding projects posted**, which is obviously the case in practice, **models with high recall, even whose recall is 1, is useless**.



## 7. Policy Recommendation

### 7.1 Recommended Model

As stated above, in practice we cannot intervene with all the posted projects and help them succeed within 60 days. As long as our resources are limited, we always need to find out the projects that are at highest risks to fail with high precision. This makes a lot of models useful again, even including those who can not beat a random guess model on a infinitely large sample.

So, for example, if I were to make recommendations to someone who's working on this model to identify 5% of posted projects that are at highest risk, I would consider the following models *(there are many models with precision equals to 1 at 5% and here I only keep the ones with either highest AUC ROC, F-1 score or accuracy)*:

| Model               | Threshold | Hyperparameters                                        | Default Parameters                                           | Accuracy | Precision | Recall | F1 Score | AUC ROC Score | p_at_0.01 | p_at_0.02 | p_at_0.05 |
| ------------------- | --------- | ------------------------------------------------------ | ------------------------------------------------------------ | -------- | --------- | ------ | -------- | ------------- | --------- | --------- | --------- |
| Random Forest       | 0.7       | {'max_features': 15}                                   | {'n_estimators': 1000, 'oob_score': True,   'n_jobs': -1, 'random_state': 123} | 0.5762   | 0.833146  | 0.5362 | 0.6525   | 0.6136        | 1         | 1         | 1         |
| Boosting            | 0.01      | {'algorithm': 'SAMME.R', 'learning_rate':   0.001}     | {'n_estimators': 100, 'random_state': 123}                   | 0.7421   | 0.7421    | 1      | 0.8519   | 0.5           | 1         | 1         | 1         |
| Logistic Regression | 0.01      | {'penalty': 'l1', 'solver': 'liblinear',   'C': 0.001} | {'random_state': 123}                                        | 0.7421   | 0.7421    | 1      | 0.8519   | 0.5           | 1         | 1         | 1         |

*(Best Models on Test Set 1 -- Precision at 5%)*



Three models all have perfect precision at 5%, which means they can perfectly detect the 5% of posted projects that are at highest risk to fail to get fully funded in 60 days. However, among these models I would recommend Random Forest. Both Boosting and Logistic Regression use a decision threshold of 0.01, which means they are extremely vulnerable to changes in the distribution of predicted probabilities of future observations. Random Forest is robust to changes in data, and the training process can be expedited with parallel computing.  Further, it has lower variance compared to the two, indicated by high AUC ROC score. Visualize its metrics.



### 7.2 Recommended Model Performances on Test Set One

![](https://github.com/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/blob/master/log/images/best-models/dt(Random%20Forest%20--%20AUC%20ROC%20Score).png)

As we can see, distribution of the predicted probabilities are pretty smooth, with a slight left skew.

![](https://github.com/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/blob/master/log/images/best-models/pr(Random%20Forest%20--%20AUC%20ROC%20Score).png)

The recommended model always beats the benchmark in terms of precision, which means it's always better than a model that random guesses in terms of efficiency.

![](https://github.com/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/blob/master/log/images/best-models/auc(Random%20Forest%20--%20AUC%20ROC%20Score).png)

As expected, the recommended model has decent AUC and an acceptable ROC curve.

![](https://github.com/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/blob/master/log/images/best-models/fi(Random%20Forest%20--%20AUC%20ROC%20Score).png)

As indicated by the recommended model, the 5 most important features, in descending order, are:

- `total_price_including_optional_support` 
- `school_latitude`
- `students_reached`
- `grade_level`
- `poverty_level`



### 7.3 Recommended Model Test Performances

- Input Directory: `../processed_data/
- Code Script: [recommended_model.py](https://github.com/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/blob/master/codes/recommended_model.py)



Check test performances of our recommended model over time in `test set 0` and `test set 2` by running script `recommended_model.py`.

| Type              | Threshold | Default Parameters                                           | Accuracy | Precision | Recall | F1 Score | AUC ROC Score | p_at_0.01 | p_at_0.02 | p_at_0.05 |
| ----------------- | --------- | ------------------------------------------------------------ | -------- | --------- | ------ | -------- | ------------- | --------- | --------- | --------- |
| 2012.02 - 2012.07 | 0.7       | {'n_estimators': 1000,'max_features': 15, oob_score': True, 'n_jobs': -1, 'random_state': 123} | 0.5811   | 0.7375    | 0.5973 | 0.6601   | 0.5719        | 1         | 1         | 1         |
| 2012.08 - 2013.01 | 0.7       | {'n_estimators': 1000,'max_features': 15, oob_score': True, 'n_jobs': -1, 'random_state': 123} | 0.5762   | 0.8331    | 0.5362 | 0.6525   | 0.6136        | 1         | 1         | 1         |
| 2013.02 - 2013.07 | 0.7       | {'n_estimators': 1000,'max_features': 15, oob_score': True, 'n_jobs': -1, 'random_state': 123} | 0.6069   | 0.7752    | 0.5941 | 0.6727   | 0.6140        | 1         | 1         | 1         |

*(Recommended Model on All Test Sets)*



We can see that our **recommended model keeps a perfect precision at 5% level across all test sets**. In other words, it can perfectly detect the 5% of posted projects that are at highest risk to fail to get fully funded in 60 days in the past.

In terms of other metrics, we notice that AUC ROC score increases as training set grows larger. This means **our model is robust to change of data, and even benefits from an enlarged training set**.

