# PRC Challenge

## Table of Contents
1. [Goal](#Goal)

2. [Dataset](#Dataset)

3. [K-fold Evaluation](#K-fold-Evaluation)

4. [Setup instructions](#Setup-instructions)

5. [Adding original challenge data](#Adding-original-challenge-data)

6. [My Method of Work & Solution](#My-Method-of-Work--Solution)
   - [I. Data processing](#i-Data-processing)
   - [II. Summarizing trajectories & adding trajectory features](#ii-Summarizing-trajectories-&-adding-trajectory-features)
   - [III. Label encoding](#iii-Label-encoding)
   - [IV. Grid searching and tuning hyperparameters of xgboost](#iv-Grid-searching-and-tuning-hyperparameters-of-xgboost)
   - [V. Selecting most important features based on gain](#v-Selecting-most-important-features-based-on-gain)
   - [VI. Choosing the number of n_estimators](#vi-Choosing-the-number-of-n_estimators)
   - [VII. Choosing the max_depth](#vii-Choosing-the-max_depth)
   - [VIII. Final tuning of hyperparameters](#viii-Final-tuning-of-hyperparameters)
   - [IX. Limiting of predicted aircraft_type based on min/max per aircraft type in training data](#ix-Limiting-of-predicted-aircraft_type-based-on-min/max-per-aircraft-type-in-training-data)
   - [X. Prediction and final submission](#x-Prediction-and-final-submission)

7. [Result Achieved](#Result-Achieved)

## Goal
Predicting the total takeoff weight of an aircraft with minimum Root Mean Squared Error RMSE and minimum percentage error.

## Dataset
- The Challenge publisher provided participants with 2 datasets (.csv files); one containing training data, and the other containing testing/submission data.
- Training data includes information features about 369013 flights along with their actual tow to use for training
- The testing or submission dataset includes the same information features about an additional 158149 flights but with their actual total takeoff weight hidden as it is the ground truth used for team ranking.
- In addition the trajectories features of these flights were also given in a .parquet shaped dataset

## K-fold Evaluation
- The idea behind k-fold evaluation is to split the training data into k equal sized parts. Then using one part at a time for testing and evaluation when using the other parts combined as training data. The final model score is then calculated as the mean of all k-fold model scores.
- The submission dataset of this challenge contains approximately 30% of the total number of challenge records “see calculation below” which means that a 3 fold validation is sensible strategy for evaluation. Since (1/3)*100 ≈ 33%.

$$
\left( \frac{\text{Number of submission records}}{\text{Number of submission records} + \text{Number of training records}} \right) \times 100 \approx 30\%
$$

$$
\left( \frac{158149}{369013 + 158149} \right) \times 100 \approx 30\%
$$

- An advantage of such approach in evaluation is that it gives a more accurate evaluation for what to expect when using the model on future unseen data.
- However, the disadvantage of k-fold validation is the need to train and test a model 3 times for each possible combination hyperparameters which can be consume extra computational power and running time than a random one-time test split. But the single k-fold model is trained with less data that makes building a single model easier than building a model with all the training data, keeping in mind that building 3 small model with less data for each is in practice more expensive than building one model with all the data.
- Accuracy is more preferred in this case, and therefore the k-fold validation technique is adopted for evaluation of proposed model and experiments.

## Setup instructions
In order to run the notebooks in this solution you can follow the step below

- Install Anaconda on your local machine
https://docs.anaconda.com/anaconda/install/

- Create a new conda environment with python installed
```
conda create --name prc python=3.12.4
```

- Activate the conda environment
```
conda activate prc
```

- Navigate to this root directory
```
cd '.\prc\'
```

- Install the required packages
```
pip install -r requirements.txt
```

- Run jupyter notebook
```
jupyter notebook
```

## Adding original challenge data
- You should place the challenge_set.csv the root folder ```/PRC_data``` and place the trajecotry .parquet files inside ```/PRC_data/trajectory_files```

## My Method of Work & Solution
* The flow of this solution is visualized in the diagram below and explained further under this diagram:
![PRC Challenge Solution Flow](/documentation/PRC_Challenge_Solution_Flow.png)

### I. Data processing
* To start with the data is viewed and a decision was made to split the date column into 3 feature columns (month_day, month and day) because the year is always 2022 is the data and thus it is non-informativ<br/>
* The same is also applied to the actual_offblock_time by creating 6 other feature columns (_hour_minute, _hour, _minute, _hour_minute, _hour, _minute)
* The dataset is them checked for any Nan values in any cells, and stored into "./data/processed_challenge_set.csv"
* The same processing steps above are also applied to the submission dataset and the results are stored into "./data/processed_final_submission_set.csv"
* The code is to be found in the notebooks 
    *   01_process_challenge_set.ipynb
    *   02_process_submission_set.ipynb

### II. Summarizing trajectories & adding trajectory features
The trajectory dataset is very large and contains detailed information about the trajectory of each flight in the challenge set. My solution was to summarize the trajectories by following the steps below
* Grouping trajectories by flight_id
* Calculating the 8 basic statistics of each trajecotry group; these are (count, mean, standard_deviation, minimum, 25percentile, median or 50percentile, 75percentile, max).
* Since we have 10 features in the trajectory dataset, using this technique above we get 10 features x 8 basic stasticts = 80 features in summary, with the advatange of reducing the number of trajectories 979680 rows by summarization.
* The summary dataset is then stored at "./data/summarized_trajectory.csv"
* These summary statstics are added to the challenge dataset by merging on the flight_id and stored into "./data/challenge_set_with_trajectories.csv" and "./data/final_submission_set_with_trajectories.csv" respectively.
* There was 8 rowws that did have a NaN value on some calculated statistics which a decided to replace with 0.
* The code is to be found in the notebooks 
    *   03_summarize_trajectories.ipynb
    *   04_add_trajectories_to_datasets.ipynb

### III. Label encoding
Because XGBoost and many other machine learning algorithm works better with numerical values, label encoding can be used to encode categorical features with numerical values.
* It is however important to encode both the training and testing data together so a categorical value (for example "A") would have the code 1 at both training and testing data, and not 1 in training data and 3 in the testing data.
* This is why both dataset are put together before encoding.
* After encoding the dataset are parted from each other and stored in "./data/encoded_challenge_set.csv" and "./data/encoded_final_submission_set.csv" respectively.
* The code is to be found in the notebook
    *   05_label_encode.ipynb

### IV. Grid searching and tuning hyperparameters of xgboost
Although multiple other machine learning algorithm was tried in several experiments, the XGBoost Regressor was chosen because it gave the best performance measured in RMSE. XGBoost is somehow similar to random forest in its nature but with inhanced optimization and better performance.
* Grid searching combined with 3-fold validation was used to build 3 models of each combination of the following values of hyperparameters in order to tune them for best performance


|       Parameter      |                           Purpose                                                      |       Values in Grid Search       |
|----------------------|----------------------------------------------------------------------------------------|-----------------------------------|
|    learning_rate     | Controls the contribution of each tree to the final prediction                         |        [0.01, 0.2, 1.0]           |
|     subsample        | Controls the fraction of training samples that are randomly selected to grow each tree |        [0.8, 0.9, 1.0]            |
|   colsample_bytree   | Specifies the fraction of features (or columns) to be randomly selected for each tree  |        [0.8, 0.9, 1.0]            |
|     max_depth        | Specifies the maximum depth of each decision tree in the ensemble                      |        [8, 9, 10]                 |
|   min_child_weight   | Control the minimum number of observations needed in a child node                      |        [8, 9, 10]                 |

* This grid search yields to the following result
```
Best parameters: {'colsample_bytree': 0.9, 'learning_rate': 0.2, 'max_depth': 10, 'min_child_weight': 10, 'subsample': 1.0}
Best RMSE: 2996.217350497871
```
* The range of parameter values in which you should grid search is derived from extensive trial and error to find a suitable seach space.
* The code is to be found in the notebook
    *   06_grid_search_xgboost.ipynb

### V. Selecting most important features based on gain
An advatage of XGBoost is that it scores each feature regarding how much it contributes to the RMSE score in the optimal model. Which we can use to further optimize our model as follows:
* Ordering features from top to least important according the the gain score from XGBoost
* Adding one n-top feature at a time, then building and evaluating a model with these n-features then keeping track of the score.
* At the end, we select the best model and use the top n features that gave the best RMSE.
* The result is that running the model with the top 49 most important features, gives best RMSE score of 2954
* The code is to be found in the notebook
    *   07_select_most_important_features.ipynb

### VI. Choosing the number of n_estimators
n_estimators are the number of smaller trees to be made under the training process in XGBoost. 
* A optimal value of n_estimators can be found from what is known as the elbow graph, where we pick the number of estimators that gives the RMSE score right before it converges to a minimum RMSE score for the model
* Grid search can be used here to search from the default 100 to 3000 estimators with the increase of 100 estimators at a time
* Then the elbow graph is plotted and the optimal number of estimators is found to be 850 estimators with an acheived RMSE that converages to 2871.
* The code is to be found in the notebook
    *   08_choosing_n_estimators.ipynb

### VII. Choosing the max_depth
max_depth is how far down an estimator tree is allowed to grow
* The same technique of elbow graph can also be used to find the optimal max_depth for this model by searching from 6 to 16 max_depth with the increase of 1. which gave an optimal performance at at max_depth of 9 with an RMSE score of 2865 
* The code is to be found in the notebook
    *   09_choosing_max_depth.ipynb

### VIII. Final tuning of hyperparameters
After finding the optimal max_depth and n_estimators, it is a good idea to re-run the tuning process of previously trained hyperparameters again to further improve the model performance or making sure the performance has not beed decreased after the choice of n_estimators and max_depth
* The code is to be found in the notebook
    *   10_final_grid_search_xgboost.ipynb
* The grid search yields to the following results for optimal performance
```

```

### IX. Limiting of predicted aircraft_type based on min/max per aircraft type in training data
As an experiment and since aircraft has their minimum and maximum take off weight limitation, i wanted to implement a measure after predicting that either increase the predicted value to the minimum threshold or decrease the predicted value to a maximum take off weight threshold.
* These thresholds are found by grouping the challenge training data by aircraft_type and then calculating the minimum and maximum for each aircraft_type.
* In addition to min and max per aircraft_type, different percentiles has also been tried from 10 to 100 with a step 10 to see if using thiese percentiles as an upper and lower bound gives better performance.
* The experiment and evaluation code is to be found in the notebook
    *   11_limit_predicted_tow_by_aircraft_type.ipynb

### X. Prediction and final submission
At the end, a final tuned model is built and used for making predictions of the submission dataset, then the submission .csv file is stored at ```/notebooks/submissions/{my_submission_v25.csv}```
* The code of this step is to be found in the notebook
    *   12_RUN_OPTIMAL_XGBOOST.ipynb
    
## Result Achieved
After submitting the predictions to the challenge ranking service, it got an RMSE score of ????????????? with an error rate of ?????????????