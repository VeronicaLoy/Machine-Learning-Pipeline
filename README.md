# Configurable Machine Learning Pipeline

This machine learning pieline processes the market historical data set of real estate valuation from Sindian Dist,. New Taipei Cit, Taiwan ("https://aisgaiap.blob.core.windows.net/aiap4-assessment/real_estate.csv")  and feeds it into user-selected machine learning algorithm. The machine learning pieline is configured to enable different ways of processing the data, and allow easy experimentation of different algorithms and their parameters.

## Getting Started

### Installing

The requirements.txt file lists all Python libraries that the machine learning pipeline depends on and can be installed using:

```
pip install --user --requirement requirements.txt
```

## Running the tests

### Processing the data

**config.json** contains a list of column names (features) from the dataset. The weight of each feature can be specified by changing value that corresponds to each feature, e.g. set weight of 0.1 to feature "X1 transaction date" in the example below
```
,"Features": {

"X1 transaction date": 0.1
...
}
```
### Selecting the Model

**config.json** contains a list of **regression models** which users can select from. To select the model, specify **"Yes"** next to the desired model, e.g KNeighboursRegressor in the example below.

```
,"Model": {

"KNeighborsRegressor": {"Select?": "Yes"
...
}}

```

To specify the **parameters** associated with the selected model, users can, under their selected model, enter the list of parameters that they would like to experiment with, e.g. ,"parameters": **"{'n_neighbors':[2,10]}"**. The machine learning pieline uses **GridSearchCV** to search over the specified parameter values for the model. 

```
,"Model": {

"KNeighborsRegressor": {"Select?": "Yes"
,"parameters": "{'n_neighbors':[2,10]}"
}
```
The number of **cross-validation** folds can be specified under "cv", e.g. "cv": **5**

```
,"cv": 5}
```

## Deployment

In the terminal window, run script **run.sh**. This will run the code **ML.py**, which will output the selected features, model, parameters and number of cross-validation folds that the user has selected in the **config.json** file. It will also output the best parameters and best test score.

```
$ sh run.sh
```

Sample output from terminal window:
``````
(myenv) Loy:veronica_loy VeronicaLoy$ sh run.sh


Data successfully extracted from URL:
https://aisgaiap.blob.core.windows.net/aiap4-assessment/real_estate.csv 

Feature " X1 transaction date " is multiplied by 0.1
Feature " X2 house age " is multiplied by 0.2
Feature " X3 distance to the nearest MRT station " is multiplied by 2
Feature " X4 number of convenience stores " is multiplied by 1
Feature " X5 latitude " is multiplied by 0.3


Model selected: 
RandomForestRegressor 

Parameters selected: 
{'n_estimators':[50,100]} 

cv selected: 
5
Best Estimator: 
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
max_features='auto', max_leaf_nodes=None,
min_impurity_decrease=0.0, min_impurity_split=None,
min_samples_leaf=1, min_samples_split=2,
min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
oob_score=False, random_state=None, verbose=0, warm_start=False)

Best Parameters: 
{'n_estimators': 100}

Best Training Score: 
0.955448168034567

Best Test Score: 
0.7054746537389447


```
