import io
import requests
import json

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

import ast

import warnings
warnings.filterwarnings("ignore")

# Load config file
with open('config.json') as json_data_file:
    config = json.load(json_data_file)

# Extract data
URL = config["URL"]
s = requests.get(URL).content
df = pd.read_csv(io.StringIO(s.decode('utf-8')))
print("\n")
print('Data successfully extracted from URL:\n', URL, '\n')

# Data pre-processing / Feature engineering
columns = []
for key, value in config['Features'].items():
    if value != 0:
        columns.append(key)
        df[key] = df[key] * value
        print('Feature "', key, '" is multiplied by', value)

print("\n")
x = np.asarray(df[columns])
y = np.asarray(df['Y house price of unit area'])

# Algorithm selection / tuning
for key, value in config['Model'].items():
    if config['Model'][key]['Select?'] == "Yes":
        model = key
        parameters = config['Model'][key]['parameters']
        break

print('Model selected: \n', model, '\n')
print('Parameters selected: \n', parameters, '\n')
print('cv selected: \n', config['cv'])

function_mappings = {
    'KNeighborsRegressor': KNeighborsRegressor, 'SVR': SVR, 'LinearRegression': LinearRegression, 'DecisionTreeRegressor': DecisionTreeRegressor, 'RandomForestRegressor': RandomForestRegressor
}

model = function_mappings[model]()

params = ast.literal_eval(parameters)

gs = GridSearchCV(model, params, cv=config['cv'])
gs.fit(x, y)

print("Best Estimator: \n{}\n".format(gs.best_estimator_))
print("Best Parameters: \n{}\n".format(gs.best_params_))
print("Best Training Score: \n{}\n".format(gs.cv_results_['mean_train_score'][gs.best_index_]))
print("Best Test Score: \n{}\n".format(gs.best_score_))
