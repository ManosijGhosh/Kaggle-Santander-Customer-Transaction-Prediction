# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

%matplotlib inline

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
train = pd.read_csv("Data/train.csv")
test = pd.read_csv("Data/test.csv")
test_df = test.head(20000)
train_df = train.head(20000)
# sklearn tools for model training and assesment
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, accuracy_score)
from sklearn.model_selection import GridSearchCV
test_ID = test_df['ID_code'].values
train_df = train_df.drop("ID_code",axis=1)
test_df = test_df.drop("ID_code",axis=1)
target_col = "target"
params = {
        "objective" : "binary",
        "metric" : "auc",
        "max_depth" : -1,
        "num_leaves" : 8,
        "min_data_in_leaf" : 25,
        "learning_rate" : 0.006,
        "bagging_fraction" : 0.2,
        "feature_fraction" : 0.4,
        "bagging_freq" : 1,
        "lambda_l1" : 5,
        "lambda_l2" : 5,
        "verbosity" : 1,
        "max_bin": 512,
        "num_threads" : 6
    }
# Create parameters to search
gridParams = {
    'learning_rate': [0.005],
    'n_estimators': [2000],
    'num_leaves': [4,2],
    'colsample_bytree' : [0.7,0.07],
    'reg_alpha' : [1,1.2,7],
    'reg_lambda' : [5,5.5,0.6],
    'subsample' : [0.7,0.07],
    'objective' : ['binary'],
    'boosting_type' : ['gbdt'],
}
#    'boosting_type' : ['gbdt'],
#    'objective' : ['binary'],
#    'random_state' : [501], # Updated from 'seed'
#    'colsample_bytree' : [0.65, 0.66],
#    'subsample' : [0.7,0.75],
#    'reg_alpha' : [1,1.2],
#    'reg_lambda' : [1,1.2,1.4],
#    }
train_y = train_df[target_col].values
features = [c for c in train_df.columns if c not in ['target']]
train_df = train_df.drop("target",axis=1)
train_X = train_df
test_X = test_df
lgtrain = lgb.Dataset(train_X, label=train_y)
#lgval = lgb.Dataset(val_X, label=val_y)
evals_result = {}
# Create classifier to use. Note that parameters have to be input manually
# not as a dict!
mdl = lgb.LGBMClassifier(boosting_type= 'gbdt',
          objective = 'binary',
          n_jobs = 6, # Updated from 'nthread'
          silent = True,
          max_depth = params['max_depth'],
          max_bin = params['max_bin'])
# Create the grid
grid = GridSearchCV(mdl, gridParams,scoring='roc_auc',
                    verbose=0,
                    cv=4,
                    n_jobs=-1)
# Run the grid
grid.fit(train_X, train_y)

print(grid.best_params_)
print(grid.best_score_)

# Using parameters already set above, replace in the best from the grid search
params['feature_fraction'] = grid.best_params_['colsample_bytree']
params['learning_rate'] = grid.best_params_['learning_rate']
# params['max_bin'] = grid.best_params_['max_bin']
params['num_leaves'] = grid.best_params_['num_leaves']
params['lambda_l1'] = grid.best_params_['reg_alpha']
params['lambda_l2'] = grid.best_params_['reg_lambda']
params['bagging_fraction'] = grid.best_params_['subsample']
numberOfEstimators = grid.best_params_['n_estimators']
# params['subsample_for_bin'] = grid.best_params_['subsample_for_bin']

def run_lgb(train_X, train_y, val_X, val_y, test_X,params):    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 300000, valid_sets=[lgtrain,lgval], early_stopping_rounds=500, verbose_eval=100, evals_result=evals_result)
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result

pred_test = 0
kf = model_selection.KFold(n_splits=6, random_state=2018, shuffle=True)
feature_importance_df = pd.DataFrame()
for dev_index, val_index in kf.split(train_df):
    dev_X, val_X = train_X.loc[dev_index,:], train_X.loc[val_index,:]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    
    pred_test_tmp, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X,params)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = model.feature_importance()
    #fold_importance_df["fold"] = dev_index + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
    pred_test += pred_test_tmp / 6
#pred_test = np.clip(pred_test / 6, 0.0, 1.0 )
cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,25))
sns.barplot(x="importance",
            y="feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')