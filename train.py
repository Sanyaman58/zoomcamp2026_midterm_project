import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score
import xgboost as xgb

def split_features_target(df, target):
    X = df.drop(columns=[target])
    y = df[target].values
    return X, y

# parameters

random_state = 1
eta = 0.1
max_depth=6
min_child_weight=5
output_file = f'xgb_model_depth_{max_depth}.bin'

# data preparation

df = pd.read_csv('star_classification.csv')

drop_columns = [
    'obj_ID',
    'run_ID',
    'rerun_ID',
    'cam_col',
    'field_ID',
    'spec_obj_ID',
    'plate',
    'MJD',
    'fiber_ID'
]

df = df.drop(columns=drop_columns, axis='columns')
df = df[df['g'] > -9999]
df['class'] = df['class'].replace({'GALAXY': 0, 'STAR': 1, 'QSO': 2}).astype(int)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=random_state)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=random_state
)

dfs = [df_train, df_val, df_test, df_full_train]
dfs = [d.reset_index(drop=True) for d in dfs]
df_train, df_val, df_test, df_full_train = dfs

target = 'class'

X_train, y_train = split_features_target(df_train, target)
X_val, y_val = split_features_target(df_val, target)
X_test, y_test = split_features_target(df_test, target)
X_full_train, y_full_train = split_features_target(
    df_full_train, target
)

features = list(X_train.columns)

dtrain = xgb.DMatrix(
    X_train,
    label=y_train,
    feature_names=features
)
dfulltrain =  xgb.DMatrix(
    X_full_train,
    label=y_full_train,
    feature_names=features
)
dval = xgb.DMatrix(
    X_val,
    label=y_val,
    feature_names=features
)
dtest = xgb.DMatrix(
    X_test,
    label=y_test,
    feature_names=features
)

xgb_params = {
    'eta': eta,
    'max_depth': max_depth,
    'min_child_weight': min_child_weight,
    
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'num_class': 3,
    
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

# training 

xgb_params = {
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight': 5,
    
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'num_class': 3,
    
    'nthread': 8,
    'seed': random_state,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=100, evals = [(dtrain, 'train'), (dtest, 'test')])
print("Model training finished successfully")

# saving the model

with open(output_file, 'wb') as f_out:
    pickle.dump((model), f_out)

print("Saved the model to %s file" % output_file)

