# -*- coding:utf-8 -*-
import pandas as pd
from utils import avg_importance

from sklearn.model_selection import StratifiedKFold

import gcforest.data_load as load
from gcforest.gcforest import GCForest

import utils

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

config = utils.load_json("demo_ca.json")
gc = GCForest(config)

datasets = ['cirrhosis', 'obesity', 't2d']

for dataset_idx, name in enumerate(datasets):
    thre_features = {}
    X = None
    Y = None
    if name == 'cirrhosis':
        X, Y = load.cirrhosis_data()
    elif name == 't2d':
        X, Y = load.t2d_data()
    elif name == 'obesity':
        X, Y = load.obesity_data()
    else:
        raise Exception('the dataset is not defined!!!')

    output_features = pd.Series()
    for train, test in cv.split(X, Y):
        x_train = X.iloc[train]
        y_train = Y[train]

        x_test = X.iloc[test]
        y_test = Y[test]

        X_train = x_train.values.reshape(-1, 1, len(x_train.columns))
        X_test = x_test.values.reshape(-1, 1, len(x_test.columns))

        X_train_enc, _features = gc.fit_transform(X_train, y_train)

        probas_ = gc.predict_proba(X_test)
        output_features = avg_importance(output_features, _features)

    output_features = output_features.sort_values(ascending=False)
    columns = list(map(int, output_features.index.tolist()))
    output_features.index = X.columns[columns]

    output_features.to_csv("output/" + name)
