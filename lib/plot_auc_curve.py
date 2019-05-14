# -*- coding:utf-8 -*-
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from scipy import interp
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import gcforest.data_load as load
from gcforest.gcforest import GCForest
from gcforest.utils.log_utils import get_logger


from sklearn.preprocessing import OneHotEncoder
import utils

LOGGER = get_logger('cascade_clf.lib.plot_roc_all')


def one_hot(integer_encoded):
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
    return one_hot_encoded


save_fig = True

# activate latex text rendering
rc('text', usetex=True)

f, ax = plt.subplots(3, 1, figsize=(15, 15))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

clf_svm = SVC(random_state=0, probability=True)

clf_knn = KNeighborsClassifier(n_neighbors=5)

clf_lg = LogisticRegression(random_state=0)

config = utils.load_json("demo_ca.json")
clf_gc = GCForest(config)

datasets = ['Cirrhosis', 'T2D', 'Obesity']
params = [(clf_lg, 'y', "LR"), (clf_svm, 'purple', 'SVM'),
          (clf_knn, 'blue', 'kNN'), ('CNN', 'green', 'CNN'), (clf_gc, 'red', 'DF')]

for dataset_idx, name in enumerate(datasets):
    X = None
    Y = None
    if dataset_idx == 0:
        X, Y = load.cirrhosis_data()
    elif dataset_idx == 1:
        X, Y = load.t2d_data()
    elif dataset_idx == 2:
        X, Y = load.obesity_data()
    else:
        raise Exception('the dataset is not defined!!!')

    gc_pred_acc = []

    for idx, x in enumerate(params):
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
        for train, test in cv.split(X, Y):
            if idx == 3:  ## CNN
                Y_trans = one_hot(Y)
                y_score = utils.cnn(X.iloc[train], X.iloc[test], Y_trans[train], Y_trans[test])
                fpr, tpr, thresholds = roc_curve(Y[test], y_score)
                v = interp(mean_fpr, fpr, tpr)
                tprs.append(v)
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
            elif idx == 4:
                gc = x[0]
                x_train, y_train, x_test, y_test = X.iloc[train], Y[train], X.iloc[test], Y[test]
                x_train = x_train.values.reshape(-1, 1, len(x_train.columns))
                x_test = x_test.values.reshape(-1, 1, len(x_test.columns))

                X_train_enc = gc.fit_transform(x_train, y_train)
                probas_ = gc.predict_proba(x_test)

                fpr, tpr, thresholds = roc_curve(Y[test], probas_[:, 1])
                v = interp(mean_fpr, fpr, tpr)
                tprs.append(v)
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
            else:
                x[0].fit(X.iloc[train], Y[train])
                probas_ = x[0].predict_proba(X.iloc[test])
                fpr, tpr, thresholds = roc_curve(Y[test], probas_[:, 1])
                v = interp(mean_fpr, fpr, tpr)
                tprs.append(v)
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        if idx == 4:
            label = '\\textbf{' + x[2] + '(AUC={:.3f})'.format(mean_auc) + '}'
            ax[dataset_idx].plot(mean_fpr, mean_tpr, color=x[1], label=label, lw=2, alpha=.8)
        else:
            label = '{}' '(AUC={:.3f})'.format(x[2], mean_auc)
            ax[dataset_idx].plot(mean_fpr, mean_tpr, color=x[1], label=label, lw=2, alpha=.8)

    ax[dataset_idx].plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=.8)
    ax[dataset_idx].set_title(name)
    ax[dataset_idx].legend(loc='lower right')
    ax[dataset_idx].set_ylabel('True Positive Rate')
    ax[dataset_idx].set_xlabel('False Positive Rate')
    ax[dataset_idx].set(adjustable='box-forced', aspect='equal')

if save_fig:
    plt.savefig('output/AUC.png', bbox_inches='tight')
    plt.close(f)
else:
    plt.show()
