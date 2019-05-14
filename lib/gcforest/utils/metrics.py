# -*- coding:utf-8 -*-
import numpy as np

from .win_utils import win_vote, win_avg
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

def accuracy(y_true, y_pred):
    # fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # roc_auc = auc(fpr, tpr)
    # return roc_auc
    return 1.0 * np.sum(y_true == y_pred) / len(y_true)

def accuracy_pb(y_true, y_proba):
    y_true = y_true.reshape(-1)
    y_pred = np.argmax(y_proba.reshape((-1, y_proba.shape[-1])), 1)
    return 1.0 * np.sum(y_true == y_pred) / len(y_true)

    # y_true = y_true.reshape(-1)
    # y_proba = y_proba.reshape((-1, y_proba.shape[-1]))
    # fpr, tpr, thresholds = roc_curve(y_true, y_proba[:, 1])
    # roc_auc = auc(fpr, tpr)
    # return roc_auc

def accuracy_win_vote(y_true, y_proba):
    """
 
    
    Parameters
    ----------
    y_true: n x n_windows
    y_proba: n x n_windows x n_classes
    """
    n_classes = y_proba.shape[-1]
    y_pred = win_vote(np.argmax(y_proba, axis=2), n_classes)
    return accuracy(y_true[:,0], y_pred)

def accuracy_win_avg(y_true, y_proba):
    """
 
    
    Parameters
    ----------
    y_true: n x n_windows
    y_proba: n x n_windows x n_classes
    """
    y_pred = win_avg(y_proba)
    return accuracy(y_true[:,0], y_pred)
