# -*- coding:utf-8 -*-
import math
import numpy as np
import tensorflow as tf
from sklearn import metrics
from sklearn.utils import shuffle

import pymrmr

from sklearn.svm import SVR
from sklearn.feature_selection import RFE

from skrebate import ReliefF

from gcforest.gcforest import GCForest


def load_json(path):
    import json
    """
    """
    lines = []
    with open(path) as f:
        for row in f.readlines():
            if row.strip().startswith("//"):
                continue
            lines.append(row)
    return json.loads("\n".join(lines))

def consistency_index(sel1, sel2, num_features):
    """ Compute the consistency index between two sets of features.
    Parameters
    ----------
    sel1: set
        First set of indices of selected features
    sel2: set
        Second set of indices of selected features
    num_features: int
        Total number of features
    Returns
    -------
    cidx: float
        Consistency index between the two sets.
    Reference
    ---------
    Kuncheva, L.I. (2007). A Stability Index for Feature Selection.
    AIAC, pp. 390--395.
    """
    observed = float(len(sel1.intersection(sel2)))
    expected = len(sel1) * len(sel2) / float(num_features)
    maxposbl = float(min(len(sel1), len(sel2)))
    cidx = -1.
    # It's 0 and not 1 as expected if num_features == len(sel1) == len(sel2) => observed = n
    # Because "take everything" and "take nothing" are trivial solutions we don't want to select
    if expected != maxposbl:
        cidx = (observed - expected) / (maxposbl - expected)
    return cidx


def consistency_index_k(sel_list, num_features):
    """ Compute the consistency index between more than 2 sets of features.
    This is done by averaging over all pairwise consistency indices.
    Parameters
    ----------
    sel_list: list of lists
        List of k lists of indices of selected features
    num_features: int
        Total number of features
    Returns
    -------
    cidx: float
        Consistency index between the k sets.
    Reference
    ---------
    Kuncheva, L.I. (2007). A Stability Index for Feature Selection.
    AIAC, pp. 390--395.
    """
    cidx = 0.
    for k1, sel1 in enumerate(sel_list[:-1]):
        # sel_list[:-1] to not take into account the last list.
        # avoid a problem with sel_list[k1+1:] when k1 is the last element,
        # that give an empty list overwise
        # the work is done at the second to last element anyway
        for sel2 in sel_list[k1+1:]:
            cidx += consistency_index(set(sel1), set(sel2), num_features)
    cidx = 2.  * cidx / (len(sel_list) * (len(sel_list) - 1))
    return "{0:.4f}".format(cidx)


def avg_importance(sa, sb):
    sc = sa.add(sb, fill_value=None).dropna() / 2
    sd = sa.add(sb, fill_value=0).drop(sc.index)
    return sc.append(sd)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def cnn(x_train, x_test, y_train, y_test):
    L1 = 32  # number of convolutions for first layer
    L2 = 64  # number of convolutions for second layer
    L3 = 512  # number of neurons for dense layer
    learning_date = 1e-4  # learning rate
    epochs = 50  # number of times we loop through training data
    batch_size = 16  # number of data per batch
    display_step = 1

    loss_rec = np.zeros([epochs, 1])
    training_eval = np.zeros([epochs, 2])

    features = x_train.shape[1]
    classes = y_train.shape[1]

    xs = tf.placeholder(tf.float32, [None, features])
    ys = tf.placeholder(tf.float32, [None, classes])
    keep_prob = tf.placeholder(tf.float32)
    x_shape = tf.reshape(xs, [-1, 1, features, 1])

    # first conv
    w_conv1 = weight_variable([5, 5, 1, L1])
    b_conv1 = bias_variable([L1])
    h_conv1 = tf.nn.relu(conv2d(x_shape, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second conv
    w_conv2 = weight_variable([5, 5, L1, L2])
    b_conv2 = bias_variable([L2])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    tmp_shape = (int)(math.ceil(features / 4.0))
    h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * tmp_shape * L2])

    # third dense layer,full connected
    w_fc1 = weight_variable([1 * tmp_shape * L2, L3])
    b_fc1 = bias_variable([L3])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # fourth layer, output
    w_fc2 = weight_variable([L3, classes])
    b_fc2 = bias_variable([classes])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

    cost = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(y_conv), reduction_indices=[1]))
    optimizer = tf.train.AdamOptimizer(learning_date).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_batch = int(np.shape(x_train)[0] / batch_size)
        for epoch in range(epochs):
            avg_cost = 0.
            x_tmp, y_tmp = shuffle(x_train, y_train)
            for i in range(total_batch - 1):
                batch_x, batch_y = x_tmp[i * batch_size:i * batch_size + batch_size], \
                                   y_tmp[i * batch_size:i * batch_size + batch_size]
                _, c, acc = sess.run([optimizer, cost, accuracy],
                                     feed_dict={xs: batch_x, ys: batch_y, keep_prob: 0.8})
                avg_cost += c / total_batch

            del x_tmp
            del y_tmp

            ## Display logs per epoch step
            if epoch % display_step == 0:
                loss_rec[epoch] = avg_cost
                acc, y_s = sess.run([accuracy, y_conv],
                                    feed_dict={xs: x_train, ys: y_train, keep_prob: 1})
                auc = metrics.roc_auc_score(y_train, y_s)
                training_eval[epoch] = [acc, auc]
                print("Epoch:", '%d' % (epoch + 1), "cost =", "{:.9f}".format(avg_cost),
                      "Training accuracy:", round(acc, 3), " Training auc:", round(auc, 3))

        y_pred = y_conv.eval(feed_dict={xs: x_test, ys: y_test, keep_prob: 1.0})[:, 1]

        return y_pred


def mRMR(x_train, y_train, n_features):
    x_train.insert(loc=0, column='class', value=y_train)
    features = pymrmr.mRMR(x_train, 'MIQ', n_features)

    column_name = x_train.columns.tolist()
    results = []
    for feature_index in features:
        idx = column_name.index(feature_index)
        results.append(idx)

    return results


def svm_rfe(x_train, y_train, n_features):
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, n_features_to_select=n_features)
    selector = selector.fit(x_train.values, y_train)
    column_name = x_train.columns.tolist()

    features = []
    for feature_name, feature_ind in zip(column_name, selector.ranking_):
        if feature_ind == 1:
            features.append(column_name.index(feature_name))

    return features


def reliefF(x_train, y_train, n_features):
    fs = ReliefF(n_features_to_select=n_features)
    fs.fit(x_train.values, y_train)

    return list(fs.top_features_)[:n_features]


def df(x_train, y_train, n_features):
    config = load_json("demo_ca.json")
    gc = GCForest(config)
    X_train = x_train.values.reshape(-1, 1, len(x_train.columns))

    _, _features = gc.fit_transform(X_train, y_train)
    _features = _features.sort_values(ascending=False)
    return _features.index.values.tolist()[:n_features]
