import multiprocessing
import os.path as osp

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy import unique
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .graph import Graph
from .ds_base import ds_base, get_dataset_base,generate_maps


def load_data():
    label_path = osp.abspath(osp.join(get_dataset_base(), "t2d", "labels.txt"))
    y = pd.read_csv(label_path, sep='\t', header=None)
    y = y.T.iloc[0]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    data_path = osp.abspath(osp.join(get_dataset_base(), "t2d", "count_matrix.csv"))
    X = pd.read_csv(data_path, sep=',', header=None)
    X = X.loc[(X != 0).any(axis=1)].as_matrix()
    # X = (X - X.min()) / (X.max() - X.min())

    train_idx, test_idx = train_test_split(range(len(X)), random_state=0, train_size=0.7, stratify=y)
    return (X[train_idx], y[train_idx]), (X[test_idx], y[test_idx])


def load_data_phy():
    my_x = []
    my_y = []

    data_path = osp.abspath(osp.join(get_dataset_base(), "t2d", "count_matrix.csv"))
    my_x = np.loadtxt(data_path, dtype=np.float32, delimiter=',')

    label_path = osp.abspath(osp.join(get_dataset_base(), "t2d", "labels.txt"))
    my_y = np.genfromtxt(label_path, dtype=np.str_, delimiter=',')

    otu_path = osp.abspath(osp.join(get_dataset_base(), "t2d", "otu.csv"))
    features = np.genfromtxt(otu_path, dtype=np.str_, delimiter=',')

    my_ref = pd.factorize(my_y)[1]
    label_reference_path = osp.abspath(osp.join(get_dataset_base(), "t2d", "label_reference.txt"))
    f = open(label_reference_path, 'w')
    f.write(str(my_ref))
    f.close()

    newick_path = osp.abspath(osp.join(get_dataset_base(), "t2d", "newick.txt"))
    g = Graph()
    g.build_graph(newick_path)

    my_data = pd.DataFrame(my_x)
    my_data = np.array(my_data)
    my_lab = pd.factorize(my_y)[0]
    my_maps = []
    my_benchmark = []

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(generate_maps)(x, g, features) for x in my_data)
    my_maps.append(np.array(np.take(results, 1, 1).tolist()))
    my_benchmark.append(np.array(np.take(results, 0, 1).tolist()))

    my_maps = np.array(my_maps)
    map_rows = my_maps.shape[2]
    map_cols = my_maps.shape[3]

    train_idx, test_idx = train_test_split(range(len(my_maps[0])), random_state=0, train_size=0.7, stratify=my_lab)
    x_train, y_train, x_test, y_test = my_maps[0][train_idx], my_lab[train_idx], my_maps[0][test_idx], my_lab[test_idx]

    x_train = np.array(x_train).reshape(-1, map_rows, map_cols)
    x_test = np.array(x_test).reshape(-1, map_rows, map_cols)
    y_train = np.squeeze(np.array(y_train).reshape(1, -1), 0)
    y_test = np.squeeze(np.array(y_test).reshape(1, -1), 0)

    return x_train, y_train, x_test, y_test


class T2D(ds_base):
    def __init__(self, **kwargs):
        super(T2D, self).__init__(**kwargs)
        (X_train, y_train), (X_test, y_test) = load_data()
        if self.data_set == "train":
            X = X_train
            y = y_train
        elif self.data_set == "test":
            X = X_test
            y = y_test
        elif self.data_set == "all":
            X = np.vstack((X_train, X_test))
            y = np.vstack((y_train, y_test))
        else:
            raise ValueError("T2D Unsupported data_set: ", self.data_set)

        X = X[:, np.newaxis, :, np.newaxis]
        X = self.init_layout_X(X)
        y = self.init_layout_y(y)
        self.X = X
        self.y = y
