import pandas as pd
import utils
import gcforest.data_load as load

from sklearn.model_selection import RepeatedStratifiedKFold

data_sets = ["cirrhosis", 't2d', 'obesity']
feature_sets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
feature_len = [542, 572, 465]
for k, data_name in enumerate(data_sets):
    X = None
    Y = None
    if k == 0:
        X, Y = load.cirrhosis_data()
    elif k == 1:
        X, Y = load.t2d_data()
    elif k == 2:
        X, Y = load.obesity_data()
    else:
        raise Exception('the dataset is not defined!!!')

    _df = pd.DataFrame(columns=['mRMR', 'svm_rfe', 'reliefF', 'df'])
    for feature_per in feature_sets:
        n_fearures = int(feature_len[k] * feature_per / 100.0)
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=0)

        mRMR_list = []
        svm_rfe_list = []
        reliefF_list = []
        df_list = []
        for train_index, test_index in rskf.split(X, Y):
            x_train, y_train, x_test, y_test = X.iloc[train_index], Y[train_index], X.iloc[test_index], Y[test_index]

            mRMR_list.append(utils.mRMR(x_train, y_train,n_fearures))
            svm_rfe_list.append(utils.svm_rfe(x_train, y_train, n_fearures))
            reliefF_list.append(utils.reliefF(x_train, y_train, n_fearures))
            df_list.append(utils.df(x_train, y_train, n_fearures))

        mRMR_KI = utils.consistency_index_k(mRMR_list, feature_len[k])
        svm_rfe_KI = utils.consistency_index_k(svm_rfe_list, feature_len[k])
        reliefF_KI = utils.consistency_index_k(reliefF_list, feature_len[k])
        df_KI = utils.consistency_index_k(df_list, feature_len[k])

        _df.loc[len(_df)] = mRMR_KI, svm_rfe_KI, reliefF_KI, df_KI

    _df.to_csv('output/' + data_name + '-KI.csv')
