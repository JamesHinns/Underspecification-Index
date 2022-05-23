# --------------------------------------------------
# Copyright (c) 2022 James Hinns
# This code is licensed under MIT license
# (see LICENSE for details)
#
# This file is used to calculate set underspecification indexes.
# Given a problem specified by it's datasets, construct a Rashomon set
# where all predictors outperform a give threshold theta.
# Measure the variation between their explanations for each data instance
# in a given test set.
#
# Adapted from previous work alongside Dr. Fan Xiuyi.
# --------------------------------------------------

import numpy as np
from numpy import dot
from numpy.linalg import norm

import pandas as pd

import shap

from numba import njit, prange, typed

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split


@njit(fastmath=True, cache=True)
def kendalltau_numba_naive(a, b):
    """
    Numba optimised o(n^2) kendall correlation coefficent
    (tau-a) calculation implementation

            Parameters:
                 a,b - np.ndarray (1d.array):
                     Input arrays which will have their kendall
                     rank correlation calculated

            Returns:
                tau - float:
                    The Kendall correlation coefficent
    """

    a = np.argsort(a)
    b = np.argsort(b)

    n = len(a)

    # Calculate symmetric difference
    sym_dis = 0

    for i in range(1, n):
        for j in range(0, i):
            sym_dis += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])

    tau = (2 * sym_dis) / (n * (n - 1))

    return tau


# Calculate the average mean squared error between two arrays
@njit(fastmath=True)
def calc_mse(pred, y_tst):
    n = len(pred)

    mses = np.zeros(n)
    for i in range(n):
        mses[i] = np.square(pred[i] - y_tst[i])

    return mses


def gen_exp_matrix(x_trn, y_trn, x_tst, y_tst, theta, model_count=100,
                   tree_count=10, var_path=None, regressor=False):
    """
    Generates an explanation matrix of the emprical Rashomon set defined by
    the given datasets, theta, model_count and tree_count.

            Parameters:
                x_trn - np.ndarray (2d-array):
                    The numerical training features
                    used to train the predictors.
                    Each index should be all features
                    for a training instance.
                y_trn - np.ndarray (2d-array):
                    The numerical training prediction
                    targets, each index should be the
                    prediction target for the features
                    at the same index of x_trn.
                x_tst - np.ndarray (2d-array):
                    Testing equivalent of x_trn.
                y_tst - np.ndarray (2d-array):
                    Testing equivalent of y_trn.
                model_count - int, default 100:
                    The number of models to include
                    in the Rashomon set.
                tree_count - int, default 10:
                    The number of trees in each
                    random forest in the Rashomon set
                var_path - string:
                    The path to save variance and accuracy temporarily,
                    if None, disable saving.
                regressor - boolean:
                    Whether the predictors should be regressors
                    (true) or classifiers (false).

            Returns:
                shap_exps - np.ndarray (3d-array),
                shape(model_count,x_tst length,# of features):
                    Array of SHAP explanations for each test instance
                    in x_tst, for each predictor in the Rashomon set.
    """

    # Throw an exception if datasets aren't numpy arrays
    if not all(isinstance(i, np.ndarray) for i in [x_trn, y_trn, x_tst, y_tst]):
        raise TypeError("Datasets must be NumPy arrays")

    # Model counter
    i = 0

    # Local explanation for each datapoint for each predictor
    out_shape = (model_count, len(x_tst), len(x_tst[0]))
    shap_exps = np.zeros(out_shape)

    save_var = var_path is not None

    if save_var:
        preds = np.zeros((model_count, len(x_tst)))

        if regressor:
            accs = np.zeros((model_count, len(x_tst)))
        else:
            accs = np.zeros(model_count)

    while i < model_count:

        if regressor:
            pred_mses, pred_exps, pred = exp_regression(x_trn, y_trn, x_tst, y_tst, theta, tree_count)
        else:
            acc, pred_exps, pred = exp_classification(x_trn, y_trn, x_tst, y_tst, theta, tree_count)

        # If shap explanations computed, then pred exceeds theta threshold
        if pred_exps is not None:

            if save_var:
                preds[i] = pred

                if regressor:
                    for j in range(len(x_tst)):
                        accs[i][j] = pred_mses[j]
                else:
                    accs[i] = acc

            shap_exps[i] = pred_exps

            i += 1

    if save_var:
        variance = np.var(preds, axis=0)

        if regressor:
            acc_out = np.mean(accs, axis=0)
        else:
            avg_acc = np.mean(accs)
            acc_out = [avg_acc for _ in range(len(x_tst))]

        out_df = pd.DataFrame({"pred_var": variance, "pred_acc": acc_out})
        out_df.to_csv(var_path, index=False)

    # Transpose output so that contiguous arrays can be passed to local underspec index
    return np.transpose(shap_exps, (1, 0, 2))


# TODO tidy both sub function
# TODO add docstrings
def exp_classification(x_trn, y_trn, x_tst, y_tst, theta, tree_count):
    f = RandomForestClassifier(n_estimators=tree_count)

    f.fit(x_trn, y_trn)

    pred = f.predict(x_tst)

    acc = accuracy_score(pred, y_tst)

    if acc >= theta:
        shap_exps = np.array(shap.TreeExplainer(f).shap_values(x_tst)[0])
        return acc, shap_exps, pred
    else:
        return -1, None, None


def exp_regression(x_trn, y_trn, x_tst, y_tst, theta, tree_count):
    f = RandomForestRegressor(n_estimators=tree_count)

    f.fit(x_trn, y_trn)

    pred = f.predict(x_tst)

    pred_mses = calc_mse(pred, y_tst)

    if np.mean(pred_mses) <= theta:
        shap_exps = np.array(shap.TreeExplainer(f).shap_values(x_tst))
        return pred_mses, shap_exps, pred
    else:
        return pred_mses, None, None


@njit(fastmath=True, cache=True)
def calc_local_underspec_index(local_exps, metrics):
    """
    Calculates the local underspecification index using selected metrics,
    given all the explanations for a particular data instance for all
    predictors in a Rashomon set

            Parameters:
                local_exps - np.ndarray (2d-array):
                    List of the local explanations for a given data
                    instance of all predictors in a rashomon set.
                metrics - numba.typed.List:
                    List of metrics to be calculated.

            Returns:
                results - list:
                    The underspecification indexes using all metrics in
                    metrics.
    """

    # Output array, index for each metrics local index
    metric_n = len(metrics)
    results = np.zeros((metric_n))

    n = len(local_exps)
    # Inner trianglular matrix coefficent for mean
    coef = (2 / (n * (n - 1)))

    # Inner trinaglular matrix pairwise calculate metrics of local_exps
    for i in range(n):

        # First predictor of pairwise comparison
        f1_e = local_exps[i]

        for j in range(i + 1, n):

            # Second predictor of pairwise comparison
            f2_e = local_exps[j]

            # Calculate all requested metrics
            for m in range(metric_n):

                # Cosine Similarity
                if (metrics[m] == "c_sim"):
                    metric = np.dot(f1_e, f2_e) / (norm(f1_e) * norm(f2_e))

                # Euclidean Distance
                if (metrics[m] == "e_dis"):
                    metric = norm(f1_e - f2_e)

                # Pearson Correlation
                elif (metrics[m] == "p_cor"):
                    metric = np.corrcoef(f1_e, f2_e)[1, 0]

                # Kendall Correlation
                elif (metrics[m] == "k_tau"):
                    metric = kendalltau_numba_naive(f1_e, f2_e)

                results[m] += coef * metric

    return results


@njit(parallel=True, cache=True)
def set_underspecification_index(exps, metrics):
    """
    Calculate the underspecifcation indexes
    for a given set of data instances.

            Parameters:
                exps - np.ndarray (3d-array),
                shape(x_tst length,model_count,# of features):
                    Array of explanations for a set of test instances,
                    for a number of predictors.
                metrics - list:
                    List of metrics to be calculated.

            Returns:
                local_is - np.ndarray (2d-array):
                    The local underspecification index for each
                    metric in metrics, for each data instnace in
                    exps.
    """

    pred_n = len(exps)
    local_is = np.zeros((pred_n, len(metrics)))

    for i in prange(pred_n):
        local_is[i] = calc_local_underspec_index(exps[i], metrics)

    return local_is
