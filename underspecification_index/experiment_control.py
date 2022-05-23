# --------------------------------------------------
# experiment_control.py
#
# Version 0
#
# Copyright (c) 2022 James Hinns
# This code is licensed under MIT license
# (see LICENSE for details)
#
# This file is used to calculate set underspecifcation indexes.
# Given a problem specefied by it's datasets, construct a Rashomon set
# where all predictors outperform a give threshold theta.
# Measure the variation between their explanations for each data instance
# in a given test set.
#
# Created 16/03/22 - adapted from previous work
#                    alongside Dr. Fan Xiuyi
#
# James Hinns, Fan Xiuyi
# Swansea University
# --------------------------------------------------

import pandas as pd
import numpy as np

import underspecification_index.index_calc as ic

import random

import time

import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


def compare_datasets(x_trns, y_trns, x_tst, y_tst, thetas,
                     metrics, model_count=100, tree_count=10,
                     save_name=None, regressor=False):
    """
    Compares the underspecification indexes of different datasets against the
    same testing dataset. Saves the local indexes for all instances in the
    testing set for each training dataset.

            Parameters:
                 trn_datasets - ()
                    :param var_path:
                    :param x_trns:
                    :param y_trns:
                    :param x_tst:
                    :param y_tst:
                    :param thetas:
                    :param metrics:
                    :param model_count:
                    :param tree_count:
                    :param file_path:
                    var_path

            Returns:
                dataset_indexes - dict:
                    Dictionary of dataset (int - key) to average set underspecification indexes (data - np.ndarray).
    """

    dataset_indexes = {}

    num_of_datasets = len(x_trns)

    # If saving file, create file
    if save_name is not None:
        out_path = f"{save_name}_Indexes.csv"
        var_path = f"{save_name}_Prediction_Variance.csv"

        cols = [list(metrics) + ["trn_length", "pred_var", "pred_acc"]]
        pd.DataFrame(columns=cols).to_csv(out_path, index=False, header="column_names")

    for i in range(num_of_datasets):

        print("Generating explanation matrix for dataset of",
              f"length {len(x_trns[i])} with theta {thetas[i]}")

        exps = ic.gen_exp_matrix(x_trns[i], y_trns[i], x_tst, y_tst, thetas[i],
                                 model_count, tree_count, var_path, regressor)

        print("Computing local indexes for dataset of length", len(x_trns[i]))

        indexes = ic.set_underspecification_index(exps, metrics)

        # Saving for experiment figures
        if save_name is not None:
            df = pd.DataFrame(indexes, columns=metrics)
            df["Dataset"] = [len(x_trns[i]) for j in range(len(x_tst))]

            df_in = pd.read_csv(var_path)

            df = pd.concat([df.reset_index(drop=True), df_in.reset_index(drop=True)], axis=1)

            safe_write(out_path, df)

        dataset_indexes[len(x_trns[i])] = np.mean(indexes, axis=0)

    return dataset_indexes


# Quick function that randomly waits if another thread
# is writing to csv, then tries again.
def safe_write(path, df):
    try:
        df.to_csv(path, mode='a', index=False, header=False)
    except PermissionError:
        wait = random.randint(1, 100) / 200
        print(f"PermissionError waiting {wait} seconds...")
        time.sleep(wait)
        safe_write(path, df)


# Testing function to check the average performance of sklearn random forests on given datasets
def check_average_performance(x_trns, y_trns, x_tst, y_tst, model_count=100, tree_count=10, regressor=False):
    num_of_datasets = len(x_trns)
    average_performances = np.zeros(len(x_trns))

    for i in range(num_of_datasets):

        for j in range(model_count):

            if regressor:
                f = RandomForestRegressor(n_estimators=tree_count)
            else:
                f = RandomForestClassifier(n_estimators=tree_count)

            f.fit(x_trns[i], y_trns[i])

            pred = f.predict(x_tst)

            if regressor:
                acc = np.mean(ic.calc_mse(pred, y_tst))
            else:
                acc = accuracy_score(pred, y_tst)

            average_performances[i] += acc / model_count

    # Fix for rounding errors over 1
    if not regressor:
        round_error_indexes = np.nonzero(average_performances > 1)
        average_performances[round_error_indexes] = 1

    return average_performances
