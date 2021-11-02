import pandas as pd
import numpy as np

def category_coverage(df, data_col, categories):
    unique_cat =  df[data_col].unique().size
    return [float(unique_cat/categories)]


def mean_max_ratio(df, data_col):
    y = df.groupby(data_col).agg({data_col: ['count']}).values.T[0]
    m = np.max(y)
    return [np.mean(y)/m, np.median(y)/m]


def ratio_of_paid_apps(df, data_col, paid_str):
    df0 = df.col[df[data_col] == paid_str]
    return float(len(df0)/len(df))

