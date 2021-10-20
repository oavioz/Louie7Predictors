import vector_creator.stats_models.estimators as est
import numpy as np
import pandas as pd

def entropy_of_duration(df, dur_col, cat_col, cat):
    if cat != 'None':
        df = df.loc[df[cat_col] == cat]
    x =  df[dur_col]
    y = x[x != '0']
    y0 = 30 * np.round(y.astype(np.float64)/30, 0)
    return [est.calc_entropy(y0)]


def entropy_of_amount(df, date_col, cat_col):
    x = df.groupby(pd.Grouper(key=date_col, freq='D')).agg({cat_col: ['count']})
    y = x[cat_col].values.T[0]
    nz = y[y > 0]
    if not np.any(nz):
        return [float(-1)]
    return [est.calc_entropy(nz)]

# for call_logs, freq column is phone numbers
def entropy_of_freq(df, freq_col, cat_col, cat):
    df['HOUR'] = df[freq_col].dt.hour
    if cat != 'None':
        df = df.loc[df[cat_col] == cat]
    y = df.groupby('HOUR')['HOUR'].agg('count').to_numpy()
    z = y[y > 0]
    if len(z) == 0:
        return [float(-1)]
    return [est.calc_entropy(z)]


def entropy_of_number(df, num_col, cat_col, cat):
    if cat != 'None':
        df = df.loc[df[cat_col] == cat]
    y = df.groupby(num_col)[num_col].agg('count').to_numpy()
    return [est.calc_entropy(y)]


def entropy_of_cat(df, cat_col, categories, fetcher_group):
    def count_occurrence_by_cat(df0, c_col, cats):
        occurr = []
        for cat in cats:
            occurr.append(df0[df0[c_col] == cat].shape[0])
        return np.array(occurr)
    if fetcher_group == 'photo-gallery':
        df[cat_col] = df[cat_col].apply(lambda x : x.split('/')[1] if x == str(x) and len(x.split('/')) == 2 else str(x))
    y = count_occurrence_by_cat(df, cat_col, categories)
    z = y[y != 0]
    if len(z) == 0:
        return [float(-1)]
    return [est.calc_entropy(y)]
