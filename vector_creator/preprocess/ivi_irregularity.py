import pandas as pd
import numpy as np
import vector_creator.stats_models.estimators as est

# args = [datetime_col, cat_col, tf, tf2]
class IVI:
    def __init__(self, *args):
        self.datetime = args[0]
        self.cat = args[1]
        self.tf = args[2]
        self.tf2 = args[3]

    def __call__(self, flag, df):
        if flag == 'occurr':
            matrix = calc_ivi_cat_by_occurrences(df, self.datetime, self.cat, self.tf, self.tf2)
        else:   # deltaT
            matrix = calc_ivi_time_delta(df, self.datetime, self.tf, self.tf2)
        return est.ivi_irregularity(matrix)


# args = [datetime_col, dur_col, num_col, cat_col, tf, tf2]
class IVI2:
    def __init__(self,  *args):
        self.datetime = args[0]
        self.dur = args[1]
        self.number = args[2]
        self.cat = args[3]
        self.tf = args[4]
        self.tf2 = args[5]

    def __call__(self, flag, df, cat='none'):
        if flag == 'number':
            matrix = calc_ivi_cat_by_occurrences(df, self.datetime, self.number, self.tf,  self.tf2)
        elif flag == 'duration':
            matrix = self.calc_ivi_dur_by_number(df, self.datetime, self.dur, self.tf, self.tf2)
        else:
            matrix = self.calc_ivi_calls_time_delta(df, self.datetime, self.cat, cat, self.tf, self.tf2)
        return est.ivi_irregularity(matrix)


    def calc_ivi_dur_by_number(self, df, datatime_col, dur_col, tf, tf2):
        df[dur_col] = df[dur_col].astype(np.uint32)
        y0 = df.groupby(pd.Grouper(key=datatime_col, freq=tf)).agg({dur_col: ['sum']})
        return create_ivi_matrix(y0, tf2)


    def calc_ivi_calls_time_delta(self, df, datetime_col, cat_col, cat, tf, tf2):
        df['DIFF'] = df[datetime_col].diff().apply(lambda x: x / np.timedelta64(1, 's')).fillna(0).astype('int64')
        df = df.loc[df[cat_col] == cat]
        y0 = df.groupby(pd.Grouper(key=datetime_col, freq=tf)).agg({'DIFF': ['mean']}).fillna(0)
        return create_ivi_matrix(y0, tf2)




''' GENERAL FUNCTIONS'''

def calc_ivi_cat_by_occurrences(df, datetime_col, cat_col, tf, tf2):
    y0 = df.groupby(pd.Grouper(key=datetime_col, freq=tf)).agg({cat_col: ['count']})
    return create_ivi_matrix(y0, tf2)


def calc_ivi_time_delta(df, datetime_col, tf, tf2):
    df['DIFF'] = df[datetime_col].diff().apply(lambda x: x / np.timedelta64(1, 's')).fillna(0).astype('int64')
    y0 = df.groupby(pd.Grouper(key=datetime_col, freq=tf)).agg({'DIFF': ['mean']}).fillna(0)
    return create_ivi_matrix(y0, tf2)


def create_ivi_matrix(y, tf):
    y1 = y.resample(tf).apply(lambda x: x.to_numpy().T.flatten())
    if y1.size < 3:
        return np.empty(1)
    y2 = y1[1:-1].to_numpy()
    rows = y2.shape[0]
    cols = y2[0].shape[0]
    return np.concatenate(y2).reshape(rows, cols)