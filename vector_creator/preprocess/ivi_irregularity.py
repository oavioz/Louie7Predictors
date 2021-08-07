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

    def __call__(self, flag, df):
        if flag == 'number':
            matrix = calc_ivi_cat_by_occurrences(df, self.datetime, self.number, self.tf,  self.tf2)
        elif flag == 'duration':
            matrix = self.calc_ivi_dur_by_number(df, self.datetime, self.dur, self.tf, self.tf2)
        elif flag == 'night':
            matrix = self.calc_ivi_by_hight_occurrences(df)
        else:
            matrix = self.calc_ivi_calls_time_delta(df, self.datetime, self.tf, self.tf2)
        return est.ivi_irregularity(matrix) if matrix.any() else [float(-1)]


    def calc_ivi_dur_by_number(self, df, datatime_col, dur_col, tf, tf2):
        df[dur_col] = df[dur_col].astype(np.uint32)
        y0 = df.groupby(pd.Grouper(key=datatime_col, freq=tf)).agg({dur_col: ['sum']})
        if len(y0) < 7:
            return [float(-1.0)]
        return create_ivi_matrix(y0.resample(tf2))


    def calc_ivi_calls_time_delta(self, df, datetime_col, tf, tf2):
        df['DIFF'] = df[datetime_col].diff().apply(lambda x: x / np.timedelta64(1, 's')).fillna(0).astype('int64')
        y0 = df.groupby(pd.Grouper(key=datetime_col, freq=tf)).agg({'DIFF': ['mean']}).fillna(0)
        if len(y0) < 7:
            return [float(-1.0)]
        return create_ivi_matrix(y0.resample(tf2))


    def calc_ivi_by_hight_occurrences(self,df):
        df1 = df.set_index(self.datetime)
        df2 = df1.between_time('20:00:00', '08:00:00')
        if df2.empty:
            return [float(-1.0)]
        y = df2.groupby(pd.Grouper(freq=self.tf)).agg({self.number: ['count']})
        if len(y) < 7:
            return [float(-1.0)]
        return create_ivi_matrix(y.resample(self.tf2))


def calc_ivi_number_by_cat(df, datetime_col, number_col, cat_col, cat, tf1, tf2):
    c_nan = df[cat_col].isnull().sum()
    if float(c_nan / len(df)) > 0.3:
        return [float(-1.0)]
    df = df.loc[df[cat_col] == cat]
    y0 = df.groupby(pd.Grouper(key=datetime_col, freq=tf1)).agg({number_col: ['count']})
    if len(y0) < 7:
        return [float(-1.0)]
    matrix =  create_ivi_matrix(y0.resample(tf2))
    return est.ivi_irregularity(matrix) if matrix.any() else [float(-1)]


''' GENERAL FUNCTIONS'''

def calc_ivi_cat_by_occurrences(df, datetime_col, data_col, tf, tf2):
    y0 = df.groupby(pd.Grouper(key=datetime_col, freq=tf)).agg({data_col: ['count']})
    if len(y0) < 7:
        return [float(-1.0)]
    return create_ivi_matrix(y0.resample(tf2))


def calc_ivi_time_delta(df, datetime_col, tf, tf2):
    df['DIFF'] = df[datetime_col].diff().apply(lambda x: x / np.timedelta64(1, 's')).fillna(0).astype('int64')
    y0 = df.groupby(pd.Grouper(key=datetime_col, freq=tf)).agg({'DIFF': ['mean']}).fillna(0)
    if len(y0) < 7:
        return [float(-1.0)]
    return create_ivi_matrix(y0.resample(tf2))


def create_ivi_matrix(y):
    y1 = y.apply(lambda x: x.to_numpy().T.flatten())
    if len(y1) < 4:
        return np.array([float(-1.0)])
    y2 = y1[1:-1].to_numpy().flatten()
    rows = len(y2)
    cols = y2[0].shape[0]
    return np.concatenate(y2).reshape(rows, cols)
