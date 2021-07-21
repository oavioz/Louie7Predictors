import numpy as np
import pandas as pd
import vector_creator.stats_models.estimators as est
from vector_creator.preprocess.utils import get_weekdays_by_loc


def sample_by_night_hours(df, sample_col ,data_col):
    df1 = df.set_index(sample_col)
    df2 = df1.between_time('20:00:00', '08:00:00')
    return df2[data_col]


def sample_by_weekend(df, sample_col, data_col, lat_long):
    weekend = get_weekdays_by_loc(lat_long[0], lat_long[1]) if lat_long != (0., 0.) else ['Saturday', 'Sunday']
    mask = pd.to_datetime((df[sample_col]).dt.date).dt.day_name().isin(weekend)
    y = df.loc[mask]
    return y[data_col]


def sample_by_day(df, sample_col, data_col):
    return df.groupby(pd.Grouper(key=sample_col, freq='D')).agg({data_col: ['count']})


class HuberM(object):
    def __init__(self, sample_col, data_col, lat_long_tuple):
        self.sample_col = sample_col
        self.data_col = data_col
        self.lat_long = lat_long_tuple

    def __call__(self, df, flag):
        if flag == 'night':
            hub = sample_by_night_hours(df, self.sample_col, self.data_col).to_numpy().T
        elif flag == 'weekend':
            hub = sample_by_weekend(df, self.sample_col, self.data_col, self.lat_long).to_numpy()
        else: # flag =  day
            hub = df[self.data_col].to_numpy()

        if hub.size == 0:
            return [float(), float()]
        return est.hober_m(hub.astype(np.float64))


class Qn(object):
    def __init__(self, sample_col, data_col, lat_long_tuple):
        self.sample_col = sample_col
        self.data_col = data_col
        self.lat_long = lat_long_tuple


    def __call__(self, df, flag='day'):
        if flag == 'night':
            q = sample_by_night_hours(df, self.sample_col, self.data_col).to_numpy().T
        elif flag == 'weekend':
            q = sample_by_weekend(df, self.sample_col, self.data_col, self.lat_long).to_numpy()
        else: # flag = day
            q = df[self.data_col].to_numpy()
        if q.size == 0:
            return [float()]
        return est.qn(q.astype(np.float64))

class Qn2(object):
    def __init__(self, sample_col, data_col, lat_long_tuple):
        self.sample_col = sample_col
        self.data_col = data_col
        self.lat_long = lat_long_tuple

    def __call__(self, df, flag='size'):
        if flag == 'type':
            df0 = sample_by_day(df, self.sample_col, self.data_col)
            q = df0[self.data_col].to_numpy().T[0]
        else: # flag = weekend
            weekend = get_weekdays_by_loc(self.lat_long[0], self.lat_long[1]) if self.lat_long != (0., 0.) else [
                'Saturday', 'Sunday']
            mask = pd.to_datetime((df[self.sample_col]).dt.date).dt.day_name().isin(weekend)
            df0 = df.loc[mask]
            if df0.size == 0:
                return [float()]
            q = sample_by_day(df0, self.sample_col, self.data_col).to_numpy().T[0]

        if q.size == 0:
            return [float()]
        return est.qn(q.astype(np.float64))


def huber_m_by_cat(h, df, cat_col, cat, flag='day'):
    df0 = df.loc[df[cat_col] == cat]
    if df0.empty:
        return [float()]
    return h(df0, flag)


def qn_by_cat(q_n, df, cat_col, cat, flag='day'):
    df0 = df.loc[df[cat_col] == cat]
    if df0.empty:
        return [float()]
    return q_n(df0, flag)

