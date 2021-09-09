import pandas as pd
import numpy as np
from vector_creator.preprocess import utils
from vector_creator.stats_models.estimators import hober_est, qn


'''
group-by Day,  filter by time frame 
return a numpy array of tuple(mean, std) for specific timeframe (day)
func in  [count , nunique, f]
'''
def daily_func(df, sample_field, data_field, func, freq):
    r0 = [float(0), float(0), float(0), float(0), float(0)]
    if df.empty:
        return r0
    x = df.groupby(pd.Grouper(key=sample_field, freq=freq)).agg({data_field : [func]})
    y = x[data_field].values.T[0]
    nz = y[y > 0]
    if len(nz) == 0:
        return r0
    r_est = [np.mean(nz), np.std(nz)]
    try:
        h_est_mean, h_est_std = hober_est(nz)
        r_est = [h_est_mean, h_est_std]
    except ValueError:
        print('huber est was not calculated')
    return [np.mean(nz), np.std(nz), r_est[0], r_est[1], len(nz)/len(y)]


def burst_func(df, sample_field, data_field, func, freq1, freq2):
    r0 = [float(0), float(0)]
    if df.empty:
        return r0
    x = df.groupby(pd.Grouper(key=sample_field, freq=freq1)).agg({data_field: [func]})
    y = x.loc[(x!=0).any(axis=1)]
    if len(y) == 0:
        return r0
    y.reset_index(level=0, inplace=True)
    y.columns = y.columns.get_level_values(0)
    z = y.groupby(pd.Grouper(key=sample_field, freq=freq2)).agg({data_field: [func]})
    mean_y = np.mean(y[data_field].values)
    mean_z = np.mean(z.values.T[0]) if len(z) > 0 else float(0)
    return [mean_y, mean_z]

# Mean and Std of continuous event (same event that happens one after the other)
def daily_cont_event(df, sample_field, data_field):
    r0 = [float(0), float(0), float(0), float(0)]
    if df.empty:
        return r0
    ds = df.groupby(pd.Grouper(key=sample_field, freq='D')).apply(lambda x : x.pivot_table(index=[data_field], aggfunc='size'))
    y = ds.groupby(level=0).agg(np.mean)
    nz = y[y > 0]
    if len(nz)  == 0:
        return r0
    r_est = [np.mean(nz), np.std(nz)]
    try:
        h_est_mean, h_est_std = hober_est(nz)
        r_est = [h_est_mean, h_est_std]
    except ValueError:
        print('huber est was not calculated')
    return [np.mean(nz.values), np.std(nz.values), r_est[0], r_est[1]]

#func in  [count , nunique, f]
def daily_func_by_cat(df, sample_field, data_field, cat_field, cat, func):
    r1 = [float(0), float(0)]
    df = df.loc[df[cat_field] == cat]
    if df.empty:
        return r1
    x = df.groupby(pd.Grouper(key=sample_field, freq='D')).agg({data_field: [func]})
    y = x[data_field].values.T[0]
    nz = y[y > 0]
    if not np.any(nz):
        return r1
    r_est = np.mean(nz)
    try:
        r_est = hober_est(nz)[0]
    except ValueError:
        print('huber est was not calculated')
    return [np.mean(nz), r_est]


# Mean and Std of continuous event (same event that happens one after the other)
def daily_cont_event_by_cat(df, sample_field, cat_field, cat, data_field):
    df = df.loc[df[cat_field] == cat]
    if df.empty:
        return [float(0), float(0)]
    ds = df.groupby(pd.Grouper(key=sample_field, freq='D')).apply(lambda x: x.pivot_table(index=[data_field], aggfunc='size'))
    if ds.empty:
        return [float(0), float(0)]
    np_list = ds.groupby(level=0).agg(np.mean).to_numpy()
    return [np.mean(np_list), np.std(np_list)]


def col_delta_stats_func(df, sample_field):
    sec_in_week = 86400*7
    df['DELTA'] = df[sample_field].diff().apply(lambda x: x / np.timedelta64(1, 's')).fillna(0).astype('int64')
    df['DELTA'] = np.abs(df.DELTA)
    hober_m, hober_s = hober_est(df['DELTA'], 500)
    return [np.mean(df['DELTA']/sec_in_week),
            np.std(df['DELTA']/sec_in_week),
            hober_m/sec_in_week, hober_s/sec_in_week]


def categories_ratio(df, cat_col, cat_len):
    y = df.groupby(cat_col).agg({cat_col: ['count']})
    used_cat = len(y)/cat_len
    max_cat_ratio = np.max(y.values) / np.sum(y.values)
    min_cat_ratio = np.min(y.values) / np.sum(y.values)
    return [used_cat, max_cat_ratio, min_cat_ratio]


class NightHours(object):
    def __init__(self, sample_col):
        self.sample_col = sample_col

    def __call__(self, df, data_col,  func='count'):  # count , nunique, f, size
        df1 = df.set_index(self.sample_col)
        df2 = df1.between_time('20:00:00', '08:00:00')
        y = self.cont_stats_func(df2, data_col) if func == 'size' else self.daily_stats_func(df2, data_col, func)
        return y

    def daily_stats_func(self, df, data_col, func):
        r0 = [float(0), float(0), float(0)]
        if df.empty:
            return r0
        x = df.groupby(pd.Grouper(freq='D')).agg({data_col: [func]})
        y = x[data_col].values.T[0]
        nz = y[y > 0]
        if not np.any(nz):
            return r0
        return [np.mean(nz), np.std(nz), len(nz)/len(y)]

    def cont_stats_func(self, df, data_col):
        if df.empty:
            return [float(0), float(0)]
        ds = df.groupby(pd.Grouper(freq='D')).apply(lambda x: x.pivot_table(index=[data_col], aggfunc='size'))
        np_list = ds.groupby(level=0).agg(np.mean).to_numpy()
        return [np.mean(np_list), np.std(np_list)]


class WeekendHours(object):
    def __init__(self, df, datetime_col, long_lat_tuple):
        df1 = utils.filter_by_weekends(df, long_lat_tuple, datetime_col, 'day_of_week')
        self.datetime_col = datetime_col
        self.df = df1

    def weekend_count_func(self, data_col, func, freq):
        r0 = [float(0), float(0), float(0)]
        if self.df.empty:
            return r0
        x = self.df.groupby(pd.Grouper(key=self.datetime_col, freq=freq)).agg({data_col: [func]})
        y = x[data_col].values.T[0]
        nz = y[y > 0]
        if len(nz) == 0:
            return r0
        return [np.mean(nz), np.std(nz), len(nz)/len(y)]

    def __call__(self, data_col, freq, func='count'):
        y = daily_cont_event(self.df, self.datetime_col, data_col) if func == 'size' else self.weekend_count_func(data_col, func, freq)
        return y


def call_response_rate(df, data_col, cat):
    dn = df.groupby(data_col).agg({data_col: ['count']})
    if len(dn) < 3:
        return [float()]
    incoming = dn.loc[cat[0] : ].iloc[0]
    missed = dn.loc[cat[2] : ].iloc[0]
    if float(incoming+missed) == 0:
        return [float()]
    return [float(incoming/(incoming+missed))]


def outgoing_answered_rate(df ,data_col, dur_col, cat):
    y0 = df.loc[df[data_col] == cat[1]]
    if y0.empty:
        return [float()]
    y = y0.groupby(data_col)[dur_col].apply(lambda x: x.astype(np.uint32)).to_numpy()
    if len(y) == 0:
        return [float()]
    ans = np.count_nonzero(y > 0)
    return [float(ans/len(y))]


def mean_time_callback(df, number_col, date_time_col, status_col):
    df = df.loc[df[status_col] in ['MISSED', 'OUTGOING']]
    y = df.groupby(number_col)

