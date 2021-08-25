import pandas as pd
import numpy as np
from vector_creator.preprocess import utils
from vector_creator.stats_models.estimators import mad_calc


'''
group-by Day,  filter by time frame 
return a numpy array of tuple(mean, std) for specific timeframe (day)
func in  [count , nunique, f]
'''
def mean_std_func(df, sample_field, data_field, func, freq):
    r0 = [float(0), float(0), float(0)]
    r1 = [float(0), float(0)]
    if df.empty:
        return r1 if func == 'nunique' else [float(0)] + r0
    x = df.groupby(pd.Grouper(key=sample_field, freq=freq)).agg({data_field : [func]})
    y = x[data_field].values.T[0]
    nz = y[y > 0]
    nz0 = [np.mean(nz), np.std(nz), mad_calc(nz)] if len(nz) > 0 else r0
    nz1 = [np.mean(nz), np.std(nz)] if len(nz) > 0 else r1
    ratio = [float(len(nz)/len(y))] if len(y) > 0 else [float()]
    return nz1 if func == 'nunique' else ratio + nz0

# Mean and Std of continuous event (same event that happens one after the other)
def daily_mean_std_cont_event(df, sample_field, data_field):
    if df.empty:
        return [float(0), float(0)]
    ds = df.groupby(pd.Grouper(key=sample_field, freq='D')).apply(lambda x : x.pivot_table(index=[data_field], aggfunc='size'))
    np_list = ds.groupby(level=0).agg(np.mean).to_numpy()
    return [np.mean(np_list), np.std(np_list)]

#func in  [count , nunique, f]
def daily_mean_std_by_cat(df, sample_field, data_field, cat_field, cat, func):
    r0 = [float(0), float(0), float(0)]
    r1 = [float(0), float(0)]
    df = df.loc[df[cat_field] == cat]
    if df.size == 0:
        if func == 'count':
            return [float(0)] + r0
        return r1 if func == 'nunique' else r0  # duration
    x = df.groupby(pd.Grouper(key=sample_field, freq='D')).agg({data_field: [func]})
    y = x[data_field].values.T[0]
    nz = y[y > 0]
    nz0 = [np.mean(nz), np.std(nz), mad_calc(nz)] if len(nz) > 0 else r0
    nz1 = [np.mean(nz), np.std(nz)] if len(nz) > 0 else r1
    ratio = [float(len(nz) / len(y))] if len(y) > 0 else [float()]
    if func == 'count':
        return ratio + nz0
    return nz1 if func == 'nunique' else nz0


# Mean and Std of continuous event (same event that happens one after the other)
def daily_mean_std_cont_event_by_cat(df, sample_field, cat_field, cat, data_field):
    df = df.loc[df[cat_field] == cat]
    if df.empty:
        return [float(0), float(0)]
    ds = df.groupby(pd.Grouper(key=sample_field, freq='D')).apply(lambda x: x.pivot_table(index=[data_field], aggfunc='size'))
    np_list = ds.groupby(level=0).agg(np.mean).to_numpy()
    return [np.mean(np_list), np.std(np_list)]


#  func = 'count'
def col_stats_func(df, sample_field, cat_field, func, freq):
    y = df.groupby(pd.Grouper(key=sample_field, freq=freq)).agg({cat_field: [func]}).to_numpy().T[0]
    return [np.mean(y), np.std(y), np.min(y), np.max(y)] # + qn(y)


def col_delta_stats_func(df, sample_field):
    sec_in_day = 86400
    df['DELTA'] = df[sample_field].diff().apply(lambda x: x / np.timedelta64(1, 's')).fillna(0).astype('int64')
    df['DELTA'] = np.abs(df.DELTA)
    return [np.mean(df['DELTA']/sec_in_day),
            np.std(df['DELTA']/sec_in_day),
            np.min(df['DELTA']/sec_in_day),
            np.max(df['DELTA']/sec_in_day)]


def minmax_ratio_by_cat(df, cat_field, func='count'):
    z = df.groupby(cat_field).agg({cat_field: [func]}).to_numpy().T[0]
    min = float(np.min(z))
    max = float(np.max(z))
    r_min = min/len(df)
    r_max = max/len(df)
    return [r_min, r_max]


def minmax_by_cat_value(df, cat_field, val_field, val, func='count'):
    df0 = df.loc[df[val_field] == val]
    z = df0.groupby(cat_field).agg({val_field: [func]}).to_numpy()
    return [np.min(z), np.max(z)]


class NightHours(object):
    def __init__(self, sample_col):
        self.sample_col = sample_col


    def __call__(self, df, data_col,  func='count'):  # count , nunique
        df1 = df.set_index(self.sample_col)
        df2 = df1.between_time('20:00:00', '08:00:00')
        r0 = [float(0), float(0)]
        r1 = [float(0), float(0), float(0)]
        if df2.empty:
            return [float(0)] + r1 if func == 'count' else r0
        x = df2.groupby(pd.Grouper(freq='D')).agg({data_col: [func]})
        y = x[data_col].values.T[0]
        nz = y[y > 0]
        res = [np.mean(nz), np.std(nz), mad_calc(nz)] if len(nz) > 0 else r1
        res0 = [np.mean(nz), np.std(nz)] if len(nz) > 0 else r0
        ratio = [float(len(nz) / len(y))] if len(y) > 0 else [float()]
        return res0 if func == 'nunique' else ratio + res


class NightHoursByCat(object):
    def __init__(self, sample_col, cat_col):
        self.sample_col = sample_col
        self.cat_col = cat_col

    def __call__(self, df, data_col,  cat, func='count'):  # count , f
        df = df.loc[df[self.cat_col] == cat]
        x = df.set_index(self.sample_col)
        y = x.between_time('20:00:00', '08:00:00')
        r = [float(0), float(0), float(0)]
        if y.empty:
            return [float(0)] + r if func == 'count' else r
        y1 = y.groupby(pd.Grouper( freq='D')).agg({data_col : [func]})
        x = y1[data_col].values.T[0]
        nz = x[x > 0]
        res = [np.mean(nz), np.std(nz), mad_calc(nz)] if len(nz) > 0 else r
        ratio = [float(len(nz) / len(y))] if len(y) > 0 else [float()]
        return ratio + res if func == 'count' else res


class WeekendHours(object):
    def __init__(self, df, datetime_col, long_lat_tuple):
        df1 = utils.filter_by_weekends(df, long_lat_tuple, datetime_col, 'day_of_week')
        self.datetime_col = datetime_col
        self.df = df1

    def __call__(self, data_col, freq, func='count'):
        y = daily_mean_std_cont_event(self.df, self.datetime_col, data_col) if func == 'size' else mean_std_func(self.df, self.datetime_col, data_col, func, freq)
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

