import pandas as pd
import numpy as np
from vector_creator.preprocess import utils

'''
group-by Day,  filter by time frame 
return a numpy array of tuple(mean, std) for specific timeframe (day)
func in  [count , nunique]
'''
def daily_mean_std_func(df, sample_field, data_field, func):
    if df.empty:
        return np.zeros(2, dtype=float)
    np_list = df.groupby(pd.Grouper(key=sample_field, freq='D')).agg({data_field : [func]}).to_numpy()
    return [np.mean(np_list), np.std(np_list)]

# Mean and Std of continuous event (same event that happens one after the other)
def daily_mean_std_cont_event(df, sample_field, data_field):
    if df.empty:
        return np.zeros(2, dtype=float)
    ds = df.groupby(pd.Grouper(key=sample_field, freq='D')).apply(lambda x : x.pivot_table(index=[data_field], aggfunc='size'))
    np_list = ds.groupby(level=0).agg(np.mean).to_numpy()
    return [np.mean(np_list), np.std(np_list)]

#func in  [count , nunique]
def daily_mean_std_by_cat(df, sample_field, data_field, cat_field, cat, func):
    if df.empty:
        return np.zeros(2, dtype=float)
    df = df.loc[df[cat_field] == cat]
    if df.size == 0:
        return [float(0), float(0)]
    np_list = df.groupby(pd.Grouper(key=sample_field, freq='D')).agg({data_field : [func]}).to_numpy()
    return [np.mean(np_list), np.std(np_list)]

# Mean and Std of continuous event (same event that happens one after the other)
def daily_mean_std_cont_event_by_cat(df, sample_field, cat_field, cat, data_field):
    if df.empty:
        return np.zeros(2, dtype=float)
    df = df.loc[df[cat_field] == cat]
    ds = df.groupby(pd.Grouper(key=sample_field, freq='D')).apply(lambda x: x.pivot_table(index=[data_field], aggfunc='size'))
    np_list = ds.groupby(level=0).agg(np.mean).to_numpy()
    return [np.mean(np_list), np.std(np_list)]



class NightHours(object):
    def __init__(self, sample_col, data_col):
        #self.df = utils.filterDayHoursActivities(df, sample_col, '20:00:00', '08:00:00')
        self.sample_col = sample_col
        self.data_col = data_col


    def __call__(self, df,  func='count'):  # count , nunique
        df1 = df.set_index(self.sample_col)
        df2 = df1.between_time('20:00:00', '08:00:00')
        if df2.empty:
            return np.zeros(2, dtype=float)
        x = df2.groupby(pd.Grouper(freq='D')).agg({self.data_col: [func]})
        y =  x.to_numpy()
        return [np.mean(y.T), np.std(y.T)]


class NightHoursByCat(object):
    def __init__(self, sample_col, data_col, cat_col):
        self.sample_col = sample_col
        self.data_col = data_col
        self.cat_col = cat_col

    def __call__(self, df, cat, func='count'):  # count , nunique
        df = df.loc[df[self.cat_col] == cat]
        x = df.set_index(self.sample_col)
        y = x.between_time('20:00:00', '08:00:00')
        if y.empty:
            return np.zeros(2, dtype=float)
        y1 = y.groupby(pd.Grouper( freq='D')).agg({self.data_col : [func]}).to_numpy()
        return [np.mean(y1), np.std(y1)]


class WeekendHours(object):
    def __init__(self, df, datetime_col, data_col, long_lat_tuple):
        df0 = utils.create_column_day_of_week(df, datetime_col=datetime_col, col_name='day_of_week')
        df1 = utils.filter_by_weekends(df0, long_lat_tuple, 'day_of_week')
        self.datetime_col = datetime_col
        self.data_col = data_col
        self.df = df1

    def __call__(self, func='count'):
        if func in ['count', 'nunique']:
            return daily_mean_std_func(self.df, self.datetime_col, self.data_col, func)
        return daily_mean_std_cont_event(self.df, self.datetime_col, self.data_col)



def call_response_rate(df, data_col, cat):
    dn = df.groupby(data_col).agg({data_col: ['count']})
    incoming = dn.loc[cat[0] : ].iloc[0]
    missed = dn.loc[cat[2] : ].iloc[0]
    if float(incoming+missed) == 0:
        return [float()]
    return [float(incoming/(incoming+missed))]

def outgoing_answered_rate(df ,data_col, dur_col, cat):
    y0 = df.loc[df[data_col] == cat[1]]
    y = y0.groupby(data_col)[dur_col].apply(lambda x: x.astype(np.uint32)).to_numpy()
    ans = np.count_nonzero(y > 0)
    if y.size == 0:
        return [float()]
    return [float(ans/y.size)]


def mean_time_callback(df, number_col, date_time_col, status_col):
    df = df.loc[df[status_col] in ['MISSED', 'OUTGOING']]
    y = df.groupby(number_col)

