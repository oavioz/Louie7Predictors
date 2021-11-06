import pandas as pd
import numpy as np
from vector_creator.preprocess import utils
from vector_creator.stats_models.estimators import huber_est, mad_calc


'''
group-by Day,  filter by time frame 
return a numpy array of tuple(mean, std) for specific timeframe (day)
func in  [count , nunique, f]
'''
def daily_func(df, sample_field, data_field, func, freq):
    r0 = [float(0), float(0), float(0)]
    if df.empty:
        return r0
    x = df.groupby(pd.Grouper(key=sample_field, freq=freq)).agg({data_field : [func]})
    y = x[data_field].values.T[0]
    nz = y[y > 0]
    if len(nz) == 0:
        return r0
    return [np.median(nz), mad_calc(nz), float(len(nz)/len(y))]



def burst_func(df, sample_field, data_field, func, freq1, freq2):
    r0 = [float(0), float(0), float(0)]
    if df.empty:
        return r0
    x = df.groupby(pd.Grouper(key=sample_field, freq=freq1)).agg({data_field: [func]})
    y = x.loc[(x!=0).any(axis=1)]
    if len(y) == 0:
        return r0
    y.reset_index(level=0, inplace=True)
    y.columns = y.columns.get_level_values(0)
    z0 = y.loc[(y[data_field] >= 2)]
    z = z0.groupby(pd.Grouper(key=sample_field, freq=freq2)).agg({data_field: [func]}) if len(z0) > 0 else []
    mean_y = np.mean(y[data_field].values)
    median_z = np.median(z.values.T[0]) if len(z) > 0 else float(-1)
    mean_z = np.mean(z.values.T[0]) if len(z) > 0 else float(-1)
    return [mean_y, mean_z, median_z]


# Mean and Std of continuous event (same event that happens one after the other)
def daily_cont_event(df, sample_field, data_field):
    r0 = [float(0), float(0)]
    if df.empty:
        return r0
    ds = df.groupby(pd.Grouper(key=sample_field, freq='D')).apply(lambda x : x.pivot_table(index=[data_field], aggfunc='size'))
    y = ds.groupby(level=0).agg(np.mean)
    nz = y[y > 0]
    if len(nz)  == 0:
        return r0
    return [np.median(nz), mad_calc(nz)]


class DailyHours(object):
    def __init__(self, sample_col):
        self.sample_col = sample_col

    def __call__(self, df, data_col, start_time, stop_time, func='count'):  # count , nunique, f, size
        df1 = df.set_index(self.sample_col)
        df2 = df1.between_time(start_time, stop_time) # ('20:00:00', '08:00:00')
        y = self.cont_stats_func(df2, data_col) if func == 'size' else self.daily_stats_func(df2, data_col, func)
        return y

    @staticmethod
    def daily_stats_func(df, data_col, func):
        r0 = [float(0), float(0), float(0)]
        if df.empty:
            return r0
        x = df.groupby(pd.Grouper(freq='D')).agg({data_col: [func]})
        y = x[data_col].values.T[0]
        nz = y[y > 0]
        if not np.any(nz):
            return r0
        return [np.median(nz), mad_calc(nz), float(len(nz)/len(y))]

    @staticmethod
    def cont_stats_func(df, data_col):
        if df.empty:
            return [float(0), float(0)]
        ds = df.groupby(pd.Grouper(freq='D')).apply(lambda x: x.pivot_table(index=[data_col], aggfunc='size'))
        np_list = ds.groupby(level=0).agg(np.mean).to_numpy()
        return [np.median(np_list), mad_calc(np_list)]


class WeekDays(object):
    def __init__(self, df, datetime_col, long_lat_tuple):
        df1 = utils.filter_by_weekends(df, long_lat_tuple, datetime_col, 'day_of_week')
        df2 = utils.filter_by_workdays(df, long_lat_tuple, datetime_col, 'day_of_week')
        self.datetime_col = datetime_col
        self.df1 = df1
        self.df2 = df2

    def weekdays_count_func(self, df, data_col, func, freq):
        r0 = [float(0), float(0), float(0)]
        if df.empty:
            return r0
        x = df.groupby(pd.Grouper(key=self.datetime_col, freq=freq)).agg({data_col: [func]})
        y = x[data_col].values.T[0]
        nz = y[y > 0]
        if len(nz) == 0:
            return r0
        return [np.median(nz), mad_calc(nz), float(len(nz)/len(y))]


    def __call__(self, data_col, freq, flag, func='count'):
        if flag == 'weekend':
            df = self.df1
        else:
            df = self.df2
        y = daily_cont_event(df, self.datetime_col, data_col) if func == 'size' else self.weekdays_count_func(df, data_col, func, freq)
        return y

