from vector_creator.preprocess import utils
from vector_creator.stats_models.auto_regression import *
from vector_creator.stats_models.estimators import huber_est, mad_calc, calc_entropy



'''
group-by Day,  filter by time frame 
return a numpy array of tuple(mean, std) for specific timeframe (day)
func in  [count , nunique, f]
'''
def daily_func(df, sample_field, data_field, func, freq):
    r0 = [float(0), float(0), float(0), float(0)]
    if df.empty:
        return r0
    x = df.groupby(pd.Grouper(key=sample_field, freq=freq)).agg({data_field : [func]})
    y = x[data_field].values.T[0]
    nz = y[y > 0]
    if len(nz) == 0:
        return r0
    return [np.median(nz), mad_calc(nz), np.mean(nz), float(len(nz)/len(y))]



def burst_func(df, sample_field, data_field, func, freq1, freq2, filter_by_hours):
    r0 = [float(0), float(0)]
    if df.empty:
        return r0
    x = df.groupby(pd.Grouper(freq=freq1)).agg({data_field: [func]}) if filter_by_hours else df.groupby(pd.Grouper(key=sample_field, freq=freq1)).agg({data_field: [func]})
    y = x.loc[(x!=0).any(axis=1)]
    if len(y) == 0:
        return r0
    y.reset_index(level=0, inplace=True)
    y.columns = y.columns.get_level_values(0)
    z0 = y.loc[(y[data_field] >= 2)]
    z = z0.groupby(pd.Grouper(key=sample_field, freq=freq2)).agg({data_field: [func]}) if len(z0) > 0 else []
    mean_y = np.mean(y[data_field].values)
    mean_z = np.mean(z.values.T[0]) if len(z) > 0 else float(-1)
    return [mean_y, mean_z]



class DailyHours(object):
    def __init__(self, sample_col, freq):
        self.sample_col = sample_col
        self.freq = freq

    def __call__(self, df, data_col, start_time, stop_time, func='count'):  # count , nunique, f, size
        df1 = df.set_index(self.sample_col)
        df2 = df1.between_time(start_time, stop_time) # ('20:00:00', '08:00:00')
        y = self.cont_stats_func(df2, data_col) if func == 'size' else self.daily_stats_func(df2, data_col, func)
        z = burst_func(df2, sample_field=self.sample_col, data_field=data_col, func='count', freq1='20S', freq2='W', filter_by_hours=True)
        return y + z

    def daily_stats_func(self, df, data_col, func, t_size=3):
        r0 = [float(0), float(0), float(0), float(0)] #, float(-1), float(-1)]
        if df.empty:
            return r0
        x = df.groupby(pd.Grouper(freq=self.freq)).agg({data_col: [func]})
        y = x[data_col].values.T[0]
        nz = y[y > 0]
        if not np.any(nz):
            return r0
        #train, test = y[0:len(y) - t_size], y[len(y) - t_size:]
        #ar_lag_1 = ar(train=train, test=test, lag=1, mse=True)
        return [np.median(nz), mad_calc(nz), np.mean(nz), float(len(nz)/len(y))] #, calc_entropy(y), ar_lag_1]

    def cont_stats_func(self, df, data_col):
        if df.empty:
            return [float(0), float(0), float(0)]
        ds = df.groupby(pd.Grouper(freq=self.freq)).apply(lambda x: x.pivot_table(index=[data_col], aggfunc='size'))
        np_list = ds.groupby(level=0).agg(np.mean).to_numpy()
        return [np.median(np_list), mad_calc(np_list), np.mean(np_list)]


class WeekDays(object):
    def __init__(self, df, datetime_col, long_lat_tuple, freq):
        df1 = utils.filter_by_weekends(df, long_lat_tuple, datetime_col, 'day_of_week')
        df2 = utils.filter_by_workdays(df, long_lat_tuple, datetime_col, 'day_of_week')
        self.datetime_col = datetime_col
        self.freq = freq
        self.df1 = df1
        self.df2 = df2

    def weekdays_count_func(self, df, data_col, func, t_size=3):
        r0 = [float(0), float(0), float(0), float(0)]
        if df.empty:
            return r0
        x = df.groupby(pd.Grouper(key=self.datetime_col, freq=self.freq)).agg({data_col: [func]})
        y = x[data_col].values.T[0]
        nz = y[y > 0]
        if len(nz) == 0:
            return r0
        #train, test = y[0:len(y) - t_size], y[len(y) - t_size:]
        #ar_lag_1 = ar(train=train, test=test, lag=1, mse=True)
        return [np.median(nz), mad_calc(nz), np.mean(nz), float(len(nz)/len(y))] #, calc_entropy(y), ar_lag_1]

    # Mean and Std of continuous event (same event that happens one after the other)
    def cont_event(self, df, sample_field, data_field):
        r0 = [float(0), float(0), float(0)]
        if df.empty:
            return r0
        ds = df.groupby(pd.Grouper(key=sample_field, freq=self.freq)).apply(
            lambda x: x.pivot_table(index=[data_field], aggfunc='size'))
        y = ds.groupby(level=0).agg(np.mean)
        nz = y[y > 0]
        if len(nz) == 0:
            return r0
        return [np.median(nz), mad_calc(nz), np.mean(nz)]

    def __call__(self, data_col, flag, func='count'):
        if flag == 'weekend':
            df = self.df1
        else:
            df = self.df2
        y = self.cont_event(df, self.datetime_col, data_col) if func == 'size' else self.weekdays_count_func(df, data_col, func)
        z = burst_func(df, sample_field=self.datetime_col, data_field=data_col, func='count', freq1='20S', freq2='W', filter_by_hours=False)
        return y + z

