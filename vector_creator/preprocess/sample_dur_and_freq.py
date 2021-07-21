from vector_creator.preprocess import utils
import numpy as np

'''
dur & freq calculation :
'''

'''
function return a tuple , 1st element is a duration list for each category ,
and the second tuple arg is the size of that list (the number of times category was active)
collecting  all categories list size will give a freq distribution  
'''
def categoryDurAndFreq_AppUsage(df, datetime_col, delta, data_col, cat):
    # filter rows by category
    df = df.loc[df[data_col] == cat]
    df = __createCulomnDuration(df, datetime_col, delta, 'dur')
    np_list =  delta * df.groupby('dur').apply(lambda x : x.shape[0]).to_numpy()
    return (np_list, np_list.size)


def categoryDurAndFreqNight_AppUsage(df, datetime_col, delta, data_col, cat):
    df = df.loc[df[data_col] == cat]
    df = utils.createColumnDayDate(df, datetime_col, 'date')
    df1 = utils.filterDayHoursActivities(df, datetime_col, '20:00', '23:59')
    df0 = utils.filterDayHoursActivities(df, datetime_col, '00:00', '08:00')
    ## colect morning metadata events by duration
    df0 = create_column_duration(df0, datetime_col, delta, 'dur')
    df0[datetime_col] = (df0[datetime_col]).dt.date
    np0 = df0.groupby('date').apply(lambda y : y.groupby('dur').apply(lambda x : x.shape[0])).drop(df0.head(1).index, inplace=True).to_numpy()
    ## colect night metadata events by duration
    df1 = create_column_duration(df1, datetime_col, delta, 'dur')
    df1[datetime_col] = (df1[datetime_col]).dt.date
    np1 = df1.groupby('date').apply(lambda y: y.groupby('dur').apply(lambda x: x.shape[0])).drop(df1.tail(1).index, inplace=True).to_numpy()
    np_list =  utils.addDayNightVectors(np0, np1)
    return (np_list, np_list.size)

'''
collect total duration for all categories as a dict : {cat_name : [total_dur]}
'''
def classDur_AppUsage(df, sample_col, delta, data_col):
    df = create_column_duration(df, sample_col, delta, 'dur')
    ds = df.groupby(data_col).apply(lambda x : x.groupby('dur').apply(lambda y : delta * y.shape[0]).to_numpy())
    return ds.to_dict()


def create_column_duration(df, datetime_col, delta, col_name):
    df[col_name] = df[datetime_col].diff().apply(lambda x: x / np.timedelta64(1, 's')).fillna(0).astype('int64').apply(lambda x: 1 if x > (delta + 1) else 0)
    df[col_name] = df.dur.cumsum()
    return df