import pandas as pd
import numpy as np
from vector_creator.stats_models.estimators import huber_est


def call_response_rate(df, data_col, cat):
    dn = df.groupby(data_col).agg({data_col: ['count']})
    if len(dn) < 3:
        return [float()]
    incoming = dn.loc[cat[0]:].iloc[0]
    missed = dn.loc[cat[2]:].iloc[0]
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
        r_est = huber_est(nz)[0]
    except ValueError:
        print('huber est was not calculated')
    except ZeroDivisionError:
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