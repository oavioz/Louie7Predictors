
import pandas as pd
import numpy as np
import  vector_creator.stats_models.estimators as est
from statsmodels.tsa.stattools import adfuller

# AR(1) , AR(2), AR(4) , AR(8), AR(16)
def ar_dur(df, datetime_col, dur_col, t_size=3):
    df[dur_col] = df[dur_col].astype(np.uint32)
    y0 = df.groupby(pd.Grouper(key=datetime_col, freq='4H')).agg({dur_col: ['mean']}).fillna(0)
    y = y0[dur_col].to_numpy().T[0]
    return y[0:len(y) - t_size], y[len(y) - t_size:]


def ar_calls(df, datetime_col, num_col, t_size=3):
    y0 = df.groupby(pd.Grouper(key=datetime_col, freq='D')).agg({num_col: ['count']})
    y = y0[num_col].to_numpy().T[0]
    return y[0:len(y) - t_size], y[len(y) - t_size:]



def ar_model(train, test, lag, mse):
    if len(train) < 32:
        return 0.0
    return [est.ar_model_2(train=train, test=test, lag=lag, mse=mse)]


def adfuller_test(train):
    tpl = adfuller(train)
    # p_value  dicky_fuller_test null hypothesis
    p_value = tpl[1]
    rej_null_h = 1 if p_value < 0.05 else 0
    return [rej_null_h]