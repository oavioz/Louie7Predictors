

from vector_creator.preprocess.est_by_df_column import mean_std_func, daily_mean_std_cont_event, daily_mean_std_by_cat, NightHours, NightHoursByCat, WeekendHours, call_response_rate, outgoing_answered_rate
from vector_creator.preprocess.ivi_irregularity import IVI2, calc_ivi_number_by_cat
from vector_creator.preprocess.entropy import entropy_of_freq_by_cat, entropy_of_duration_by_cat
from vector_creator.preprocess.auto_regression import ar_calls, adfuller_test, ar_model, ar_dur
import pandas as pd
import numpy as np


'''
call_logs_score_vec = [
    daily_num_calls_mean,
    daily_num_calls_std,
    daily_num_unique_calls_mean,
    daily_num_unique_calls_std,
    daily_num_continuous_calls_mean,
    daily_num_continuous_calls_std,
    daily_num_incoming_calls_mean,
    daily_num_incoming_calls_std,
    daily_num_outgoing_calls_mean,
    daily_num_outgoing_calls_std,
    daily_num_missed_calls_mean,
    daily_num_missed_calls_std,
    nightly_num_call_mean,
    nightly_num_call_std,
    nightly_num_unique_calls_mean,
    nightly_num_unique_calls_std,
    nightly_num_incoming_calls_mean,
    nightly_num_incoming_calls_std,
    nightly_num_outgoing_calls_mean,
    nightly_num_outgoing_calls_std,
    nightly_num_missed_calls_mean,
    nightly_num_missed_calls_std,
    daily_num_weekend_calls_mean,
    daily_num_weekend_calls_std,
    daily_num_unique_weekend_calls_mean,
    daily_num_unique_weekend_calls_std,
    daily_num_continuous_weekend_calls_mean,
    daily_num_continuous_weekend_calls_std,
    huberM_dur_calls_mean, 
    huberM_dur_calls_std,
    huberM_dur_incoming_calls_mean,
    huberM_dur_incoming_calls_std,
    huberM_dur_outgoing_calls_mean,
    huberM_dur_outgoing_calls_std,
    huberM_dur_night_calls_mean,
    huberM_dur_night_calls_std,
    huberM_dur_incoming_night_calls_mean,
    huberM_dur_incoming_night_calls_std,
    huberM_dur_outgoing_night_calls_mean,
    huberM_dur_outgoing_night_calls_std,
    huberM_dur_weekend_calls_mean,
    huberM_dur_weekend_calls_std,
    huberM_dur_incoming_weekend_calls_mean,
    huberM_dur_incoming_weekend_calls_std,
    huberM_dur_outgoing_weekend_calls_mean,
    huberM_dur_outgoing_weekend_calls_std,
    qn_dur_calls,
    qn_dur_incoming_calls,
    qn_dur_outgoing_calls,
    qn_dur_night_calls,
    qn_dur_incoming_night_calls,
    qn_dur_outgoing_night_calls,
    qn_dur_weekend_calls,
    qn_dur_incoming_weekend_calls,
    qn_dur_outgoing_weekend_calls,
    entropy_dur_incoming_calls,
    entropy_dur_outgoing_calls,
    entropy_freq_incoming_calls,
    entropy_freq_outgoing_calls,
    call_response_rate, 
    outgoing_answered_rate,
    ad_fuller_num_calls
    ar_num_calls_lag_1,
    ar_num_calls_lag_4,
    ar_num_calls_lag_8,
    ad_fuller_dur_calls
    ar_dur_calls_lag_1,
    ar_dur_calls_lag_4,
    ar_dur_calls_lag_8,
    calc_ivi_calls_by_number_tf_6h,
    calc_ivi_dur_by_number_tf_6h,
    calc_ivi_calls_by_time_diff_tf_6h,
    calc_ivi_calls_by_number_tf_3h,
    calc_ivi_dur_by_number_tf_3h,
    calc_ivi_calls_by_time_diff_tf_3h
]   
'''

call_logs = {'categories' : ['INCOMING', 'OUTGOING', 'MISSED'],
             'columns' : ['CALL_DATE_TIME', 'CALL_NUMBER', 'CALL_DURATION', 'CALL_TYPE']}

def f(data):
    y = lambda x : x.astype(np.float64)
    return np.sum(y(data))

def create_call_logs_vector_for_unique_id(uid, df0, lat_long):
    cl_col = call_logs['columns']  # ['CALL_DATE_TIME', 'CALL_NUMBER', 'CALL_DURATION', 'CALL_TYPE']
    cl_cat = call_logs['categories']  # ['INCOMING', 'OUTGOING', 'MISSED']
    df = df0.sort_values(cl_col[0])
    #
    call_logs_score_vec = mean_std_func(df, sample_field=cl_col[0], data_field=cl_col[1], func='count', freq='D')
    call_logs_score_vec += mean_std_func(df, sample_field=cl_col[0], data_field=cl_col[1], func='nunique', freq='D')
    call_logs_score_vec += mean_std_func(df, sample_field=cl_col[0], data_field=cl_col[2], func=f, freq='D')
    call_logs_score_vec += daily_mean_std_cont_event(df, sample_field=cl_col[0], data_field=cl_col[1])
    call_logs_score_vec += daily_mean_std_by_cat(df, sample_field=cl_col[0], data_field=cl_col[1], cat_field=cl_col[3], cat=cl_cat[0], func='count')
    call_logs_score_vec += daily_mean_std_by_cat(df, sample_field=cl_col[0], data_field=cl_col[2], cat_field=cl_col[3], cat=cl_cat[0], func=f)
    call_logs_score_vec += daily_mean_std_by_cat(df, sample_field=cl_col[0], data_field=cl_col[1], cat_field=cl_col[3], cat=cl_cat[1], func='count')
    call_logs_score_vec += daily_mean_std_by_cat(df, sample_field=cl_col[0], data_field=cl_col[2], cat_field=cl_col[3], cat=cl_cat[1], func=f)
    call_logs_score_vec += daily_mean_std_by_cat(df, sample_field=cl_col[0], data_field=cl_col[1], cat_field=cl_col[3], cat=cl_cat[2], func='count')
    #
    night_h = NightHours(sample_col=cl_col[0])
    call_logs_score_vec += night_h(df, data_col=cl_col[1], func='count')
    call_logs_score_vec += night_h(df, data_col=cl_col[1], func='nunique')
    night_h_cat = NightHoursByCat(cat_col=cl_col[3], sample_col=cl_col[0])
    call_logs_score_vec += night_h_cat(df, data_col=cl_col[1], cat=cl_cat[0], func='count')
    call_logs_score_vec += night_h_cat(df, data_col=cl_col[2], cat=cl_cat[0], func=f)
    call_logs_score_vec += night_h_cat(df, data_col=cl_col[1], cat=cl_cat[1], func='count')
    call_logs_score_vec += night_h_cat(df, data_col=cl_col[2], cat=cl_cat[1], func=f)
    call_logs_score_vec += night_h_cat(df, data_col=cl_col[1], cat=cl_cat[2], func='count')
    #
    weekend_h = WeekendHours(df, datetime_col=cl_col[0], long_lat_tuple=lat_long)
    call_logs_score_vec += weekend_h(data_col= cl_col[1], freq='D', func='count')
    call_logs_score_vec += weekend_h(data_col= cl_col[2], freq='D', func=f)
    call_logs_score_vec += weekend_h(data_col= cl_col[1], freq='D', func='nunique')
    call_logs_score_vec += weekend_h(data_col= cl_col[1], freq='D', func='size')
    #
    call_logs_score_vec += entropy_of_duration_by_cat(df, cl_col[2], cl_col[3], cat=cl_cat[0])
    call_logs_score_vec += entropy_of_duration_by_cat(df, cl_col[2], cl_col[3], cat=cl_cat[1])
    call_logs_score_vec += entropy_of_freq_by_cat(df, cl_col[1], cl_col[3], cat=cl_cat[0])
    call_logs_score_vec += entropy_of_freq_by_cat(df, cl_col[1], cl_col[3], cat=cl_cat[1])
    #
    call_logs_score_vec += call_response_rate(df, cl_col[3], cl_cat)
    call_logs_score_vec += outgoing_answered_rate(df, cl_col[3], cl_col[2], cl_cat)
    #
    train, test = ar_calls(df, cl_col[0], cl_col[1])
    call_logs_score_vec += adfuller_test(train)
    call_logs_score_vec += ar_model(train, test, 1, True)
    call_logs_score_vec += ar_model(train, test, 4, True)
    call_logs_score_vec += ar_model(train, test, 8, True)
    train, test = ar_dur(df, cl_col[0], cl_col[2])
    call_logs_score_vec += adfuller_test(train)
    call_logs_score_vec += ar_model(train, test, 1, False)
    call_logs_score_vec += ar_model(train, test, 4, False)
    call_logs_score_vec += ar_model(train, test, 8, False)
    #
    ivi_obj = IVI2(cl_col[0], cl_col[2], cl_col[1], cl_col[3], 'D', 'W')
    call_logs_score_vec += ivi_obj(flag='number', df=df)
    call_logs_score_vec += ivi_obj(flag='duration', df=df)
    call_logs_score_vec += ivi_obj(flag='deltaT', df=df)
    call_logs_score_vec += calc_ivi_number_by_cat(df, cl_col[0], cl_col[1], cl_col[3], cl_cat[0], 'D', 'W')
    call_logs_score_vec += calc_ivi_number_by_cat(df, cl_col[0], cl_col[1], cl_col[3], cl_cat[1], 'D', 'W')
    call_logs_score_vec += calc_ivi_number_by_cat(df, cl_col[0], cl_col[1], cl_col[3], cl_cat[2], 'D', 'W')
    #
    return call_logs_score_vec
    #return pd.Series(call_logs_score_vec, name=uid)