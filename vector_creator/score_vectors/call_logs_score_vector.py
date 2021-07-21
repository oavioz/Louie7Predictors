import vector_creator.preprocess.est_by_df_column as df_col
import vector_creator.preprocess.robust_estimation as rb_est
import vector_creator.preprocess.ivi_irregularity as ivi
import vector_creator.preprocess.entropy as ent
import vector_creator.preprocess.auto_regression as ar
import pandas as pd


'''
call_logs_score_vec[
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

def create_call_logs_vector_for_unique_id(uid, df, lat_long):
    call_logs_score_vec = []
    cl_col = call_logs['columns']  # ['CALL_DATE_TIME', 'CALL_NUMBER', 'CALL_DURATION', 'CALL_TYPE']
    cl_cat = call_logs['categories']  # ['INCOMING', 'OUTGOING', 'MISSED']
    # append [mean, std]
    call_logs_score_vec += df_col.daily_mean_std_func(df, sample_field=cl_col[0], data_field=cl_col[1], func='count')
    call_logs_score_vec += df_col.daily_mean_std_func(df, sample_field=cl_col[0], data_field=cl_col[1], func='nunique')
    call_logs_score_vec += df_col.daily_mean_std_cont_event(df, sample_field=cl_col[0], data_field=cl_col[1])
    call_logs_score_vec += df_col.daily_mean_std_by_cat(df, sample_field=cl_col[0], data_field=cl_col[1], cat_field=cl_col[3], cat=cl_cat[0], func='count')
    call_logs_score_vec += df_col.daily_mean_std_by_cat(df, sample_field=cl_col[0], data_field=cl_col[1], cat_field=cl_col[3], cat=cl_cat[1], func='count')
    call_logs_score_vec += df_col.daily_mean_std_by_cat(df, sample_field=cl_col[0], data_field=cl_col[1], cat_field=cl_col[3], cat=cl_cat[2], func='count')
    #
    night_h = df_col.NightHours(sample_col=cl_col[0], data_col=cl_col[1])
    call_logs_score_vec += night_h(df, func='count')
    call_logs_score_vec += night_h(df, func='nunique')
    night_h_cat = df_col.NightHoursByCat(cat_col=cl_col[3], sample_col=cl_col[0], data_col=cl_col[1])
    call_logs_score_vec += night_h_cat(df, cat=cl_cat[0], func='count')
    call_logs_score_vec += night_h_cat(df, cat=cl_cat[1], func='count')
    call_logs_score_vec += night_h_cat(df, cat=cl_cat[2], func='count')
    #
    weekend_h = df_col.WeekendHours(df, datetime_col=cl_col[0], data_col=cl_col[1], long_lat_tuple=lat_long)
    call_logs_score_vec += weekend_h(func='count')
    call_logs_score_vec += weekend_h(func='nunique')
    call_logs_score_vec += weekend_h(func='size')
    #
    hub_m = rb_est.HuberM(sample_col=cl_col[0], data_col=cl_col[2], lat_long_tuple=lat_long)
    call_logs_score_vec += hub_m(df, flag='day')
    call_logs_score_vec += rb_est.huber_m_by_cat(hub_m, df, cat_col=cl_col[3], cat=cl_cat[0], flag='day')
    call_logs_score_vec += rb_est.huber_m_by_cat(hub_m, df, cat_col=cl_col[3], cat=cl_cat[1], flag='day')
    call_logs_score_vec += hub_m(df, flag='night')
    call_logs_score_vec += rb_est.huber_m_by_cat(hub_m, df, cat_col=cl_col[3], cat=cl_cat[0], flag='night')
    call_logs_score_vec += rb_est.huber_m_by_cat(hub_m, df, cat_col=cl_col[3], cat=cl_cat[1], flag='night')
    call_logs_score_vec += hub_m(df, flag='weekend')
    call_logs_score_vec += rb_est.huber_m_by_cat(hub_m, df, cat_col=cl_col[3], cat=cl_cat[0], flag='weekend')
    call_logs_score_vec += rb_est.huber_m_by_cat(hub_m, df, cat_col=cl_col[3], cat=cl_cat[1], flag='weekend')
    #
    q_n = rb_est.Qn(sample_col=cl_col[0], data_col=cl_col[2], lat_long_tuple=lat_long)
    call_logs_score_vec += q_n(df, flag='day')
    call_logs_score_vec += rb_est.qn_by_cat(q_n, df, cat_col=cl_col[3], cat=cl_cat[0], flag='day')
    call_logs_score_vec += rb_est.qn_by_cat(q_n, df, cat_col=cl_col[3], cat=cl_cat[1], flag='day')
    call_logs_score_vec += q_n(df, flag='night')
    call_logs_score_vec += rb_est.qn_by_cat(q_n, df, cat_col=cl_col[3], cat=cl_cat[0], flag='night')
    call_logs_score_vec += rb_est.qn_by_cat(q_n, df, cat_col=cl_col[3], cat=cl_cat[1], flag='night')
    call_logs_score_vec += q_n(df, flag='weekend')
    call_logs_score_vec += rb_est.qn_by_cat(q_n, df, cat_col=cl_col[3], cat=cl_cat[0], flag='weekend')
    call_logs_score_vec += rb_est.qn_by_cat(q_n, df, cat_col=cl_col[3], cat=cl_cat[1], flag='weekend')
    #
    call_logs_score_vec += ent.entropy_of_duration_by_cat(df, cl_col[2], cl_col[3], cat=cl_cat[0])
    call_logs_score_vec += ent.entropy_of_duration_by_cat(df, cl_col[2], cl_col[3], cat=cl_cat[1])
    call_logs_score_vec += ent.entropy_of_freq_by_cat(df, cl_col[1], cl_col[3], cat=cl_cat[0])
    call_logs_score_vec += ent.entropy_of_freq_by_cat(df, cl_col[1], cl_col[3], cat=cl_cat[1])
    #
    call_logs_score_vec += df_col.call_response_rate(df, cl_col[3], cl_cat)
    call_logs_score_vec += df_col.outgoing_answered_rate(df, cl_col[3], cl_col[2], cl_cat)
    #
    train, test = ar.ar_calls(df, cl_col[0], cl_col[1])
    call_logs_score_vec += ar.adfuller_test(train)
    call_logs_score_vec += ar.ar_model(train, test, 1, True)
    call_logs_score_vec += ar.ar_model(train, test, 4, True)
    call_logs_score_vec += ar.ar_model(train, test, 8, True)
    train, test = ar.ar_dur(df, cl_col[0], cl_col[2])
    call_logs_score_vec += ar.adfuller_test(train)
    call_logs_score_vec += ar.ar_model(train, test, 1, False)
    call_logs_score_vec += ar.ar_model(train, test, 4, False)
    call_logs_score_vec += ar.ar_model(train, test, 8, False)
    #
    ivi_obj = ivi.IVI2(cl_col[0], cl_col[2], cl_col[1], cl_col[3], '6H', 'D')
    call_logs_score_vec += ivi_obj(flag='number', df=df)
    call_logs_score_vec += ivi_obj(flag='duration', df=df)
    call_logs_score_vec += ivi_obj(flag='deltaT', df=df, cat=cl_cat[0])
    call_logs_score_vec += ivi_obj(flag='deltaT', df=df, cat=cl_cat[1])
    ivi_obj = ivi.IVI2(cl_col[0], cl_col[2], cl_col[1], cl_col[3], '3H', 'D')
    call_logs_score_vec += ivi_obj(flag='number', df=df)
    call_logs_score_vec += ivi_obj(flag='duration', df=df)
    call_logs_score_vec += ivi_obj(flag='deltaT', df=df, cat=cl_cat[0])
    call_logs_score_vec += ivi_obj(flag='deltaT', df=df, cat=cl_cat[1])
    #
    return pd.Series(call_logs_score_vec, name=uid)