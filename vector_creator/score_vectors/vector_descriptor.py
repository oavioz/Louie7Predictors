from vector_creator.preprocess.est_by_df_column import *
from vector_creator.preprocess.install_apps_features import *
from vector_creator.preprocess.ivi_irregularity import IVI2, calc_ivi_number_by_cat
from vector_creator.preprocess.auto_regression import *
from vector_creator.preprocess.entropy import *
import numpy as np
from itertools import chain

'''
we create 3 dicts for call-logs photo gallery and install apps
each dictionary key is a description of the function and the value
is the function name itself
'''

apps_installed = {'categories':
                      ['Art & Design', 'Auto & Vehicles', 'Beauty', 'Books & Reference', 'Business', 'Comics', 'Communications',
                      'Dating', 'Education', 'Entertainment', 'Events', 'Finance', 'Food & Drink', 'Health & Fitness', 'House & Home',
                      'Lifestyle', 'Maps & Navigation', 'Medical', 'Music & Audio', 'News & Magazines', 'Parenting',
                      'Personalization', 'Photography', 'Productivity', 'Shopping', 'Social', 'Sports', 'Tools', 'Travel & Local',
                      'Video Players & Editors', 'Weather', 'Libraries & Demo', 'Action', 'Adventure', 'Arcade', 'Board', 'Cards',
                      'Casino', 'Casual', 'Educational', 'Music Games', 'Puzzle', 'Racing', 'Role Playing',
                      'Simulation', 'Sport Games', 'Strategy', 'Trivia', 'Word Games', 'Family All Ages', 'Family Action',
                      'Family Brain Games', 'Family Create', 'Family Education', 'Family Music & Video', 'Family Pretend Play'],
                  'columns':
                      ['INSTALL_DATETIME', 'APP_CATEGORY', 'APP_VARIANT']}



def apps_installed_vector_descriptor(df):
    app_col = apps_installed['columns']
    app_cat = apps_installed['categories']

    vector_descriptor = [
        mean_max_ratio(df, data_col=app_col[1]),
        category_coverage(df, data_col=app_col[1], categories=len(app_cat)),
        #ratio_of_paid_apps(df, data_col=app_col[2], paid_str='Paid Feature'),
        entropy_of_cat(df=df, cat_col=app_col[1], categories=app_cat, fetcher_group='apps-installed'),
        entropy_by_time(df=df, dt_col=app_col[0], freq='3H')
    ]
    return list(chain.from_iterable(vector_descriptor))


photo_gallery = {'columns': ['IMAGE_DATE_TIME', 'IMAGE_TYPE'],
                 'categories': ['jpeg', 'png', 'gif', 'webp', 'heif', 'mp4', 'mkv', '3gp', 'webm']}


def photo_gallery_vector_descriptor(df, lat_long):
    pg_col = photo_gallery['columns']
    pg_cat = photo_gallery['categories']
    #
    day_h = DailyHours(sample_col=pg_col[0])
    week_d = WeekDays(df, datetime_col=pg_col[0], long_lat_tuple=lat_long)
    train, test = ar_count(df, pg_col[0], pg_col[1])
    # ivi_obj = IVI(pg_col[0], pg_col[1], 'D', 'W')
    #
    vector_descriptor = [
        daily_func(df, sample_field=pg_col[0], data_field=pg_col[1], func='count', freq='D'),
        burst_func(df, sample_field=pg_col[0], data_field=pg_col[1], func='count', freq1='20S', freq2='W'),
        day_h(df=df, data_col=pg_col[1], start_time='20:00:00', stop_time='08:00:00', func='count'),
        day_h(df=df, data_col=pg_col[1], start_time='08:00:00', stop_time='20:00:00', func='count'),
        week_d(data_col=pg_col[1], freq='D', flag='weekend' , func='count'),
        week_d(data_col=pg_col[1], freq='D', flag='workdays', func='count'),
        ar(train, test, 1, True),
        ar(train, test, 2, True),
        ar(train, test, 4, True),
        entropy_of_cat(df, pg_col[1], pg_cat, 'photo-gallery'),
        entropy_of_amount(df=df, date_col=pg_col[0], cat_col=pg_col[1])
    ]
    return list(chain.from_iterable(vector_descriptor))


call_logs = {'categories' : ['INCOMING', 'OUTGOING', 'MISSED'],
             'columns' : ['CALL_DATE_TIME', 'CALL_NUMBER', 'CALL_DURATION', 'CALL_TYPE']}



def call_logs_vector_descriptor(df, lat_long):
    def f(data):
        y = lambda x: x.astype(np.float64)
        return np.sum(y(data))

    cl_col = call_logs['columns']  # ['CALL_DATE_TIME', 'CALL_NUMBER', 'CALL_DURATION', 'CALL_TYPE']
    cl_cat = call_logs['categories']  # ['INCOMING', 'OUTGOING', 'MISSED']

    night_h = DailyHours(sample_col=cl_col[0])
    weekend_h = WeekDays(df, datetime_col=cl_col[0], long_lat_tuple=lat_long)
    train_c, test_c = ar_count(df, cl_col[0], cl_col[1])
    train_d, test_d = ar_dur(df, cl_col[0], cl_col[2])
    ivi_obj = IVI2(cl_col[0], cl_col[2], cl_col[1], cl_col[3], 'D', 'W')

    vector_descriptor = [
        daily_func(df, sample_field=cl_col[0], data_field=cl_col[1], func='count', freq='D'),
        daily_func(df, sample_field=cl_col[0], data_field=cl_col[1], func='nunique', freq='D'),
        daily_func(df, sample_field=cl_col[0], data_field=cl_col[2], func=f, freq='D'),
        daily_cont_event(df, sample_field=cl_col[0], data_field=cl_col[1]),
        daily_func_by_cat(df, sample_field=cl_col[0], data_field=cl_col[1], cat_field=cl_col[3], cat=cl_cat[0], func='count'),
        daily_func_by_cat(df, sample_field=cl_col[0], data_field=cl_col[1], cat_field=cl_col[3], cat=cl_cat[0], func='nunique'),
        daily_func_by_cat(df, sample_field=cl_col[0], data_field=cl_col[2], cat_field=cl_col[3], cat=cl_cat[0], func=f),
        daily_cont_event_by_cat(df, sample_field=cl_col[0], cat_field=cl_col[3], cat=cl_cat[0], data_field=cl_col[1]),
        daily_func_by_cat(df, sample_field=cl_col[0], data_field=cl_col[1], cat_field=cl_col[3], cat=cl_cat[1], func='count'),
        daily_func_by_cat(df, sample_field=cl_col[0], data_field=cl_col[1], cat_field=cl_col[3], cat=cl_cat[1], func='nunique'),
        daily_func_by_cat(df, sample_field=cl_col[0], data_field=cl_col[2], cat_field=cl_col[3], cat=cl_cat[1], func=f),
        daily_cont_event_by_cat(df, sample_field=cl_col[0], cat_field=cl_col[3], cat=cl_cat[1], data_field=cl_col[1]),
        daily_func_by_cat(df, sample_field=cl_col[0], data_field=cl_col[1], cat_field=cl_col[3], cat=cl_cat[2], func='count'),
        night_h(df, data_col=cl_col[1], start_time='20:00:00', stop_time='08:00:00', func='count'),
        night_h(df, data_col=cl_col[1], start_time='20:00:00', stop_time='08:00:00', func='nunique'),
        night_h(df, data_col=cl_col[2], start_time='20:00:00', stop_time='08:00:00', func=f),
        night_h(df, data_col=cl_col[1], start_time='20:00:00', stop_time='08:00:00', func='size'),
        weekend_h(data_col=cl_col[1], freq='D', flag='weekend', func='count'),
        weekend_h(data_col=cl_col[1], freq='D', flag='weekend', func='nunique'),
        weekend_h(data_col=cl_col[2], freq='D', flag='weekend', func=f),
        weekend_h(data_col=cl_col[1], freq='D', flag='weekend', func='size'),
        call_response_rate(df, cl_col[3], cl_cat),
        outgoing_answered_rate(df, cl_col[3], cl_col[2], cl_cat),
        entropy_of_duration(df, cl_col[2], cl_col[3], cat='None'),
        entropy_of_duration(df, cl_col[2], cl_col[3], cat=cl_cat[0]),
        entropy_of_duration(df, cl_col[2], cl_col[3], cat=cl_cat[1]),
        entropy_of_freq(df, cl_col[0], cl_col[3], cat='None'),
        entropy_of_freq(df, cl_col[0], cl_col[3], cat=cl_cat[0]),
        entropy_of_freq(df, cl_col[0], cl_col[3], cat=cl_cat[1]),
        entropy_of_number(df, cl_col[1], cl_col[3], cat='None'),
        entropy_of_number(df, cl_col[1], cl_col[3], cat=cl_cat[0]),
        entropy_of_number(df, cl_col[1], cl_col[3], cat=cl_cat[1]),
        ar(train_c, test_c, 1, True),
        ar(train_c, test_c, 4, True),
        ar(train_c, test_c, 8, True),
        ar(train_c, test_c, 16, True),
        ar(train_d, test_d, 1, True),
        ar(train_d, test_d, 4, True),
        ar(train_d, test_d, 8, True),
        ar(train_d, test_d, 16, True),
        ivi_obj(flag='number', df=df),
        ivi_obj(flag='duration', df=df),
        ivi_obj(flag='deltaT', df=df),
        ivi_obj(flag='night', df=df),
        calc_ivi_number_by_cat(df, cl_col[0], cl_col[1], cl_col[3], cl_cat[0], 'D', 'W'),
        calc_ivi_number_by_cat(df, cl_col[0], cl_col[1], cl_col[3], cl_cat[1], 'D', 'W')
    ]
    return list(chain.from_iterable(vector_descriptor))

