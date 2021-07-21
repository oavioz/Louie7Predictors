from vector_creator.preprocess.est_by_df_column import col_stats_func, minmax_by_cat, minmax_by_cat_value, col_delta_stats_func
from vector_creator.preprocess.utils import filter_day_hours, filter_by_weekends
from vector_creator.preprocess.entropy import entropy_of_cat
import pandas as pd


'''
installed_apps_score_vector[
     num_of_app_installed,
     mean_installed_apps_in_month,
     std_installed_apps_in_month,
     max_installed_apps_in_month,
     min_installed_apps_in_month,
     mean_time_between_installs,
     std_time_between_installs,
     max_time_between_installs,
     min_time_between_installs,
     max_category,
     min_category,
     day_time_installs_ratio,
     night_time_installs_ratio,
     weekend_installs_ratio,
     paid_installs_ratio,
     free_installs_ratio,
     max_category_paid,
     min_category_paid,
     max_category_free,
     min_category_free,
     category_entropy
]
'''


apps_installed = {'categories': ['Art & Design', 'Auto & Vehicles', 'Beauty', 'Books & Reference', 'Business', 'Comics', 'Communications',
                         'Dating', 'Education', 'Entertainment', 'Events', 'Finance', 'Food & Drink', 'Health & Fitness', 'House & Home',
                         'Lifestyle', 'Maps & Navigation', 'Medical', 'Music & Audio', 'News & Magazines', 'Parenting',
                         'Personalization', 'Photography', 'Productivity', 'Shopping', 'Social', 'Sports', 'Tools', 'Travel & Local',
                         'Video Players & Editors', 'Weather', 'Libraries & Demo', 'Action', 'Adventure', 'Arcade', 'Board', 'Cards',
                         'Casino', 'Casual', 'Educational', 'Music Games', 'Puzzle', 'Racing', 'Role Playing',
                         'Simulation', 'Sport Games', 'Strategy', 'Trivia', 'Word Games', 'Family All Ages', 'Family Action',
                         'Family Brain Games', 'Family Create', 'Family Education', 'Family Music & Video', 'Family Pretend Play'],
                  'columns': ['INSTALL_DATETIME', 'APP_CATEGORY', 'APP_VARIANT']}


def create_app_install_vector_for_unique_id(uid, df0, lat_long):
    app_col = apps_installed['columns']
    app_cat = apps_installed['categories']
    df = df0.sort_values(app_col[0])
    s = len(df)
    #
    apps_installed_score_vec = col_stats_func(df=df, sample_field=app_col[0], cat_field=app_col[1], func='count', freq='M')
    apps_installed_score_vec += minmax_by_cat(df=df, cat_field=app_col[1], func='count')
    apps_installed_score_vec += col_delta_stats_func(df=df, sample_field=app_col[0])
    apps_installed_score_vec += [len(filter_day_hours(df, app_col[0], '07:00:00', '19:00:00')) / s] # daytime app installs ratio
    apps_installed_score_vec += [len(filter_day_hours(df, app_col[0], '19:00:00', '07:00:00')) / s] # nighttime app installs ratio
    apps_installed_score_vec += [len(filter_by_weekends(df, lat_long, app_col[0], 'day_of_week')) / s] # weekend app installs ratio
    apps_installed_score_vec += minmax_by_cat_value(df, cat_field=app_col[1], val_field=app_col[2], val='Free')
    free_apps_ratio = len(df[df[app_col[2]] == 'Free']) / s
    apps_installed_score_vec += [free_apps_ratio, 1-free_apps_ratio]
    apps_installed_score_vec += entropy_of_cat(df=df, cat_col=app_col[1], categories=app_cat)

    return apps_installed_score_vec
    # return pd.Series(apps_installed_score_vec, name=uid)
