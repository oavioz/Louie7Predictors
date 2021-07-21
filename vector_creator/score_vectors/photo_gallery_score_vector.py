import vector_creator.preprocess.est_by_df_column as df_col
import vector_creator.preprocess.robust_estimation as rb_est
import vector_creator.preprocess.ivi_irregularity as ivi
import vector_creator.preprocess.entropy as ent
import vector_creator.preprocess.auto_regression as ar
import pandas as pd

'''
photo_gallery_score_vec[
    daily_num_photos_mean,
    daily_num_photos_std,
    daily_unique_photos_mean,
    daily_unique_photos_std,
    daily_num_weekend_photos_mean,
    daily_num_weekend_photos_std,
    daily_num_photos_mean_by_category,
    daily_num_photos_std_by_category,
    daily_unique_weekend_photos_mean,
    daily_unique_weekend_photos_std,
    huberM_num_photos_mean, 
    huberM_num_photos_std,
    huberM_num_weekend_photos_mean,
    huberM_num_weekend_photos_std,
    huber_m_by_cat,
    qn_num_photos,
    qn_num_weekend_photos,
    entropy_type_photos,
    ad_fuller_num_photos
    ar_num_photos_lag_1,
    ar_num_photos_lag_4,
    ar_num_photos_lag_8,
    ar_num_photos_lag_12,
    calc_ivi_photos_tf_3h,
    calc_ivi_photos_tf_6h,
    calc_ivi_photos_tf_d,
]
'''

photo_gallery = {'columns': ['IMAGE_DATE_TIME', 'IMAGE_TYPE'],
                 'categories': ['jpeg', 'png', 'gif', 'bmp', 'webp', 'heif', 'mp4', 'mkv', '3gp', 'webm']}


def create_photo_gallery_vector_for_unique_id(uid, df, lat_long):
    photo_gallery_score_vec = []
    pg_col = photo_gallery['columns']
    pg_cat = photo_gallery['categories']
    #
    photo_gallery_score_vec += df_col.daily_mean_std_func(df, sample_field=pg_col[0], data_field=pg_col[1], func='count')
    photo_gallery_score_vec += df_col.daily_mean_std_func(df, sample_field=pg_col[0], data_field=pg_col[1], func='nunique')
    for cat in pg_cat:
        photo_gallery_score_vec += df_col.daily_mean_std_by_cat(df, pg_col[0], pg_col[1], pg_col[1], cat, 'count')
    #
    weekend_h = df_col.WeekendHours(df, datetime_col=pg_col[0], data_col=pg_col[1], long_lat_tuple=lat_long)
    photo_gallery_score_vec += weekend_h('count')
    photo_gallery_score_vec += weekend_h('nunique')
    #
    q_n = rb_est.Qn2(sample_col=pg_col[0], data_col=pg_col[1], lat_long_tuple=lat_long)
    photo_gallery_score_vec += q_n(df, flag='type')
    photo_gallery_score_vec += q_n(df, flag='weekend')
    #
    train, test = ar.ar_calls(df, pg_col[0], pg_col[1])
    photo_gallery_score_vec += ar.adfuller_test(train)
    photo_gallery_score_vec += ar.ar_model(train, test, 1, True)
    photo_gallery_score_vec += ar.ar_model(train, test, 4, True)
    photo_gallery_score_vec += ar.ar_model(train, test, 8, True)
    photo_gallery_score_vec += ar.ar_model(train, test, 12, True)
    #
    photo_gallery_score_vec += ent.entropy_of_cat(df, pg_col[1], pg_cat)
    #
    ivi_obj = ivi.IVI(pg_col[0], pg_col[1], 'D', 'W')
    photo_gallery_score_vec += ivi_obj(flag='occurr', df=df)
    photo_gallery_score_vec += ivi_obj(flag='deltaT', df=df)
    ivi_obj = ivi.IVI(pg_col[0], pg_col[1], '6H', 'D')
    photo_gallery_score_vec += ivi_obj(flag='occurr', df=df)
    photo_gallery_score_vec += ivi_obj(flag='deltaT', df=df)
    ivi_obj = ivi.IVI(pg_col[0], pg_col[1], '3H', 'D')
    photo_gallery_score_vec += ivi_obj(flag='occurr', df=df)
    photo_gallery_score_vec += ivi_obj(flag='deltaT', df=df)
    #
    return pd.Series(photo_gallery_score_vec, name=uid)