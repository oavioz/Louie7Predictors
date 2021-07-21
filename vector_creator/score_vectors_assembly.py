from vector_creator.raw_to_df.rawdata_to_df import group_metadata, uid_df_init_metadata
from vector_creator.score_vectors.call_logs_score_vector import create_call_logs_vector_for_unique_id
from vector_creator.score_vectors.photo_gallery_score_vector import create_photo_gallery_vector_for_unique_id
from vector_creator.score_vectors.apps_installed_score_vector import create_app_install_vector_for_unique_id
import pandas as pd


'''
processed_unique_id_list : already processed unique_id json files
return : 
        dict {metadata_file_list : [files],  unique_uid_list : [u_ids], processed_unique_id_list : [updated list]}
'''


def file_list_with_unique_id(path):
    return group_metadata(path)


'''
return :
        dict {sampling config},
        dict {init_metadata_field_name(CallLogs, ImgMetaData, InstallApps, LocationInfo) : df}
'''
vector_len = {'call_logs' : 78, 'photo_gallery' : 41, 'install_apps' : 19}
thd = {'call_logs' : 100, 'photo_gallery' : 100, 'install_apps' : 15}

def df_for_init_meta(uid, path, metadata_file_list):
    return uid_df_init_metadata(uid, path, metadata_file_list)


def score_vector_for_init_metadata(uid, df_dict, lat_long):
    df = df_dict.get(uid+'_CallLogs')
    print(len(df))
    call_logs_score_vector =  create_call_logs_vector_for_unique_id(uid=uid, df0=df, lat_long=lat_long) if len(df) >= thd['call_logs'] else [0] * vector_len['call_logs']
    df = df_dict.get(uid+'_ImgMetaData')
    print(len(df))
    photo_gallery_vector = create_photo_gallery_vector_for_unique_id(uid=uid, df0=df, lat_long=lat_long) if len(df) >= thd['photo_gallery'] else [0] * vector_len['photo_gallery']
    df = df_dict.get(uid+'_InstallApps')
    print(len(df))
    app_installed_vector = create_app_install_vector_for_unique_id(uid=uid, df0=df, lat_long=lat_long) if len(df) >= thd['install_apps'] else [0] * vector_len['install_apps']
    score_vector = call_logs_score_vector + photo_gallery_vector + app_installed_vector
    return pd.Series(score_vector, name=uid)


def combine_score_vectors(path):
    uid_list = group_metadata(path=path)
    score_vector_dict = {}
    for uid in uid_list.get('unique_ids'):
        print(uid)
        loc_dict, uid_df_dict = uid_df_init_metadata(uid, path, uid_list.get('file_list'))
        loc_tuple = (loc_dict[0]['Latitude'], loc_dict[0]['Longitude'])
        score_vector = score_vector_for_init_metadata(uid, uid_df_dict, loc_tuple)
        score_vector_dict[score_vector.name] = score_vector
    return pd.concat(score_vector_dict, axis=1)


#def convert_to_oci_ds(df):
