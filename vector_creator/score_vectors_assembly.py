from vector_creator.raw_to_df.rawdata_to_df import group_metadata, uid_df_init_metadata
from vector_creator.score_vectors.vector_descriptor import apps_installed_vector_descriptor, photo_gallery_vector_descriptor, call_logs_vector_descriptor
from vector_creator.score_vectors.vector_indexer import vector_desc_call_logs, vector_desc_photo_gallery, vector_desc_installed_apps
import pandas as pd
import shutil


'''
processed_unique_id_list : already processed unique_id json files
return : 
        dict {metadata_file_list : [files],  unique_uid_list : [u_ids], processed_unique_id_list : [updated list]}
'''



vector_len = {'call_logs' : len(vector_desc_call_logs),
              'photo_gallery' : len(vector_desc_photo_gallery),
              'install_apps' : len(vector_desc_installed_apps)}
thd = {'call_logs' : 100, 'photo_gallery' : 100, 'install_apps' : 15}
folders = {'processed_files' : 'processed/', 'small_files' : 'less_then_thd_size/'}


'''
    call the apps_installed_vector_descriptor with sorted dataframe
    and latitude , longitude GPS coordinates to get the app installed vector values   
'''

def create_app_install_vector_for_unique_id(df0, lat_long):
    df = df0.sort_values('INSTALL_DATETIME')
    apps_installed_score_vec = []
    func_dict = apps_installed_vector_descriptor(df=df, lat_long=lat_long)
    for func_key in func_dict.keys():
        apps_installed_score_vec += func_dict[func_key]
    return apps_installed_score_vec


def create_photo_gallery_vector_for_unique_id(df0, lat_long):
    df = df0.sort_values('IMAGE_DATE_TIME')
    photo_gallery_score_vec = []
    func_dict = photo_gallery_vector_descriptor(df=df, lat_long=lat_long)
    for func_key in func_dict.keys():
        photo_gallery_score_vec += func_dict[func_key]
    return photo_gallery_score_vec


def create_call_logs_vector_for_unique_id(df0, lat_long):
    df = df0.sort_values('CALL_DATE_TIME')
    call_logs_score_vec = []
    func_dict = call_logs_vector_descriptor(df=df, lat_long=lat_long)
    for func_key in func_dict.keys():
        call_logs_score_vec += func_dict[func_key]
    return call_logs_score_vec


def score_vector_for_init_metadata(uid, df_dict, lat_long):
    df = df_dict.get(uid+'_CallLogs')
    print('call-logs: ', len(df))
    call_logs_score_vector = create_call_logs_vector_for_unique_id(df0=df, lat_long=lat_long) if len(df) >= thd['call_logs'] else [0] * vector_len['call_logs']
    '''
    df = df_dict.get(uid+'_ImgMetaData')
    print('photo-gallery: ', len(df))
    photo_gallery_vector = create_photo_gallery_vector_for_unique_id(df0=df, lat_long=lat_long) if len(df) >= thd['photo_gallery'] else [0] * vector_len['photo_gallery']
    df = df_dict.get(uid+'_InstallApps')
    print('install apps: ', len(df))
    app_installed_vector = create_app_install_vector_for_unique_id(df0=df, lat_long=lat_long) if len(df) >= thd['install_apps'] else [0] * vector_len['install_apps']
    '''
    score_vector = call_logs_score_vector # + photo_gallery_vector + app_installed_vector
    return pd.Series(score_vector, name=uid)


def combine_score_vectors(path):
    uid_list = group_metadata(path=path)
    score_vector_dict = {}
    for uid in uid_list.get('unique_ids'):
        print(uid)
        uid_file, loc_dict, uid_df_dict = uid_df_init_metadata(uid, path, uid_list.get('file_list'))
        if not 'empty' in uid_df_dict.keys():
            loc_tuple = (loc_dict[0]['Latitude'], loc_dict[0]['Longitude'])
            score_vector = score_vector_for_init_metadata(uid, uid_df_dict, loc_tuple)
            if not score_vector.any():
                dst = path + folders['small_files'] + uid_file
            else:
                score_vector_dict[score_vector.name] = score_vector
                dst = path + folders['processed_files'] + uid_file
        else:
            dst = path + folders['small_files'] + uid_file
        shutil.move(path+uid_file, dst)
    df = pd.concat(score_vector_dict, axis=1)
    df['description'] = vector_desc_call_logs # + vector_desc_photo_gallery + vector_desc_installed_apps
    return df.set_index('description')

