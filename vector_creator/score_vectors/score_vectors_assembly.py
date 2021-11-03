from vector_creator.raw_to_df.rawdata_to_df import create_df_from_init_metadata
from vector_creator.raw_to_df.load_oci_bucket import namespace, bucket_name
from vector_creator.score_vectors.vector_descriptor import *
from vector_creator.score_vectors.vector_indexer import *
from vector_creator.preprocess.utils import calc_number_of_days
from vector_creator.stats_models.estimators import minmax_scale , z_score
import pandas as pd
import numpy as np
import os
import json
import codecs
import shutil


'''
processed_unique_id_list : already processed unique_id json files
return : 
        dict {metadata_file_list : [files],  unique_uid_list : [u_ids], processed_unique_id_list : [updated list]}
'''



vector_len = {'call_logs' : len(vector_desc_call_logs),
              'photo_gallery' : len(vector_desc_photo_gallery),
              'install_apps' : len(vector_desc_installed_apps)}
thd = {'call_logs' : 300, 'photo_gallery' : 200, 'install_apps' : 15, 'sample_days' : 30}


'''
    call the apps_installed_vector_descriptor with sorted dataframe
    and latitude , longitude GPS coordinates to get the app installed vector values   
'''


def create_app_install_vector(uid, df_dict):
    key = uid + '_InstallApps'
    df = df_dict.get(key) if key in df_dict.keys() else pd.DataFrame({'empty': []})
    score_vec = [0] * vector_len['install_apps']
    print('install_apps: ', len(df))
    if not df.empty:
        df0 = df.sort_values('INSTALL_DATETIME')
        mask1 = len(df0) > thd['install_apps']
        if mask1:
            score_vec = apps_installed_vector_descriptor(df=df0)
    return score_vec


def create_image_gallery_vector(uid, df_dict, lat_long):
    key = uid + '_ImgMetaData'
    df = df_dict.get(key) if key in df_dict.keys() else pd.DataFrame({'empty' : []})
    score_vec = [0] * (vector_len['photo_gallery'] + vector_len['install_apps'])
    print('photo-gallery: ', len(df))
    if not df.empty:
        df0 = df.sort_values('IMAGE_DATE_TIME')
        df0 = df0.reset_index(drop=True)
        days = np.abs(calc_number_of_days(df0, 'IMAGE_DATE_TIME'))
        mask1 = days >= thd['sample_days'] and len(df) >= thd['photo_gallery']
        if mask1:
            n_nan = df['IMAGE_TYPE'].isnull().sum()
            mask0 = float(n_nan / len(df)) < 0.5
            app_vector = create_app_install_vector(uid, df_dict)
            score_vec = photo_gallery_vector_descriptor(df=df0,lat_long=lat_long) + app_vector  if mask0 else score_vec
    return pd.Series(score_vec, name=uid)


def score_vector_for_init_metadata(uid, df_dict, lat_long):
    key = uid+'_CallLogs'
    call_logs_score_vector = [0] * vector_len['call_logs']
    df = df_dict.get(key) if key in df_dict.keys() else pd.DataFrame({'empty' : []})
    print('call-logs: ', len(df))
    if not df.empty:
        df0 = df.sort_values(by='CALL_DATE_TIME', ascending=True)
        days = np.abs(calc_number_of_days(df0, 'CALL_DATE_TIME'))
        mask1 = days >= 60 and len(df) >= 30 or days >= thd['sample_days'] and len(df) >= thd['call_logs']
        if mask1:
            n_nan = df['CALL_TYPE'].isnull().sum()
            n_zero = (df['CALL_DURATION'] == '0').sum()
            mask0 = float(n_nan / len(df)) < 0.5 or float(n_zero / len(df)) < 0.5
            call_logs_score_vector = call_logs_vector_descriptor(df=df0, lat_long=lat_long) if mask0 else call_logs_score_vector
    return pd.Series(call_logs_score_vector, name=uid)

'''
    run base on pre-grouped (uid, size, dictionary of raw-data by field) list of tuples 
'''


def run_score_vector(uid, raw_data, flag):
    print(uid)
    score_vector = [0] * vector_len['call_logs'] if flag == 'call_logs' else [0] * (vector_len['photo_gallery'] + vector_len['install_apps'])
    loc_dict, uid_df_dict = create_df_from_init_metadata(uid=uid, raw_data_json=raw_data)
    if not 'empty' in uid_df_dict.keys():
        loc_tuple = (loc_dict[0]['Latitude'], loc_dict[0]['Longitude'])  # if loc_dict[0] and loc_dict[0]['Latitude'] and loc_dict[0]['Longitude'] else (-1.0, -1.0)
        if flag == 'call-logs':
            score_vector = score_vector_for_init_metadata(uid, uid_df_dict, loc_tuple)
        else: #elif flag == 'others':
            score_vector = create_image_gallery_vector(uid, uid_df_dict, loc_tuple)
        s = '-> processed' if score_vector.any() else '-> to small to process'
        print(s)
    else:
        score_vector = pd.Series(score_vector, name=uid)
        print('-> json to small to process')
    return score_vector


'''
    for each user calc unique_id, file_size, and score_vector 
'''

def score_vector_constructor(path, flag):
    score_vector_dict = {}
    for file in os.listdir(path):
        if file.endswith('.json'):
            unique_id = file.split('_')[0]
            raw_data = json.load(codecs.open(path + file, 'r', 'utf-8-sig'))
            vscore = run_score_vector(uid=unique_id, raw_data=raw_data, flag=flag)
            if vscore.any():
                score_vector_dict[vscore.name] = vscore
    df0 = pd.concat(score_vector_dict, axis=1)
    if flag == 'call-logs':
        df0['description'] = vector_desc_call_logs  # + vector_desc_photo_gallery + vector_desc_installed_apps
    else: # elif flag == 'others':
        df0['description'] = vector_desc_photo_gallery + vector_desc_installed_apps
    dft = df0.set_index('description').transpose()
    print(dft.shape)
    return dft


def score_vector_from_bucket(object_storage_client, flag, start_str):
    score_vector_dict = {}
    counter = 0
    object_list = object_storage_client.list_objects(namespace, bucket_name, start=start_str, fields='name, timeCreated, size')
    for f in object_list.data.objects:
        if f.name.endswith('.json'):
            obj = object_storage_client.get_object(namespace, bucket_name, f.name).data # accept: 'text/json'
            raw_data = json.loads(obj.content)
            uid = f.name.split('_')[0]
            vscore = run_score_vector(uid=uid, raw_data=raw_data, flag=flag)
            if vscore.any():
                score_vector_dict[vscore.name] = vscore
            counter = counter + 1
            print(counter)
    df0 = pd.concat(score_vector_dict, axis=1)
    if flag == 'call-logs':
        df0['description'] = vector_desc_call_logs  # + vector_desc_photo_gallery + vector_desc_installed_apps
    else: #elif flag == 'photo-gallery':
        df0['description'] = vector_desc_photo_gallery #+ vector_desc_installed_apps
    dft = df0.set_index('description').transpose()
    print(dft.shape)
    return dft


def normalize_scores(df, method):
    df_scale = df.transpose()
    if len(df) > 20:
        df_scale =  z_score(df_scale) if method == 'z-score' else minmax_scale(df_scale) # 'minmax'
    return df_scale.transpose()