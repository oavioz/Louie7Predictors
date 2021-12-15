from vector_creator.raw_to_df.rawdata_to_df import create_df_from_init_metadata
from vector_creator.score_vectors.vector_descriptor import *
from vector_creator.score_vectors.vector_indexer import *
from vector_creator.preprocess.utils import calc_number_of_days
import pandas as pd
import numpy as np
import os
import json
import codecs
import shutil
import multiprocessing as mp
from functools import partial


'''
processed_unique_id_list : already processed unique_id json files
return : 
        dict {metadata_file_list : [files],  unique_uid_list : [u_ids], processed_unique_id_list : [updated list]}
'''



vector_len = {'call_logs' : len(vector_desc_call_logs),
              'photo_gallery' : len(vector_desc_photo_gallery),
              'install_apps' : len(vector_desc_installed_apps)}
thd = {'call_logs' : 300, 'photo_gallery' : 20, 'install_apps' : 10, 'sample_days' : 7}


'''
    call the apps_installed_vector_descriptor with sorted dataframe
    and latitude , longitude GPS coordinates to get the app installed vector values   
'''


def create_app_install_vector(uid, df_dict):
    key = uid + '_InstallApps'
    df = df_dict.get(key) if key in df_dict.keys() else pd.DataFrame({'empty': []})
    score_vec = [0] * vector_len['install_apps']
    #print('install_apps: ', len(df))
    if not df.empty:
        df0 = df.sort_values('INSTALL_DATETIME')
        mask1 = len(df0) > thd['install_apps']
        if mask1:
            score_vec = apps_installed_vector_descriptor(df=df0)
    return score_vec


def create_image_gallery_vector(uid, df_dict, lat_long):
    key = uid + '_ImgMetaData'
    df = df_dict.get(key) if key in df_dict.keys() else pd.DataFrame({'empty' : []})
    score_vec = [0] * vector_len['photo_gallery']
    app_vec = [0] * vector_len['install_apps']
    if not df.empty:
        df0 = df.sort_values('IMAGE_DATE_TIME')
        df0 = df0.reset_index(drop=True)
        days = np.abs(calc_number_of_days(df0, 'IMAGE_DATE_TIME'))
        mask1 = days >= thd['sample_days'] and len(df) >= thd['photo_gallery']
        app_vec = create_app_install_vector(uid, df_dict)
        if mask1:
            n_nan = df['IMAGE_TYPE'].isnull().sum()
            mask0 = float(n_nan / len(df)) < 0.95
            score_vec = photo_gallery_vector_descriptor(df=df0,lat_long=lat_long) if mask0 else score_vec
    score_vec += app_vec
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


def run_score_vector(uid, raw_data):
    #print(uid)
    score_vector = [0] * (vector_len['photo_gallery'] + vector_len['install_apps'])
    loc_dict, uid_df_dict = create_df_from_init_metadata(uid=uid, raw_data_json=raw_data)
    if not 'empty' in uid_df_dict.keys():
        loc_tuple = (loc_dict[0]['Latitude'], loc_dict[0]['Longitude'])  # if loc_dict[0] and loc_dict[0]['Latitude'] and loc_dict[0]['Longitude'] else (-1.0, -1.0)
        score_vector = create_image_gallery_vector(uid, uid_df_dict, loc_tuple)
    else:
        score_vector = pd.Series(score_vector, name=uid)
    return score_vector


'''
    for each user calc unique_id, file_size, and score_vector 
'''


def score_vector_creator(f, p):
    unique_id = f.split('_')[0]
    raw_data = json.load(codecs.open(p + f, 'r', 'utf-8-sig'))
    return run_score_vector(uid=unique_id, raw_data=raw_data)


def score_vector_constructor(path, procs):
    score_vector_dict = {}
    filter_list = list(filter(lambda x : x.endswith('.json'), os.listdir(path)))
    pool = mp.Pool(processes=procs)
    vector_creator = partial(score_vector_creator, p=path)
    v_scores = pool.map(vector_creator, filter_list)
    for v_score in v_scores:
        score_vector_dict[v_score.name] = v_score
    df0 = pd.concat(score_vector_dict, axis=1)
    df0['description'] = vector_desc_photo_gallery + vector_desc_installed_apps
    dft = df0.set_index('description').transpose()
    print(dft.shape)
    return dft


def score_vector_creator2(f, client_params):
    object_storage_client = client_params[0]
    namespace = client_params[1]
    bucket_name = client_params[2]
    if f.name.endswith('.json'):
        obj = object_storage_client.get_object(namespace, bucket_name, f.name).data # accept: 'text/json'
        raw_data = json.loads(obj.content)
        uid = f.name.split('_')[0]
        return run_score_vector(uid=uid, raw_data=raw_data)
    else:
        return pd.Series([], dtype=float)

def score_vector_from_bucket(object_storage_client, namespace, bucket_name, start_str, procs):
    score_vector_dict = {}
    object_list = object_storage_client.list_objects(namespace, bucket_name, start=start_str, fields='name, timeCreated, size')
    pool = mp.Pool(processes=procs)
    score_vec = partial(score_vector_creator2, client_params=(object_storage_client, namespace, bucket_name))
    v_scores = pool.map(score_vec, object_list.data.objects)
    ne_scores = list(filter(lambda x : not x.empty, v_scores))
    for v_score in ne_scores:
        score_vector_dict[v_score.name] = v_score
    df0 = pd.concat(score_vector_dict, axis=1)
    df0['description'] = vector_desc_photo_gallery + vector_desc_installed_apps
    dft = df0.set_index('description').transpose()
    print(dft.shape)
    return dft

