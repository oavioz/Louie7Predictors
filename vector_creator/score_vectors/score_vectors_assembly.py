from vector_creator.raw_to_df.rawdata_to_df import create_df_from_init_metadata
from vector_creator.raw_to_df.load_oci_bucket import namespace, bucket_name
from vector_creator.score_vectors.vector_descriptor import apps_installed_vector_descriptor, photo_gallery_vector_descriptor, call_logs_vector_descriptor
from vector_creator.score_vectors.vector_indexer import vector_desc_call_logs, vector_desc_photo_gallery, vector_desc_installed_apps
from vector_creator.preprocess.utils import calc_number_of_days
import pandas as pd
import numpy as np
import os
import json
import codecs


'''
processed_unique_id_list : already processed unique_id json files
return : 
        dict {metadata_file_list : [files],  unique_uid_list : [u_ids], processed_unique_id_list : [updated list]}
'''



vector_len = {'call_logs' : len(vector_desc_call_logs),
              'photo_gallery' : len(vector_desc_photo_gallery),
              'install_apps' : len(vector_desc_installed_apps)}
thd = {'call_logs' : 50, 'photo_gallery' : 100, 'install_apps' : 15, 'sample_days' : 21}


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


def score_vector_for_init_metadata(uid, df_dict, lat_long):
    key = uid+'_CallLogs'
    call_logs_score_vector = [0] * vector_len['call_logs']
    df = df_dict.get(key) if key in df_dict.keys() else pd.DataFrame({'empty' : []})
    print('call-logs: ', len(df))
    if not df.empty:
        df0 = df.sort_values(by='CALL_DATE_TIME', ascending=True)
        days = np.abs(calc_number_of_days(df0, 'CALL_DATE_TIME'))
        mask = days >= 60 and len(df0) >= 10 or  days >= thd['sample_days'] and len(df) >= thd['call_logs']
        print('days : ', days)
        call_logs_score_vector = call_logs_vector_descriptor(df=df0, lat_long=lat_long) if mask else call_logs_score_vector

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

'''
    run base on pre-grouped (uid, size, dictionary of raw-data by field) list of tuples 
'''


def run_score_vector(uid, raw_data):
    score_vector = [0] * vector_len['call_logs']
    loc_dict, uid_df_dict = create_df_from_init_metadata(uid=uid, raw_data_json=raw_data)
    if not 'empty' in uid_df_dict.keys():
        loc_tuple = (loc_dict[0]['Latitude'], loc_dict[0]['Longitude']) if loc_dict[0] and loc_dict[0]['Latitude'] and loc_dict[0]['Longitude'] else (-1.0, -1.0)
        score_vector = score_vector_for_init_metadata(uid, uid_df_dict, loc_tuple)
        str = ' processed' if score_vector.any() else ' call-logs to small to process'
        print(uid + str)
    else:
        score_vector = pd.Series(score_vector, name=uid)
        print(uid + ' json to small to process')
    return score_vector


'''
    for each user calc unique_id, file_size, and score_vector 
'''

def score_vector_constructor(path):
    score_vector_dict = {}
    for file in os.listdir(path):
        if file.endswith('.json'):
            unique_id = file.split('_')[0]
            #file_size = os.path.getsize(path + file)
            raw_data = json.load(codecs.open(path + file, 'r', 'utf-8-sig'))
            vscore = run_score_vector(uid=unique_id, raw_data=raw_data)
            if vscore.any():
                score_vector_dict[vscore.name] = vscore
    df = pd.concat(score_vector_dict, axis=1)
    df['description'] = vector_desc_call_logs  # + vector_desc_photo_gallery + vector_desc_installed_apps
    return df.set_index('description')


def score_vector_from_bucket(object_storage_client):
    score_vector_dict = {}
    counter = 0
    object_list = object_storage_client.list_objects(namespace, bucket_name, fields='name, timeCreated, size')
    for f in object_list.data.objects:
        if f.name.endswith('.json'):
            obj = object_storage_client.get_object(namespace, bucket_name, f.name).data # accept: 'text/json'
            raw_data = json.loads(obj.content)
            uid = f.name.split('_')[0]
            #f_size = int(f.size)
            vscore = run_score_vector(uid=uid, raw_data=raw_data)
            if vscore.any():
                score_vector_dict[vscore.name] = vscore
                counter = counter + 1
                print(counter)
    df = pd.concat(score_vector_dict, axis=1)
    df['description'] = vector_desc_call_logs  # + vector_desc_photo_gallery + vector_desc_installed_apps
    return df.set_index('description')