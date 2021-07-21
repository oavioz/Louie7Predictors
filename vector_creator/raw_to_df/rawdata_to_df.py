import numpy as np
import pandas as pd
import datetime
import os
import fnmatch
import json
import codecs


df_init_fields =  {
    'CallLogs' : 'CALL_DATE_TIME',
    'ImgMetaData' : 'IMAGE_DATE_TIME',
    'InstallApps' : 'INSTALL_DATETIME'}


df_cont_fields = [('ScreenInfo', 'Sampling_Collect_Time'),
                  ('WifiInfo', 'Sampling_Collect_Time'),
                  ('BatteryInfo', 'Sampling_Collect_Time'),
                  ('LocationInfo', 'Sampling_Collect_Time'),
                  ('ActiveAppSamplingInfo', 'ActiveAppSamplingTime')]


def group_metadata(path):
    metadata_file_list = list_of_json_files(path)
    if not metadata_file_list:
        return {}
    unique_uid_set = set(map(lambda x: x.split('_')[0], metadata_file_list))
    return {'unique_ids' : list(unique_uid_set), 'file_list' : metadata_file_list}


def uid_df_init_metadata(uid, path, metadata_file_list):
    init_metadata_df = {}
    uid_file = list(filter(lambda x: x.startswith(uid), metadata_file_list))[0]
    raw_data = json.load(codecs.open(path + uid_file, 'r', 'utf-8-sig')) # convert uid's json to dict
    init_keys = df_init_fields.keys()
    for key in raw_data.keys():
        if key in init_keys:
            ts =  df_init_fields[key]
            df_name = uid + '_' + key
            df = uid_init_metadata_to_df(raw_data, key, ts)
            if key == 'ImgMetaData':
                df['IMAGE_TYPE'] = df['IMAGE_TYPE'].map(lambda x: x.split(sep='/')[1])
            init_metadata_df[df_name] = df
    loc_info = [{"Latitude" :-1.0, "Longitude": -1.0, "Sampling_Collect_Time": '00:00:00'}]
    if 'LocationInfo' in raw_data.keys():
        loc_info = raw_data['LocationInfo']
    return  loc_info, init_metadata_df


def uid_df_cont_metadata(uid, path, json_file_list,  cont_fields):
    cont_metadata_df = {}
    for tp in cont_fields:
        df_name = uid + '_' + tp[0] + '_df'
        df = uid_cont_metadata_to_df(uid, path, json_file_list, tp[0], tp[1])
        cont_metadata_df[df_name] = df
    return cont_metadata_df


def list_of_json_files(path):
    list_of_files = os.listdir(path)
    for entry in list_of_files:
        if not fnmatch.fnmatch(entry, '*.json'):
            list_of_files.remove(entry)
    return list_of_files


'''
    get initial raw data from JSON file by field name :
    call_logs{call_datetime, call_duration, call_type, call_number}, 
    gallery{image_type, image_datetime}, 
    application_list{install_datetime, app_category, app_variant} 
'''


def uid_init_metadata_to_df(raw_dict, field, ts):
    df =  pd.json_normalize(raw_dict, record_path=[field])
    # converting datetime timestamp from string to float to datetime
    if ts == 'IMAGE_DATE_TIME':
        sample = df[ts][0]
        s = sample.split(".")
        if len(s[0]) == 7:  # bad format
            df[ts] = df[ts].map(lambda x: x.replace('.', ''))
    df[ts] = pd.to_numeric(df[ts])
    df = df[df[ts] != 0.0]
    df[ts] = pd.to_datetime(df[ts], unit='s')
    return df


def uid_cont_metadata_to_df(u_id, path, cont_metadata_files, field, date_time):
    df_set = []
    user_id_files = list(filter(lambda x: x.startswith(u_id), cont_metadata_files))
    for uid_file in user_id_files:
        d = uid_file.split('_')[1]  # extract date
        df = load_metadata(path, uid_file, field)
        # update datetime field in df with ISO 8601
        date_obj = datetime.datetime.strptime(d, '%d-%m-%Y').strftime('%Y-%m-%d')
        df[date_time] = date_obj + ' ' + df[date_time]
        df[date_time] = pd.to_datetime(df[date_time], format='%Y-%m-%d %H:%M:%S')
        df['quantity'] = df.groupby(date_time)[field].transform('count')
        df = add_day_of_week(df, date_time)
        df_set.append(df)
    return pd.concat(df_set)


def add_day_of_week(df, date_time):
    df['day_of_week'] = (df[date_time]).dt.date
    df['day_of_week'] = pd.to_datetime(df['day_of_week']).dt.day_name()
    return df


'''
return raw_data file (init.cont by unique uid)
'''


def load_metadata(path, uid, field):
    raw_data = json.load(codecs.open(path + uid, 'r', 'utf-8-sig'))
    return pd.json_normalize(raw_data, record_path=[field])


def uid_init_metadata_to_df_simple(path, fname, field, ts):
    with open(path+fname) as initRawData:
        raw_data = json.load(initRawData)
    df = pd.json_normalize(raw_data, record_path=[field])
    df[ts] = pd.to_numeric(ts, downcast="float")
    df[ts] = pd.to_datetime(df[ts], unit='s')
    df = add_day_of_week(df, ts)
    return df
