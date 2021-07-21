import vector_creator.raw_to_df.rawdata_to_df as mt2df
from vector_creator.score_vectors.call_logs_score_vector import create_call_logs_vector_for_unique_id
from vector_creator.score_vectors.photo_gallery_score_vector import create_photo_gallery_vector_for_unique_id


# appsCategoriesSimple = ['Game', 'Audio', 'Video', 'Image', 'Social', 'News', 'Maps', 'Productivity']
appsSubCategoriesAndroid = ['Art & Design', 'Auto & Vehicles', 'Beauty', 'Books & Reference', 'Business', 'Comics', 'Communications',
                         'Dating', 'Education', 'Entertainment', 'Events', 'Finance', 'Food & Drink', 'Health & Fitness', 'House & Home',
                         'Lifestyle', 'Maps & Navigation', 'Medical', 'Music & Audio', 'News & Magazines', 'Parenting',
                         'Personalization', 'Photography', 'Productivity', 'Shopping', 'Social', 'Sports', 'Tools', 'Travel & Local',
                         'Video Players & Editors', 'Weather', 'Libraries & Demo', 'Action', 'Adventure', 'Arcade', 'Board', 'Cards',
                         'Casino', 'Casual', 'Educational', 'Music Games', 'Puzzle', 'Racing', 'Role Playing',
                         'Simulation', 'Sport Games', 'Strategy', 'Trivia', 'Word Games', 'Family All Ages', 'Family Action',
                         'Family Brain Games', 'Family Create', 'Family Education', 'Family Music & Video', 'Family Pretend Play']


'''
processed_unique_id_list : already processed unique_id json files
return : 
        dict {metadata_file_list : [files],  unique_uid_list : [u_ids], processed_unique_id_list : [updated list]}
'''
def file_list_with_unique_id(path, processed_unique_id_list):
    return mt2df.group_init_metadata(path, processed_unique_id_list)


'''
return :
        dict {sampling config},
        dict {init_metadata_field_name(CallLogs, ImgMetaData, InstallApps, LocationInfo) : df}
'''
def df_for_init_meta(uid, path, metadata_file_list):
    return mt2df.uid_df_init_metadata(uid, path, metadata_file_list)


def score_vector_init_metadata(uid, df, lat_long):
    call_logs_vector =  create_call_logs_vector_for_unique_id(uid=uid, df=df, lat_long=lat_long)
    photo_gallery_vector = create_photo_gallery_vector_for_unique_id(uid=uid, df=df, lat_long=lat_long)
    score_vector =  call_logs_vector.append(photo_gallery_vector)
    return score_vector
