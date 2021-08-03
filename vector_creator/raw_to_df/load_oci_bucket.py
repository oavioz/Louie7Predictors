import oci
import json
from oci.config import validate_config

def load_buckets_metadata():
    tenancyId="louie7ai" # Your tenancies OCID.
    authUserId="" # The OCID of the user ID being used.
    OCI_KEY_PATH="filename"; # Path of the key file.
    keyFingerprint="12:13:14:15"; # The fingerprint of the key file being used
    namespace = "lrzustouvvrg"
    bucket_name = "sdk-initial-data-bucket"
    config = {
        "user": authUserId,
        "key_file": OCI_KEY_PATH,
        "fingerprint": keyFingerprint,
        "tenancy": tenancyId,
        "region": "uk-london-1"
    }
    # validates the above fields for connection
    validate_config(config)
    '''
    1. prefix and fields are optional parameter.
    2. prefix is for filename pattern but not a regex
    3. fields valid values - md5,name,timeCreated,size
    '''
    object_storage_client = oci.object_storage.ObjectStorageClient(config)
    object_list = object_storage_client.list_objects(namespace, bucket_name , fields='name, timeCreated, size')
    list_of_files = []
    for f in object_list.data.objects:
        obj = object_storage_client.get_object(namespace, bucket_name, f.name).data
        raw_data = json.loads(obj) # json.load(codecs.open(f.name, 'r', 'utf-8-sig'))
        uid = f.name.split('_')[0]
        print(f.name, f.size)
        list_of_files.append((uid, f.size, raw_data))
    return list_of_files
