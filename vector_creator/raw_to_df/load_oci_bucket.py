import oci
from oci.config import validate_config

namespace = "lrzustouvvrg"
bucket_name = "sdk-initial-data-bucket"

def load_buckets_metadata():
    tenancyId = "ocid1.tenancy.oc1..aaaaaaaahkzl4rzvgxjfnrfcj7rzlsdav25h7tsx7kjxyv2bmwkfpzpag26q"  # Your tenancies OCID.
    authUserId = "ocid1.user.oc1..aaaaaaaayltryoz33ole5eubwphwijkcvmi4ukbombh4aym2crp5j6xqxaxq"  # The OCID of the user ID being used.
    OCI_KEY_PATH = "/home/datascience/.oci/oci_api_key.pem"  # Path of the key file.
    keyFingerprint = "9a:47:38:96:c8:4d:bf:7e:32:3d:7c:c3:24:23:71:35"  # The fingerprint of the key file being used
    config = {
        "user": authUserId,
        "key_file": OCI_KEY_PATH,
        "fingerprint": keyFingerprint,
        "tenancy": tenancyId,
        "region": "uk-london-1"
    }
    # validates the above fields for connection
    validate_config(config)
    #
    object_storage_client = oci.object_storage.ObjectStorageClient(config)
    return object_storage_client

