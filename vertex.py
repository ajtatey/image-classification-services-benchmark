from google.cloud import storage
import os
# pip install --upgrade google-cloud-storage.


def upload_to_bucket(blob_name, path_to_file, bucket_name):
    """ Upload data to a bucket"""

    # Explicitly use service account credentials by specifying the private key
    # file.
    storage_client = storage.Client.from_service_account_json(
        'gcreds.json')

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(path_to_file)

    # returns a public url
    return blob.public_url


dataset = 'gs://chest-xray'
for file in os.listdir('training_uploads'):
    public_url = upload_to_bucket(
        file, f'training_uploads/{file}', dataset)
    print(public_url)

for file in os.listdir('val_uploads'):
    public_url = upload_to_bucket(
        file, f'val_uploads/{file}', dataset)
    print(public_url)
