from google.cloud import storage
import os
import sys
from create_tests import get_bucket_uris


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


def upload(dataset, google_bucket_name):

    for file in os.listdir(f'data/{dataset}/training_uploads'):
        public_url = upload_to_bucket(
            file, f'data/{dataset}/training_uploads/{file}', google_bucket_name)
        print(public_url)

    for file in os.listdir(f'data/{dataset}/val_uploads'):
        public_url = upload_to_bucket(
            file, f'data/{dataset}/val_uploads/{file}', google_bucket_name)
        print(public_url)


if __name__ == '__main__':
    dataset = sys.argv[1]
    google_bucket_name, _, _, _ = get_bucket_uris(
        dataset)
    upload(dataset, google_bucket_name)
