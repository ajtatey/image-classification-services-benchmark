import boto3
import csv
import sys
import os


def upload_folder(dataset, ablation):
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    with open(f'data/{dataset}/train_aws_{ablation}.csv') as csvfile:
        reader = csv.reader(csvfile)
        # get the class labels from classes.txt

        for row in reader:
            s3.upload_file(f'data/{dataset}/train/{row[1]}/{row[0]}',
                           AWS_STORAGE_BUCKET_NAME, f'train/{row[1]}/{row[0]}')

    with open(f'data/{dataset}/ablations/val_aws_{ablation}.csv') as csvfile:
        reader = csv.reader(csvfile)
        # get the class labels from classes.txt

        for row in reader:
            s3.upload_file(f'data/{dataset}train/{row[1]}/{row[0]}',
                           AWS_STORAGE_BUCKET_NAME, f'train/{row[1]}/{row[0]}')

    with open(f'data/{dataset}/{dataset}_test_aws.csv') as csvfile:
        reader = csv.reader(csvfile)
        # get the class labels from classes.txt

        for row in reader:
            s3.upload_file(f'data/{dataset}/test/{row[1]}/{row[0]}',
                           AWS_STORAGE_BUCKET_NAME, f'test/{row[1]}/{row[0]}')


if __name__ == '__main__':
    if sys.argv[1] == 'upload':
        dataset = sys.argv[2]
        ablation = sys.argv[5]
        upload_folder(dataset, ablation)
