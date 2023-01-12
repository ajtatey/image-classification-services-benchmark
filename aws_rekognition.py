import boto3
import csv
import sys


def upload_folder(ablation):
    # os.getenv('AWS_ACCESS_KEY_ID')
    AWS_ACCESS_KEY_ID = 'AKIAT434HFX5FQ2APAO6'
    # os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_SECRET_ACCESS_KEY = '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'
    # os.getenv('AWS_STORAGE_BUCKET_NAME')
    AWS_STORAGE_BUCKET_NAME = 'argot-chest-xrays'
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    with open(f'train_aws_{ablation}.csv') as csvfile:
        reader = csv.reader(csvfile)
        # get the class labels from classes.txt

        for row in reader:
            s3.upload_file(f'train/{row[1]}/{row[0]}',
                           AWS_STORAGE_BUCKET_NAME, f'train/{row[1]}/{row[0]}')

    with open(f'val_aws_{ablation}.csv') as csvfile:
        reader = csv.reader(csvfile)
        # get the class labels from classes.txt

        for row in reader:
            s3.upload_file(f'train/{row[1]}/{row[0]}',
                           AWS_STORAGE_BUCKET_NAME, f'train/{row[1]}/{row[0]}')

    with open('chest_xray_test_aws.csv') as csvfile:
        reader = csv.reader(csvfile)
        # get the class labels from classes.txt

        for row in reader:
            s3.upload_file(f'test/{row[1]}/{row[0]}',
                           AWS_STORAGE_BUCKET_NAME, f'test/{row[1]}/{row[0]}')


if __name__ == '__main__':
    if sys.argv[1] == 'upload':
        ablation = sys.argv[2]
        upload_folder(ablation)
