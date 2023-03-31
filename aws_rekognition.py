import io
import boto3
import csv
import sys
import os
import logging
import time
import pandas as pd
from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm


from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def create_client(type):
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

    if type == "rekognition":
        print("creating rekognition client")
        return boto3.client(
            "rekognition",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name="us-west-2",
        )
    elif type == "s3":
        print("creating s3 client")
        return boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name="us-west-2",
        )


def create_bucket(bucket_name):
    s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": "us-west-2"})


def upload_folder(dataset, ablation, bucket_name, s3):
    if ablation != "test":
        with open(f"data/{dataset}/ablations/{dataset}_train_aws_{ablation}.csv") as csvfile:
            reader = csv.reader(csvfile)
            # get the class labels from classes.txt

            for row in reader:
                s3.upload_file(f"data/{dataset}/train/{row[1]}/{row[0]}", bucket_name, f"train/{row[1]}/{row[0]}")

        with open(f"data/{dataset}/ablations/{dataset}_val_aws_{ablation}.csv") as csvfile:
            reader = csv.reader(csvfile)
            # get the class labels from classes.txt

            for row in reader:
                s3.upload_file(f"data/{dataset}/train/{row[1]}/{row[0]}", bucket_name, f"val/{row[1]}/{row[0]}")
    else:
        with open(f"data/{dataset}/{dataset}_test_aws.csv") as csvfile:
            reader = csv.reader(csvfile)
            # get the class labels from classes.txt

            for row in reader:
                s3.upload_file(
                    f"data/{dataset}/test/{row[1]}/{row[0]}", f"argot-{bucket_name}", f"test/{row[1]}/{row[0]}"
                )


def analyze_local_image(rek_client, model, photo, min_confidence):
    """
    Analyzes an image stored as a local file.
    :param rek_client: The Amazon Rekognition Boto3 client.
    :param s3_connection: The Amazon S3 Boto3 S3 connection object.
    :param model: The ARN of the Amazon Rekognition Custom Labels model that you want to use.
    :param photo: The name and file path of the photo that you want to analyze.
    :param min_confidence: The desired threshold/confidence for the call.
    """

    try:
        logger.info("Analyzing local file: %s", photo)
        image = Image.open(photo)
        image_type = Image.MIME[image.format]

        if (image_type == "image/jpeg" or image_type == "image/png") is False:
            logger.error("Invalid image type for %s", photo)
            raise ValueError(f"Invalid file format. Supply a jpeg or png format file: {photo}")

        # get images bytes for call to detect_anomalies
        image_bytes = io.BytesIO()
        image.save(image_bytes, format=image.format)
        image_bytes = image_bytes.getvalue()
        start = time.time()

        response = rek_client.detect_custom_labels(
            Image={"Bytes": image_bytes}, MinConfidence=min_confidence, ProjectVersionArn=model
        )
        end = time.time()
        latency = end - start

        return response["CustomLabels"], latency

    except ClientError as client_err:
        logger.error(format(client_err))
        raise
    except FileNotFoundError as file_error:
        logger.error(format(file_error))
        raise


def invoke(dataset, ablationSize, model):
    results = [[], [], [], [], []]
    if not os.path.exists(f"data/{dataset}/results"):
        os.makedirs(f"data/{dataset}/results")
    if os.path.exists(f"data/{dataset}/results/{dataset}-aws-results-{str(ablationSize)}.csv"):
        df = pd.read_csv(f"data/{dataset}/results/{dataset}-aws-results-{str(ablationSize)}.csv", header=None)
        results[0] = df[0].tolist()
        results[1] = df[1].tolist()
        results[2] = df[2].tolist()
        results[3] = df[3].tolist()
        results[4] = df[4].tolist()

    rek_client = create_client("rekognition")

    with open(f"data/{dataset}/{dataset}_test_aws.csv") as csvfile:
        reader = csv.reader(csvfile)
        # get the class labels from classes.txt

        for row in reader:
            filename = row[0].split("/")[-1]
            print(filename)
            if filename in results[0]:
                print("already done")
                continue
            else:
                custom_labels, latency = analyze_local_image(
                    rek_client, model, f"data/{dataset}/test/{row[1]}/{filename}", 1
                )
                print(custom_labels)

                if len(custom_labels) == 0:
                    results[0].append(filename)
                    results[1].append(row[1])
                    results[2].append("none")
                    results[3].append(0.0)
                    results[4].append(latency)
                else:
                    results[0].append(filename)
                    results[1].append(row[1])
                    results[2].append(custom_labels[0]["Name"])
                    results[3].append(custom_labels[0]["Confidence"])
                    results[4].append(latency)
                df = pd.DataFrame(results)
                df = df.transpose()
                df.to_csv(
                    f"data/{dataset}/results/{dataset}-aws-results-{str(ablationSize)}.csv", index=False, header=False
                )


def parallel_invoke(dataset, ablationSize, model):
    

    rek_client = create_client("rekognition")

    filenames = []
    labels = []
    with open(f"data/{dataset}/{dataset}_test_aws.csv") as csvfile:
        reader = csv.reader(csvfile)
        # get the class labels from classes.txt
        count = 0
        for row in reader:
            filenames.append(row[0].split("/")[-1])
            labels.append(row[1])
            count += 1
            if count == 1000:
                break


    def _parallel_invoke(filename: str, label: str):

        custom_labels, latency = analyze_local_image(
                    rek_client, model, f"data/{dataset}/test/{label}/{filename}", 1
                )

    start = time.time()
    Parallel(n_jobs=10, prefer="threads")(
        delayed(_parallel_invoke)(filename, label)
        for filename, label in tqdm(zip(filenames, labels), total=len(labels))
    )
    end = time.time()
    print(f"Time to 1000 invokes {dataset}-{ablationSize}: {end - start}")

if __name__ == "__main__":
    if sys.argv[1] == "upload":
        dataset = sys.argv[2]
        ablation = sys.argv[3]

        bucket_name = f"{dataset}-{ablation}"
        s3 = create_client("s3")
        create_bucket(bucket_name)

        upload_folder(dataset, ablation, bucket_name, s3)
        print(f"s3://{bucket_name}/train/")
        print(f"s3://{bucket_name}/val/")
        print(f"s3://{bucket_name}/")
    elif sys.argv[1] == "invoke":
        dataset = sys.argv[2]
        ablation = sys.argv[3]
        model = sys.argv[4]
        invoke(dataset, ablation, model)
    elif sys.argv[1] == "parallel":
        dataset = sys.argv[2]
        ablation = sys.argv[3]
        model = sys.argv[4]
        parallel_invoke(dataset, ablation, model)
