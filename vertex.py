from google.cloud import storage, aiplatform
import os
import sys
import csv
import time
import pandas as pd
from create_tests import get_bucket_uris
import base64
from joblib import Parallel, delayed


from google.cloud.aiplatform.gapic.schema import predict
from google.oauth2 import service_account


def upload_to_bucket(blob_name, path_to_file, bucket_name, storage_client):
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_filename(path_to_file)
    return blob.public_url


def check_if_exists(blob_name, bucket_name, storage_client):
    bucket = storage_client.get_bucket(bucket_name)
    return storage.Blob(bucket=bucket, name=blob_name).exists(storage_client)


def get_storage_client():
    return storage.Client.from_service_account_json("gcreds.json")


def upload(dataset, google_bucket_name):
    storage_client = get_storage_client()

    for file in os.listdir(f"data/{dataset}/training_uploads"):
        public_url = upload_to_bucket(
            f"training_uploads/{file}", f"data/{dataset}/training_uploads/{file}", google_bucket_name, storage_client
        )
        print(public_url)

    for file in os.listdir(f"data/{dataset}/val_uploads"):
        public_url = upload_to_bucket(
            f"val_uploads/{file}", f"data/{dataset}/val_uploads/{file}", google_bucket_name, storage_client
        )
        print(public_url)

    for file in os.listdir(f"data/{dataset}/test_uploads"):
        print(file)
        if not check_if_exists(f"test_uploads/{file}", google_bucket_name, storage_client):
            public_url = upload_to_bucket(
                f"test_uploads/{file}", f"data/{dataset}/test_uploads/{file}", google_bucket_name, storage_client
            )
            print(public_url)


def predict_image_classification_sample(
    project: str,
    endpoint_id: str,
    filename: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    credentials = service_account.Credentials.from_service_account_file("gcreds.json")
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options, credentials=credentials)

    with open(filename, "rb") as f:
        file_content = f.read()

    # The format of each instance should conform to the deployed model's prediction input schema.
    encoded_content = base64.b64encode(file_content).decode("utf-8")
    instance = predict.instance.ImageClassificationPredictionInstance(
        content=encoded_content,
    ).to_value()
    instances = [instance]
    # See gs://google-cloud-aiplatform/schema/predict/params/image_classification_1.0.0.yaml for the format of the parameters.
    parameters = predict.params.ImageClassificationPredictionParams(
        confidence_threshold=0.1,
        max_predictions=5,
    ).to_value()
    endpoint = client.endpoint_path(project=project, location=location, endpoint=endpoint_id)
    # start timer to measure latency
    start = time.time()

    response = client.predict(endpoint=endpoint, instances=instances, parameters=parameters)
    # end timer
    end = time.time()
    latency = end - start
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # See gs://google-cloud-aiplatform/schema/predict/prediction/image_classification_1.0.0.yaml for the format of the predictions.
    predictions = response.predictions
    for prediction in predictions:
        # convert to dict and get displayNames and confidences
        prediction = dict(prediction)
        displayNames = prediction["displayNames"]
        confidences = prediction["confidences"]

    return displayNames, confidences, latency


# [END aiplatform_predict_image_classification_sample]


def invoke(dataset, ablationSize, project_id, endpoint_id):
    results = [[], [], [], [], []]
    if not os.path.exists(f"data/{dataset}/results"):
        os.makedirs(f"data/{dataset}/results")
    if os.path.exists(f"data/{dataset}/results/{dataset}-vertex-results-{str(ablationSize)}.csv"):
        df = pd.read_csv(f"data/{dataset}/results/{dataset}-vertex-results-{str(ablationSize)}.csv", header=None)
        results[0] = df[0].tolist()
        results[1] = df[1].tolist()
        results[2] = df[2].tolist()
        results[3] = df[3].tolist()
        results[4] = df[4].tolist()

    with open(f"data/{dataset}/{dataset}_test_vertex.csv") as csvfile:
        reader = csv.reader(csvfile)
        # get the class labels from classes.txt

        for row in reader:
            filename = row[0].split("/")[-1]
            print(filename)
            if filename in results[0]:
                print("already done")
                continue
            else:
                with open(f"data/{dataset}/test/{row[1]}/{filename}", "rb") as f:
                    # measure the latency for this request

                    displayNames, confidences, latency = predict_image_classification_sample(
                        project=project_id,
                        endpoint_id=endpoint_id,
                        location="us-central1",
                        filename=f"data/{dataset}/test/{row[1]}/{filename}",
                    )
                    if len(displayNames) == 0:
                        results[0].append(filename)
                        results[1].append(row[1])
                        results[2].append("none")
                        results[3].append(0.0)
                        results[4].append(latency)
                    else:
                        results[0].append(filename)
                        results[1].append(row[1])
                        results[2].append(displayNames[0])
                        results[3].append(confidences[0])
                        results[4].append(latency)

                print(filename, row[1], displayNames, confidences, latency)
                df = pd.DataFrame(results)
                df = df.transpose()
                df.to_csv(
                    f"data/{dataset}/results/{dataset}-vertex-results-{str(ablationSize)}.csv",
                    index=False,
                    header=False,
                )

def parallel_invoke(dataset, ablationSize, project_id, endpoint_id):
    

    filenames = []
    labels = []
    with open(f"data/{dataset}/{dataset}_test_vertex.csv") as csvfile:
        reader = csv.reader(csvfile)
        # get the class labels from classes.txt
        count = 0
        for row in reader:
            filenames.append(row[0].split("/")[-1])
            labels.append(row[1])
            count += 1
            if count == 1000:
                break

    def _parallel_invoke(filename: str, label: str, project_id: str, endpoint_id: str):

        displayNames, confidences, latency = predict_image_classification_sample(
                        project=project_id,
                        endpoint_id=endpoint_id,
                        location="us-central1",
                        filename=f"data/{dataset}/test/{label}/{filename}",
                    )

    start = time.time()
    Parallel(n_jobs=10, prefer="threads")(
        delayed(_parallel_invoke)(filename, label, project_id, endpoint_id)
        for filename, label in zip(filenames, labels)
    )
    end = time.time()
    print(f"Time to 1000 invokes {dataset}-{ablationSize}: {end - start}")
                    # measure the latency for this request

                    
                    


if __name__ == "__main__":
    if sys.argv[1] == "upload":
        dataset = sys.argv[2]
        google_bucket_name, _, _, _ = get_bucket_uris(dataset)
        google_bucket_name = google_bucket_name.split("/")[2]
        upload(dataset, google_bucket_name)
    elif sys.argv[1] == "invoke":
        dataset = sys.argv[2]
        ablation = sys.argv[3]
        project_id = sys.argv[4]
        endpoint_id = sys.argv[5]
        invoke(dataset, ablation, project_id, endpoint_id)
    elif sys.argv[1] == "parallel":
        dataset = sys.argv[2]
        ablation = sys.argv[3]
        project_id = sys.argv[4]
        endpoint_id = sys.argv[5]
        parallel_invoke(dataset, ablation, project_id, endpoint_id)
