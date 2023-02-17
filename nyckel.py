import csv
import os
import sys

import pandas as pd
import requests
from tqdm import tqdm
from joblib import Parallel, delayed


def get_token():

    client_id = os.getenv("NYCKEL_CLIENT_ID")
    client_secret = os.getenv("NYCKEL_CLIENT_SECRET")

    token_url = "https://www.nyckel.com/connect/token"
    data = {"client_id": client_id, "client_secret": client_secret, "grant_type": "client_credentials"}

    result = requests.post(token_url, data=data)
    return result.json()["access_token"]


def create_function(access_token, function_name):

    url = "https://www.nyckel.com/v1/functions"
    headers = {"Authorization": f"Bearer {access_token}"}

    result = requests.post(
        url, headers=headers, json={"name": function_name, "input": "Image", "output": "Classification"}
    )

    print(result.text)


def create_label(access_token, classes, function_id):

    url = f"https://www.nyckel.com/v1/functions/{function_id}/labels"
    headers = {"Authorization": f"Bearer {access_token}"}

    print("Posting labels ...")
    for cls in tqdm(classes):
        response = requests.post(url, headers=headers, json={"name": cls})
        if not response.status_code == 200:
            raise RuntimeError(f"Invalid response {response.text=} {response.status_code=}")
    url = f"https://www.nyckel.com/v1/functions/{function_id}/labels/?batchSize=200"
    response = requests.get(url, headers=headers)
    print(f"Created {len(response.json())} labels for function: {function_id}")


def upload(access_token, function_id, ablationSize, dataset):
    url = f"https://www.nyckel.com/v1/functions/{function_id}/samples"
    headers = {"Authorization": f"Bearer {access_token}"}

    def _post_annotated_image(filename: str, label: str):
        with open(f"data/{dataset}/train/{label}/{filename}", "rb") as f:
            response = requests.post(url, headers=headers, files={"data": f}, data={"annotation.labelName": label})
            if not response.status_code == 200:
                print(f"Invalid response {response.text=} {response.status_code=} {filename=} {label=}")

    training_file = f"data/{dataset}/ablations/{dataset}_train_nyckel_{ablationSize}.csv"

    with open(f"data/{dataset}/classes.txt") as f:
        classes = f.read().split(",")
    create_label(access_token, classes, function_id)

    filenames = []
    labels = []
    with open(training_file) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            filenames.append(row[0])
            labels.append(row[1])

    print("Posting samples ...")
    Parallel(n_jobs=10, prefer="threads")(
        delayed(_post_annotated_image)(filename, label)
        for filename, label in tqdm(zip(filenames, labels), total=len(labels))
    )


def invoke(access_token, function_id, ablationSize, dataset):
    url = f"https://www.nyckel.com/v1/functions/{function_id}/invoke"

    headers = {"Authorization": f"Bearer {access_token}"}

    results = [[], [], [], [], []]
    if not os.path.exists(f"data/{dataset}/results"):
        os.makedirs(f"data/{dataset}/results")
    if os.path.exists(f"data/{dataset}/results/{dataset}-nyckel-results-{str(ablationSize)}.csv"):
        df = pd.read_csv(f"data/{dataset}/results/{dataset}-nyckel-results-{str(ablationSize)}.csv", header=None)
        results[0] = df[0].tolist()
        results[1] = df[1].tolist()
        results[2] = df[2].tolist()
        results[3] = df[3].tolist()
        results[4] = df[4].tolist()

    accurate = 0
    total = 0
    with open(f"data/{dataset}/{dataset}_test_nyckel.csv") as csvfile:
        reader = csv.reader(csvfile)
        # get the class labels from classes.txt

        for row in reader:
            print(row[0])
            if row[0] in results[0]:
                print("already done")
                continue
            else:
                with open(f"data/{dataset}/test/{row[1]}/{row[0]}", "rb") as f:
                    # measure the latency for this request
                    result = requests.post(url, headers=headers, files={"data": f})
                    print(result.json())
                    results[0].append(row[0])
                    results[1].append(row[1])
                    results[2].append(result.json()["labelName"])
                    results[3].append(result.json()["confidence"])
                    results[4].append(result.elapsed)
                    # check whether the result.labelName is pneumonia
                    if result.json()["labelName"] == row[1]:
                        accurate += 1
                        print("accurate")
                    else:
                        print("inaccurate")
                    total += 1

                df = pd.DataFrame(results)
                df = df.transpose()
                df.to_csv(
                    f"data/{dataset}/results/{dataset}-nyckel-results-{str(ablationSize)}.csv",
                    index=False,
                    header=False,
                )
                print(f"Accuracy: {accurate/total}")
                print(f"Accurate: {accurate}")
                print(f"Total: {total}")


if __name__ == "__main__":
    access_token = get_token()
    if sys.argv[1] == "create":
        function_name = sys.argv[2]
        create_function(access_token, function_name)
    elif sys.argv[1] == "upload":
        dataset = sys.argv[2]
        function_id = sys.argv[3]
        ablationSize = sys.argv[4]
        upload(access_token, function_id, ablationSize, dataset)
    elif sys.argv[1] == "invoke":
        dataset = sys.argv[2]
        function_id = sys.argv[3]
        ablationSize = sys.argv[4]
        invoke(access_token, function_id, ablationSize, dataset)
