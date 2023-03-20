import requests
import json
import pandas as pd
import sys
import csv
import os
import time
from joblib import Parallel, delayed
from tqdm import tqdm


def invoke(inference_endpoint, dataset, ablationSize):
    # API_URL = f"https://api-inference.huggingface.co/models/{inference_endpoint}"
    API_URL = inference_endpoint
    access_token = os.getenv("HG_ACCESS_TOKEN")
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "image/jpeg"}

    results = [[], [], [], [], []]
    if not os.path.exists(f"data/{dataset}/results"):
        os.makedirs(f"data/{dataset}/results")
    if os.path.exists(f"data/{dataset}/results/{dataset}-hg-results-{str(ablationSize)}.csv"):
        df = pd.read_csv(f"data/{dataset}/results/{dataset}-hg-results-{str(ablationSize)}.csv", header=None)
        results[0] = df[0].tolist()
        results[1] = df[1].tolist()
        results[2] = df[2].tolist()
        results[3] = df[3].tolist()
        results[4] = df[4].tolist()

    accurate = 0
    total = 0
    with open(f"data/{dataset}/{dataset}_test_hg.csv") as csvfile:
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
                    data = f.read()
                    # retry the request if it fails

                    response = requests.request("POST", API_URL, headers=headers, data=data)

                    if response.status_code != 200:
                        print(f"Request failed: {response.content.decode('utf-8')}")
                        print("Retrying in 20 seconds")
                        time.sleep(20)
                        response = requests.request("POST", API_URL, headers=headers, data=data)

                    results[0].append(row[0])
                    results[1].append(row[1])
                    results[2].append(json.loads(response.content.decode("utf-8"))[0]["label"])
                    results[3].append(json.loads(response.content.decode("utf-8"))[0]["score"])
                    results[4].append(response.elapsed)
                    if json.loads(response.content.decode("utf-8"))[0]["label"] == row[1]:
                        accurate += 1
                        print("accurate")
                    else:
                        print("inaccurate")
                    total += 1

                df = pd.DataFrame(results)
                df = df.transpose()
                df.to_csv(
                    f"data/{dataset}/results/{dataset}-hg-results-{str(ablationSize)}.csv", index=False, header=False
                )
                print(f"Accuracy: {accurate/total}")
                print(f"Accurate: {accurate}")
                print(f"Total: {total}")


def parallel_invoke(inference_endpoint, dataset, ablationSize):
    API_URL = inference_endpoint
    access_token = os.getenv("HG_ACCESS_TOKEN")
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "image/jpeg"}

    filenames = []
    labels = []
    with open(f"data/{dataset}/{dataset}_test_hg.csv") as csvfile:
        reader = csv.reader(csvfile)
        # get the class labels from classes.txt
        count = 0
        for row in reader:
            filenames.append(row[0])
            labels.append(row[1])
            count += 1
            if count == 1000:
                break

    def _parallel_invoke(filename: str, label: str):
        with open(f"data/{dataset}/test/{label}/{filename}", "rb") as f:
            data = f.read()
            response = response = requests.post(API_URL, headers=headers, data=data)
            if not response.status_code == 200:
                print(f"Invalid response {response.text=} {response.status_code=} {filename=} {label=}")

    start = time.time()
    Parallel(n_jobs=10, prefer="threads")(
        delayed(_parallel_invoke)(filename, label)
        for filename, label in tqdm(zip(filenames, labels), total=len(labels))
    )
    end = time.time()
    print(f"Time to 1000 invokes {dataset}-{ablationSize}: {end - start}")


if __name__ == "__main__":
    if sys.argv[1] == "invoke":
        inference_endpoint = sys.argv[2]
        dataset = sys.argv[3]
        ablationSize = sys.argv[4]
        invoke(inference_endpoint, dataset, ablationSize)
    elif sys.argv[1] == "parallel":
        inference_endpoint = sys.argv[2]
        dataset = sys.argv[3]
        ablationSize = sys.argv[4]
        parallel_invoke(inference_endpoint, dataset, ablationSize)
