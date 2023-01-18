import requests
import json
import pandas as pd
import sys
import csv
import os


def invoke(inference_endpoint, ablationSize):
    API_URL = f"https://api-inference.huggingface.co/models/{inference_endpoint}"
    access_token = os.getenv('HG_ACCESS_TOKEN')
    headers = {"Authorization": f"Bearer {access_token}"}
    options = {"wait_for_model": True}

    results = [[], [], [], []]
    accurate = 0
    total = 0
    with open(f'data/{dataset}/{dataset}_test_hg.csv') as csvfile:
        reader = csv.reader(csvfile)
        # get the class labels from classes.txt

        for row in reader:
            print(row[0])
            with open(f'data/{dataset}/test/{row[1]}/{row[0]}', 'rb') as f:
                data = f.read()
                response = requests.request(
                    "POST", API_URL, headers=headers, data=data, params=options)
                print(response.content.decode("utf-8"))
                results[0].append(row[1])
                results[1].append(json.loads(
                    response.content.decode("utf-8"))[0]['label'])
                results[2].append(json.loads(
                    response.content.decode("utf-8"))[0]['score'])
                results[3].append(response.elapsed)
                if json.loads(response.content.decode("utf-8"))[0]['label'] == row[1]:
                    accurate += 1
                    print('accurate')
                else:
                    print('inaccurate')
                total += 1

            df = pd.DataFrame(results)
            df = df.transpose()
            df.to_csv(f'data/{dataset}/hg-{dataset}-results-{str(ablationSize)}.csv',
                      index=False, header=False)
            print(f'Accuracy: {accurate/total}')
            print(f'Accurate: {accurate}')
            print(f'Total: {total}')


if __name__ == '__main__':

    if sys.argv[1] == 'invoke':
        inference_endpoint = sys.argv[2]
        dataset = sys.argv[3]
        ablationSize = sys.argv[4]
        invoke(inference_endpoint, dataset, ablationSize)
