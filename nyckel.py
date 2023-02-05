import requests
import os
import sys
import pandas as pd
import csv


def get_token():

    client_id = os.getenv('NYCKEL_CLIENT_ID')
    client_secret = os.getenv('NYCKEL_CLIENT_SECRET')

    token_url = 'https://www.nyckel.com/connect/token'
    data = {'client_id': client_id,
            'client_secret': client_secret, 'grant_type': 'client_credentials'}

    result = requests.post(token_url, data=data)
    return result.json()['access_token']


def create_function(access_token, function_name):

    url = 'https://www.nyckel.com/v1/functions'
    headers = {'Authorization': f'Bearer {access_token}'}

    result = requests.post(url, headers=headers, json={
                           "name": function_name, "input": "Image", "output": "Classification"})

    print(result.text)


def create_label(access_token, classes, function_id):

    url = f'https://www.nyckel.com/v1/functions/{function_id}/labels'
    headers = {'Authorization': f'Bearer {access_token}'}

    for cls in classes:
        result = requests.post(url, headers=headers, json={
            "name": cls})
    print(result.text)


def upload(access_token, function_id, ablationSize, dataset):

    training_file = f'data/{dataset}/ablations/{dataset}_train_nyckel_{ablationSize}.csv'

    with open(f'data/{dataset}/classes.txt') as f:
        classes = f.read().split(',')
    create_label(access_token, classes, function_id)

    url = f'https://www.nyckel.com/v1/functions/{function_id}/samples'
    headers = {'Authorization': f'Bearer {access_token}'}

    with open(training_file) as csvfile:
        reader = csv.reader(csvfile)
        # get the class labels from classes.txt

        for row in reader:
            print(row[0])
            with open(f'data/{dataset}/train/{row[1]}/{row[0]}', 'rb') as f:
                result = requests.post(url, headers=headers, files={'data': f}, data={
                    'annotation.labelName': row[1]})


def invoke(access_token, function_id, ablationSize, dataset):
    url = f'https://www.nyckel.com/v1/functions/{function_id}/invoke'

    headers = {'Authorization': f'Bearer {access_token}'}

    results = [[], [], [], []]
    accurate = 0
    total = 0
    with open(f'data/{dataset}/{dataset}_test_nyckel.csv') as csvfile:
        reader = csv.reader(csvfile)
        # get the class labels from classes.txt

        for row in reader:
            print(row[0])
            with open(f'data/{dataset}/test/{row[1]}/{row[0]}', 'rb') as f:
                # measure the latency for this request
                result = requests.post(url, headers=headers, files={'data': f})
                print(result.json())
                results[0].append(row[0])
                results[1].append(result.json()['labelName'])
                results[2].append(result.json()['confidence'])
                results[3].append(result.elapsed)
                # check whether the result.labelName is pneumonia
                if result.json()['labelName'] == row[1]:
                    accurate += 1
                    print('accurate')
                else:
                    print('inaccurate')
                total += 1

            df = pd.DataFrame(results)
            df = df.transpose()
            df.to_csv(f'data/{dataset}/{dataset}-nyckel-results-{str(ablationSize)}.csv',
                      index=False, header=False)
            print(f'Accuracy: {accurate/total}')
            print(f'Accurate: {accurate}')
            print(f'Total: {total}')


if __name__ == '__main__':
    access_token = get_token()
    if sys.argv[1] == 'create':
        function_name = sys.argv[2]
        create_function(access_token, function_name)
    elif sys.argv[1] == 'upload':
        dataset = sys.argv[2]
        function_id = sys.argv[3]
        ablationSize = sys.argv[4]
        upload(access_token, function_id, ablationSize, dataset)
    elif sys.argv[1] == 'invoke':
        dataset = sys.argv[2]
        function_id = sys.argv[3]
        ablationSize = sys.argv[4]
        invoke(access_token, function_id, ablationSize, dataset)
