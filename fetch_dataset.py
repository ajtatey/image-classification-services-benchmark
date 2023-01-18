from kaggle.api.kaggle_api_extended import KaggleApi
import sys
import requests
import os
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

# zipurl = 'https://drive.google.com/uc?export=download&id=1yAmFc15GtP52El_RTxl6uqmZZJi-h4BG'


def fetch_dataset(training_set):
    path = f'data/{training_set}/'
    if not os.path.exists(path):
        os.makedirs(path)
    if training_set == 'intel':
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            'puneet6060/intel-image-classification', path=path, unzip=True)
    elif training_set == 'beans':
        datasets = ['https://huggingface.co/datasets/beans/resolve/main/data/test.zip',
                    'https://huggingface.co/datasets/beans/resolve/main/data/train.zip',
                    'https://huggingface.co/datasets/beans/resolve/main/data/validation.zip']
    elif training_set == 'xrays':
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            'paultimothymooney/chest-xray-pneumonia', path=path, unzip=True)
    elif training_set == 'cars':
        datasets = ['http://ai.stanford.edu/~jkrause/car196/car_ims.tgz',
                    'http://ai.stanford.edu/~jkrause/car196/cars_annos.mat']
    elif training_set == 'pets':
        datasets = ['https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz',
                    'https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz']
    elif training_set == 'food':
        datasets = ['http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz']
    elif training_set == 'shopping':
        pass

    if training_set in ['beans', 'cars', 'pets', 'food']:
        for dataset in datasets:
            resp = requests.get(dataset)  # making requests to server
            # opening a file handler to create new file
            with open(f'{path}{dataset.split("/")[-1]}', "wb") as f:
                f.write(resp.content)  # writing content to file


if __name__ == '__main__':
    training_set = sys.argv[1]
    fetch_dataset(training_set)
