import functools
import os
import pathlib
import shutil
import sys
from typing import List

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm.auto import tqdm

zipurl = "https://drive.google.com/uc?export=download&id=1yAmFc15GtP52El_RTxl6uqmZZJi-h4BG"


def download(url, filename):

    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get("Content-Length", 0))

    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    desc = "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)

    return path


def fetch_from_www(base_path, data_urls: List[str], force_redo_download):
    for data_url in data_urls:
        target_file = f"{base_path}{data_url.split('/')[-1]}"
        if not os.path.exists(target_file) or force_redo_download:
            print(f"Fetching {data_url} from remote")
            download(data_url, target_file)


def fetch_from_kaggle(key: str, target_path: str, force_redo_download: bool):

    if not os.path.exists(target_path) or force_redo_download:
        print("Fetching {key} to {target_path}")
        os.makedirs(target_path, exist_ok=True)
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(key, path=target_path, unzip=True)


def fetch_dataset(training_set, force_redo_download=False):
    path = f"data/{training_set}/"
    if training_set == "intel":
        fetch_from_kaggle("puneet6060/intel-image-classification", path)
    if training_set == "xrays":
        fetch_from_kaggle("paultimothymooney/chest-xray-pneumonia", path)

    if training_set == "beans":
        fetch_from_www(
            path,
            [
                "https://huggingface.co/datasets/beans/resolve/main/data/test.zip",
                "https://huggingface.co/datasets/beans/resolve/main/data/train.zip",
                "https://huggingface.co/datasets/beans/resolve/main/data/validation.zip",
            ],
            force_redo_download,
        )

    if training_set == "cars":
        fetch_from_www(
            path,
            [
                "http://ai.stanford.edu/~jkrause/car196/car_ims.tgz",
                "http://ai.stanford.edu/~jkrause/car196/cars_annos.mat",
            ],
            force_redo_download,
        )
    if training_set == "pets":
        fetch_from_www(
            path,
            [
                "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz",
                "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz",
            ],
            force_redo_download,
        )
    if training_set == "food":
        fetch_from_www(
            path,
            ["http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"],
            force_redo_download,
        )
    if training_set == "shopping":
        pass


if __name__ == "__main__":
    training_set = sys.argv[1]
    fetch_dataset(training_set)
