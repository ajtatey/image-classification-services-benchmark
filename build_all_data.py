import fire

from fetch_dataset import fetch_dataset
from preprocessing import preprocessing_main
from create_ablations import create_ablations_main
from create_tests import create_test_main
from create_folders import create_folders

DATASETS = ["beans", "cars", "food", "intel", "pets", "xrays", "clothing"]


def build_one(dataset: str, force_redo_download: bool):
    print(f"\n====== BUILDING LOCAL DATA FILES FOR {dataset} ========\n")

    fetch_dataset(dataset, force_redo_download=force_redo_download)
    preprocessing_main(dataset)
    ablations = create_ablations_main(dataset)
    create_test_main(dataset)
    create_folders(dataset, ablations)


def build_all(force_redo_download=False):
    for dataset in DATASETS:
        build_one(dataset, force_redo_download)


if __name__ == "__main__":
    fire.Fire(build_all)
