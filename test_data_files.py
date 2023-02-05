import pytest
from collections import Counter

ablation_sizes_by_dataset = {
    "beans": [320, 80, 20, 5],
    "cars": [20, 5],
    "food": [320, 80, 20, 5],
    "intel": [1280, 320, 80, 20, 5],
    "pets": [80, 20, 5],
    "xrays": [1280, 320, 80, 20, 5],
}

n_classes_by_datasets = {
    "beans": 3,
    "cars": 196,
    "food": 101,
    "intel": 6,
    "pets": 37,
    "xrays": 2,
}

DATASETS = list(n_classes_by_datasets.keys())


@pytest.fixture(params=DATASETS)
def dataset(request):
    return request.param


def train_ablation(dataset, ablation_size):
    return f"data/{dataset}/ablations/{dataset}_train_{ablation_size}.csv"


def val_ablation(dataset, ablation_size):
    return f"data/{dataset}/ablations/{dataset}_val_{ablation_size}.csv"


def test_ablation_files_correct_size(dataset):
    ablation_sizes = ablation_sizes_by_dataset[dataset]
    n_classes = n_classes_by_datasets[dataset]
    for ablation_size_index in range(len(ablation_sizes)):
        with open(train_ablation(dataset, ablation_sizes[ablation_size_index])) as ablation_file:
            ablation_data_train = ablation_file.readlines()
        assert len(ablation_data_train) == n_classes * ablation_sizes[ablation_size_index] * 0.8

        with open(val_ablation(dataset, ablation_sizes[ablation_size_index])) as ablation_file:
            ablation_data_val = ablation_file.readlines()
        assert len(ablation_data_val) == n_classes * ablation_sizes[ablation_size_index] * 0.2


def test_no_split_mixing(dataset):
    ablation_sizes = ablation_sizes_by_dataset[dataset]
    with open(f"data/{dataset}/{dataset}_test_nyckel.csv") as ablation_file:
        ablation_data_test = ablation_file.readlines()
    for ablation_size_index in range(len(ablation_sizes)):
        with open(train_ablation(dataset, ablation_sizes[ablation_size_index])) as ablation_file:
            ablation_data_train = ablation_file.readlines()
        with open(val_ablation(dataset, ablation_sizes[ablation_size_index])) as ablation_file:
            ablation_data_val = ablation_file.readlines()
        for entry in ablation_data_train:
            assert entry not in ablation_data_test
            assert entry not in ablation_data_val

        for entry in ablation_data_val:
            assert entry not in ablation_data_test
            assert entry not in ablation_data_train


def test_ablations_are_cumulative(dataset):
    """Checks that larger ablations contain the data used in smaller ablations"""
    ablation_sizes = ablation_sizes_by_dataset[dataset]
    ablation_sizes.sort()
    for ablation_size_index in range(len(ablation_sizes) - 1):
        with open(train_ablation(dataset, ablation_sizes[ablation_size_index + 1])) as ablation_file:
            larger_ablation_data = ablation_file.readlines()
        with open(train_ablation(dataset, ablation_sizes[ablation_size_index])) as ablation_file:
            smaller_ablation_data = ablation_file.readlines()
        for entry in smaller_ablation_data:
            assert entry in larger_ablation_data


def test_ablations_are_balanced(dataset):
    ablation_sizes = ablation_sizes_by_dataset[dataset]
    for ablation_size_index in range(len(ablation_sizes)):
        with open(train_ablation(dataset, ablation_sizes[ablation_size_index])) as ablation_file:
            ablation_data = ablation_file.readlines()
        class_names = [entry.split(",")[1].rstrip().lstrip() for entry in ablation_data]
        counts = Counter(class_names)
        for count in counts.values():
            assert count == ablation_sizes[ablation_size_index] * 0.8


def test_nyckel_files(dataset):
    ablation_sizes = ablation_sizes_by_dataset[dataset]
    for ablation_size_index in range(len(ablation_sizes)):
        with open(train_ablation(dataset, ablation_sizes[ablation_size_index])) as ablation_file:
            ablation_data_train = ablation_file.readlines()
        with open(val_ablation(dataset, ablation_sizes[ablation_size_index])) as ablation_file:
            ablation_data_val = ablation_file.readlines()
        with open(
            f"data/{dataset}/ablations/{dataset}_train_nyckel_{ablation_sizes[ablation_size_index]}.csv"
        ) as ablation_file:
            ablation_data_nyckel = ablation_file.readlines()
        for entry in ablation_data_nyckel:
            assert entry in ablation_data_train + ablation_data_val


def test_vertex_files(dataset):
    ablation_sizes = ablation_sizes_by_dataset[dataset]
    for split in ["train", "val"]:
        for ablation_size_index in range(len(ablation_sizes)):
            with open(
                f"data/{dataset}/ablations/{dataset}_{split}_{ablation_sizes[ablation_size_index]}.csv"
            ) as ablation_file:
                ablation_data = ablation_file.readlines()
            with open(
                f"data/{dataset}/ablations/{dataset}_{split}_vertex_{ablation_sizes[ablation_size_index]}.csv"
            ) as ablation_file:
                vertex_data = ablation_file.readlines()
            vertex_data = [entry.removeprefix("gs://argot-xrays/training_uploads/") for entry in vertex_data]
            vertex_data = [entry.removeprefix("gs://argot-xrays/val_uploads/") for entry in vertex_data]
            assert set(vertex_data) == set(ablation_data), f"failed on {split} {ablation_size_index}"


def test_huggingface_files(dataset):
    ablation_sizes = ablation_sizes_by_dataset[dataset]
    for split in ["train", "val"]:
        for ablation_size_index in range(len(ablation_sizes)):
            with open(
                f"data/{dataset}/ablations/{dataset}_{split}_{ablation_sizes[ablation_size_index]}.csv"
            ) as ablation_file:
                ablation_data = ablation_file.readlines()
            with open(
                f"data/{dataset}/ablations/{dataset}_{split}_hg_{ablation_sizes[ablation_size_index]}.csv"
            ) as ablation_file:
                hf_data = ablation_file.readlines()
            hf_data = hf_data[1:]  # Pop header
            assert set(hf_data) == set(ablation_data), f"failed on {split} {ablation_size_index}"


def test_aws_files(dataset):
    ablation_sizes = ablation_sizes_by_dataset[dataset]
    for split in ["train", "val"]:
        for ablation_size_index in range(len(ablation_sizes)):
            with open(
                f"data/{dataset}/ablations/{dataset}_{split}_{ablation_sizes[ablation_size_index]}.csv"
            ) as ablation_file:
                ablation_data = ablation_file.readlines()
            with open(
                f"data/{dataset}/ablations/{dataset}_{split}_aws_{ablation_sizes[ablation_size_index]}.csv"
            ) as ablation_file:
                aws_data = ablation_file.readlines()
            assert set(aws_data) == set(ablation_data), f"failed on {split} {ablation_size_index}"


def test_azure_files(dataset):
    ablation_sizes = ablation_sizes_by_dataset[dataset]
    for split in ["train", "val"]:
        for ablation_size_index in range(len(ablation_sizes)):
            with open(
                f"data/{dataset}/ablations/{dataset}_{split}_{ablation_sizes[ablation_size_index]}.csv"
            ) as ablation_file:
                ablation_data = ablation_file.readlines()
            with open(
                f"data/{dataset}/ablations/{dataset}_{split}_azure_{ablation_sizes[ablation_size_index]}.csv"
            ) as ablation_file:
                azure_data = ablation_file.readlines()
            azure_data = azure_data[1:]  # Pop header
            azure_data = [entry.removeprefix("azureml://training_uploads/") for entry in azure_data]
            azure_data = [entry.removeprefix("azureml://val_uploads/") for entry in azure_data]
            assert set(azure_data) == set(ablation_data), f"failed on {split} {ablation_size_index}"
