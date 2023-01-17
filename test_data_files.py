ablation_sizes = [5, 20, 80, 320, 1280]


def test_ablation_files_correct_size():
    for ablation_size_index in range(len(ablation_sizes)):
        with open(f"train_{ablation_sizes[ablation_size_index]}.csv") as ablation_file:
            ablation_data_train = ablation_file.readlines()
        assert len(ablation_data_train) == 2 * \
                   ablation_sizes[ablation_size_index] * 0.8

        with open(f"val_{ablation_sizes[ablation_size_index]}.csv") as ablation_file:
            ablation_data_val = ablation_file.readlines()
        assert len(ablation_data_val) == 2 * \
                   ablation_sizes[ablation_size_index] * 0.2


def test_no_split_mixing():
    with open(f"chest_xray_test_nyckel.csv") as ablation_file:
        ablation_data_test = ablation_file.readlines()
    for ablation_size_index in range(len(ablation_sizes)):
           with open(f"train_{ablation_sizes[ablation_size_index]}.csv") as ablation_file:
                ablation_data_train = ablation_file.readlines()
            with open(f"val_{ablation_sizes[ablation_size_index]}.csv") as ablation_file:
                ablation_data_val = ablation_file.readlines()
            for entry in ablation_data_train:
                assert entry not in ablation_data_test
                assert entry not in ablation_data_val

            for entry in ablation_data_val:
                assert entry not in ablation_data_test
                assert entry not in ablation_data_train


def test_ablations_are_cumulative():
    """Checks that larger ablations contain the data used in smaller ablations"""

    for ablation_size_index in range(len(ablation_sizes)-1):
        with open(f"train_{ablation_sizes[ablation_size_index+1]}.csv") as ablation_file:
            larger_ablation_data = ablation_file.readlines()
        with open(f"train_{ablation_sizes[ablation_size_index]}.csv") as ablation_file:
            smaller_ablation_data = ablation_file.readlines()
        for entry in smaller_ablation_data:
            assert entry in larger_ablation_data


def test_ablations_are_balanced():
    for ablation_size_index in range(len(ablation_sizes)):
        with open(f"train_{ablation_sizes[ablation_size_index]}.csv") as ablation_file:
            ablation_data = ablation_file.readlines()
        pneumonia_sample_count = sum(
            ["PNEUMONIA" in entry for entry in ablation_data])
        assert pneumonia_sample_count * 2 == len(ablation_data), ablation_data


def test_nyckel_files():
    for ablation_size_index in range(len(ablation_sizes)):
        with open(f"train_{ablation_sizes[ablation_size_index]}.csv") as ablation_file:
            ablation_data_train = ablation_file.readlines()
        with open(f"val_{ablation_sizes[ablation_size_index]}.csv") as ablation_file:
            ablation_data_val = ablation_file.readlines()
        with open(f"train_nyckel_{ablation_sizes[ablation_size_index]}.csv") as ablation_file:
            ablation_data_nyckel = ablation_file.readlines()
        for entry in ablation_data_nyckel:
            assert entry in ablation_data_train + ablation_data_val


def test_vertex_files():
    for split in ["train", "val"]:
        for ablation_size_index in range(len(ablation_sizes)):
            with open(f"{split}_{ablation_sizes[ablation_size_index]}.csv") as ablation_file:
                ablation_data = ablation_file.readlines()
            with open(f"{split}_vertex_{ablation_sizes[ablation_size_index]}.csv") as ablation_file:
                vertex_data = ablation_file.readlines()
            vertex_data = [entry.removeprefix(
                "gs://argot-xrays/training_uploads/") for entry in vertex_data]
            vertex_data = [entry.removeprefix(
                "gs://argot-xrays/val_uploads/") for entry in vertex_data]
            assert set(vertex_data) == set(
                ablation_data), f"failed on {split} {ablation_size_index}"


def test_huggingface_files():
    for split in ["train", "val"]:
        for ablation_size_index in range(len(ablation_sizes)):
            with open(f"{split}_{ablation_sizes[ablation_size_index]}.csv") as ablation_file:
                ablation_data = ablation_file.readlines()
            with open(f"{split}_hg_{ablation_sizes[ablation_size_index]}.csv") as ablation_file:
                hf_data = ablation_file.readlines()
            hf_data = hf_data[1:]  # Pop header
            assert set(hf_data) == set(
                ablation_data), f"failed on {split} {ablation_size_index}"


def test_aws_files():
    for split in ["train", "val"]:
        for ablation_size_index in range(len(ablation_sizes)):
            with open(f"{split}_{ablation_sizes[ablation_size_index]}.csv") as ablation_file:
                ablation_data = ablation_file.readlines()
            with open(f"{split}_aws_{ablation_sizes[ablation_size_index]}.csv") as ablation_file:
                aws_data = ablation_file.readlines()
            assert set(aws_data) == set(
                ablation_data), f"failed on {split} {ablation_size_index}"


def test_azure_files():
    for split in ["train", "val"]:
        for ablation_size_index in range(len(ablation_sizes)):
            with open(f"{split}_{ablation_sizes[ablation_size_index]}.csv") as ablation_file:
                ablation_data = ablation_file.readlines()
            with open(f"{split}_azure_{ablation_sizes[ablation_size_index]}.csv") as ablation_file:
                azure_data = ablation_file.readlines()
            azure_data = azure_data[1:]  # Pop header
            azure_data = [entry.removeprefix(
                "azureml://training_uploads/") for entry in azure_data]
            azure_data = [entry.removeprefix(
                "azureml://val_uploads/") for entry in azure_data]

            assert set(azure_data) == set(ablation_data), f"failed on {split} {ablation_size_index}"
