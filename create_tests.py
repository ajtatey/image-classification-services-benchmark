import csv
import os
import shutil
import sys

from create_ablations import get_bucket_uris

testing_list = []


def create_test_list(dataset, google_bucket_name, azure_test_uploads):

    # read CLASSES from file
    with open(f"data/{dataset}/classes.txt", "r") as f:
        CLASSES = f.read().split(",")

    test_list = [[], []]
    for cls in CLASSES:
        for file in os.listdir(f"data/{dataset}/test/{cls}"):
            test_list[0].append(file)
            test_list[1].append(cls)

    if not os.path.exists(f"data/{dataset}/test_uploads"):
        os.makedirs(f"data/{dataset}/test_uploads")
    for file_name, file_class in zip(test_list[0], test_list[1]):
        shutil.copyfile(f"data/{dataset}/test/{file_class}/{file_name}", f"data/{dataset}/test_uploads/{file_name}")

    vertex_list = [[], []]
    for file_name, file_class in zip(test_list[0], test_list[1]):
        vertex_list[0].append(f"{google_bucket_name}/test_uploads/{file_name}")
        vertex_list[1].append(file_class)

    with open(f"data/{dataset}/{dataset}_test_vertex.csv", "w") as f:
        a = csv.writer(f, delimiter=",")
        a.writerows(zip(*vertex_list))

    azure_list = [["image_url"], ["label"]]
    for file_name, file_class in zip(test_list[0], test_list[1]):
        azure_list[0].append(f"{azure_test_uploads}{file_name}")
        azure_list[1].append(file_class)

    with open(f"data/{dataset}/{dataset}_test_azure.csv", "w") as f:
        a = csv.writer(f, delimiter=",")
        a.writerows(zip(*azure_list))

    with open(f"data/{dataset}/{dataset}_test_nyckel.csv", "w") as f:
        a = csv.writer(f, delimiter=",")
        a.writerows(zip(*test_list))

    hg_list = [["image_url"], ["label"]]
    for file_name, file_class in zip(test_list[0], test_list[1]):
        hg_list[0].append(file_name)
        hg_list[1].append(file_class)

    with open(f"data/{dataset}/{dataset}_test_hg.csv", "w") as f:
        a = csv.writer(f, delimiter=",")
        a.writerows(zip(*hg_list))

    aws_list = [[], []]
    for file_name, file_class in zip(test_list[0], test_list[1]):
        aws_list[0].append(file_name)
        aws_list[1].append(file_class)

    with open(f"data/{dataset}/{dataset}_test_aws.csv", "w") as f:
        a = csv.writer(f, delimiter=",")
        a.writerows(zip(*aws_list))


def create_test_main(dataset):
    google_bucket_name, _, _, azure_test_uploads = get_bucket_uris(dataset)
    create_test_list(dataset, google_bucket_name, azure_test_uploads)


if __name__ == "__main__":
    dataset = sys.argv[1]
    create_test_list(dataset)
