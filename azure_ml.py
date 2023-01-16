from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes

from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential
import json
import os
import sys


def get_mlclient():

    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")
    try:
        ml_client = MLClient.from_config(credential=credential)

    except Exception as ex:
        # NOTE: Update following workspace information to contain
        #       your subscription ID, resource group name, and workspace name
        client_config = {
            "subscription_id": "subscription_id",
            "resource_group": "resource_group",
            "workspace_name": "workspace_name",
        }

        config_path = "../.azureml/config.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as fo:
            fo.write(json.dumps(client_config))
        ml_client = MLClient.from_config(
            credential=credential, path=config_path)

    return ml_client


def upload():

    dataset_dirs = ["test_uploads", "train_uploads", "val_uploads"]

    for dataset_dir in dataset_dirs:
        my_data = Data(
            path=dataset_dir,
            type=AssetTypes.URI_FOLDER,
            description="argot-xrays",
            name="argot-xrays",
        )
        ml_client = get_mlclient()
        uri_folder_data_asset = ml_client.data.create_or_update(my_data)

        print(uri_folder_data_asset)
        print("")
        print("Path to folder in Blob Storage:")
        print(uri_folder_data_asset.path)


if __name__ == '__main__':
    if sys.argv[1] == 'upload':
        upload()
