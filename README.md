# ml-benchmarking

- Clone this repository.
- Create a python virtual environment. For example say:

```bash
python3 -m venv env
source env/bin/activate
```

- Install requirements like so `pip install -r requirements.txt`

### Create Ablations

Run `create_ablations.py`

```bash
python3 create_ablations.py
```

This generates 37 `.csv` files and 1 `.txt` in this folder:

- `chest_xray_train.csv` listing all images in the training folders
- 5 training and 5 validation files (one for each ablation) formatted `filename, class`, named `train_{ablation}.csv` and `val_{ablation}.csv`
- 5 training and 5 validation files (one for each ablation) formatted `{google_bucket_name}/{training_uploads|val_uploads}/filename, class` for use with Vertex AI, named `train_vertex_{ablation}.csv` and `val_vertex_{ablation}.csv`
  5 training and 5 validation files (one for each ablation) formatted `{azure_datastore_name}/filename, class` for use with Azure ML, named `train_azure_{ablation}.csv` and `val_azure_{ablation}.csv`
- 5 training and 5 validation files (one for each ablation) formatted `filename, class` for use with Huggingface, named `train_hg_{ablation}.csv` and `val_hg_{ablation}.csv`
- 5 training files (one for each ablation) formatted `filename, class` for use with nyckel, named `train_nyckel_{ablation}.csv`.
- 1 `classes.txt` file containing the class names.

### Verify Ablation Correctness

Generate data-files then run `pytest`

## Create Testing data

Run `create_tests.py`

```bash
python3 create_tests.py
```

This will create 5 testing files, one for each of the ML services.

## Create data folders

Run `create_folders.py`

```bash
python3 create_folders.py
```

This will create 10 folders containing training and validation data for each of the ablations. These can be used with huggingface.

## Nyckel

Create environment variables for your `client_id` and `client_secret` like so:

```bash
export NYCKEL_CLIENT_ID=<Your client ID>export NYCKEL_CLIENT_SECRET=<Your client secret>
```

Then run:

```bash
python3 nyckel.py create <function_name>
```

This will create a new function and output a `function_id` to use in subsequent calls. To upload the training set, run:

```bash
python3 nyckel.py upload <function_id> <ablation_size>
```

This will create classes from classes.txt and upload the images listed in `train_nyckel_{ablation_size}.csv`

Once upload has completed and the model trained, you can then run:

```bash
python nyckel.py invoke <your_function_id> <ablation_size>
```

This will invoke the model endpoint against each image listed in `chest_xray_test.csv` and give you a running accuracy score, as well as outputting `xray-results-{ablation_size}.csv` which has the format:

```bash
actual_class, predicted_class, confidence, invoke_time
```

## Huggingface

Create a ‘new project’ at [https://ui.autotrain.huggingface.co/projects](https://ui.autotrain.huggingface.co/projects). Give the project a name and select Task: Vision and Model choice: Automatic and ‘create project.’

Choose Use a .CSV or .JSONL file (Method 2) and then:

1. select the `train_hg_{ablation}.csv` for the ablation size you want to test.
2. Then add the images from the `training_uploads_{ablation}` folder.
3. Choose ‘Training’ as the split type.
4. Then map the data column names

Do the same for the corresponding `val_hg_{ablation}.csv` and `val_uploads_{ablation}`.

When uploaded, choose 'go to trainings,' select number of model candidates, and then 'start models training.' Once hte models have trained, choose the most accurate one and 'view on model hub' and copy the model name to use as you `inference_endpoint`.

Run:

```bash
python3 huggingface.py invoke <inference_endpoint>
```

This will invoke the model endpoint against each image listed in `chest_xray_test_hg.csv` and give you a running accuracy score, as well as outputting `xray-results-{ablation_size}.csv` which has the format:

```json
actual_class, predicted_class, confidence, invoke_time
```

## Google Vertex AI

Go to [https://console.cloud.google.com/storage/create-bucket](https://console.cloud.google.com/storage/create-bucket) and create a new bucket the same as `google_bucket_name`. You will need to also create a credentials json file for your service account, then add that json to `ml-benchmarking`. Then run:

```bash
python3 vertex.py
```

This will upload the images from the `training_uploads` and `val_uploads` folders into the bucket.

Once they have uploaded, go to [https://console.cloud.google.com/vertex-ai/datasets](https://console.cloud.google.com/vertex-ai/datasets) and:

- create a new dataset
- choose image classification
- choose “upload import files from your computer”
- upload the `train_vertex_{ablation}.csv` and `val_vertex_{ablation}.csv` for the ablation size you are testing along with `chest_xray_test.csv` and assign them to their corresponding data splits. Choose `google_bucket_name` as the storage path.

Once the dataset has imported, choose ‘Train new model.’ Choose Advanced Options and then a ‘Manual’ data split.

Once the model has trained, the accuracy will be available in the UI.

## Microsoft Azure

After you sign up for Azure Machine Learning, create a workspace. Take note of the workspace and resource group name.

You'll initially have to create credentials for accessing Azure using the SDK. To do so run the Azure CLI with:

```bash
az login
```

This will return an object containing your tenant and subscription ids. Firstly run:

```
az ad sp create-for-rbac --sdk-auth --name ml-auth --role Contributor --scopes /subscriptions/<SUBSCRIPTION_ID>
```

with your SUBSCRIPTION_ID to create a contributor role. This will give you a client id and client secret you'll need to use the SDK. Add these to your environmental variables:

```
export AZURE_CLIENT_ID=<CLIENT_ID>
export AZURE_TENANT_ID=<TENANT_ID>
export AZURE_CLIENT_SECRET=<CLIENT_SECRET>
```

Within `azure_ml.py` add your subscription_id, resource_group, and workspace_name.
Then you can run:

```
python3 azure_ml.py upload
```

This will upload the training, validation, and testing data to separate data blobs. (note: do this before running `create_ablations.py` as you'll need these blobs for the image urls).

Once uploaded, go to (https://ml.azure.com/)[https://ml.azure.com/] and choose your workspace. Choose 'Automated ML,' then 'New Automated ML job.' On the next page select 'Create' to add a new data asset, give it a name, select 'Next' and then 'From local files.' Choose a datastore then choose Upload on the next page. Choose you initial ablation training file (e.g. `train_azure_5.csv`). Click through the pages and 'Create.' Then do the same for the validation and testing files.

Once you have created your data assets, select the training dataset and click 'Next.' On the next page, choose 'label' as the target and either an existing compute cluster or create a new compute cluster. Select 'Classification' on the next page. On the next page choose 'User validation data' as the validation type and choose the validation file. Choose 'Provide a test data asset' as the Test data asset and choose the test file.

Click finish and the job will be created and start. You can find the acuuracy of the model on the Metrics page for the completed model.

## AWS Rekognition
