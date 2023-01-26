# ml-benchmarking

- Clone this repository.
- Create a python virtual environment. For example say:

```bash
python3 -m venv env
source env/bin/activate
```

- Install requirements like so `pip install -r requirements.txt`

## Prepare data

Run `python build_all_data.py` to run all steps below.

### Fetch Datasets

Fetch Kaggle credential by downloading kaggle.json into ~/.kaggle/kaggle.json. See <https://github.com/Kaggle/kaggle-api>

Run `fetch_datasets.py`

```bash
python3 fetch_datasets.py <dataset>
```

Where `dataset` is one of:

- beans [https://huggingface.co/datasets/beans](https://huggingface.co/datasets/beans)
- cars [http://ai.stanford.edu/~jkrause/cars/car_dataset.html](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)
- food [https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- intel [https://www.kaggle.com/datasets/puneet6060/intel-image-classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- pets [https://www.robots.ox.ac.uk/~vgg/data/pets/](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- xrays [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

This will download the requisite data for that dataset to `data/<dataset>`.

### Preprocess dataset

Run `preprocessing.py`

```bash
python3 preprocessing.py <dataset>
```

This will extract the data from the compressed files and create a `train` folder and a `test` folder, each containing subfolders organized by class, with the training or testing images within.

### Create Ablations

Edit `bucket_config.json` to add the names of the Google and Azure buckets containing your images. See Google Vertex AI and Azure sections below for details.

Run `create_ablations.py`

```bash
python3 create_ablations.py <dataset>
```

This generates 55 `.csv` files in `data/<dataset>/ablations`:

- 5 training and 5 validation files (one for each ablation) formatted `filename, class`, named `<dataset>_train_{ablation}.csv` and `<dataset>_val_{ablation}.csv`
- 5 training and 5 validation files (one for each ablation) formatted `{google_bucket_name}/{training_uploads|val_uploads}/filename, class` for use with Vertex AI, named `<dataset>_train_vertex_{ablation}.csv` and `<dataset>_val_vertex_{ablation}.csv`
  5 training and 5 validation files (one for each ablation) formatted `{azure_datastore_name}/filename, class` for use with Azure ML, named `<dataset>_train_azure_{ablation}.csv` and `<dataset>_val_azure_{ablation}.csv`
- 5 training and 5 validation files (one for each ablation) formatted `filename, class` for use with Huggingface, named `<dataset>_train_hg_{ablation}.csv` and `<dataset>_val_hg_{ablation}.csv`
- 5 training files (one for each ablation) formatted `filename, class` for use with nyckel, named `<dataset>_train_nyckel_{ablation}.csv`.
- 5 training files and 5 validation files (one for each ablation) formatted `filename, class` for use with AWS Rekognition, named `<dataset>_train_aws_{ablation}.csv` and `<dataset>_val_aws_{ablation}.csv`.

It will also create `classes.txt` in `data/<dataset>` containing the class names.

### Verify Ablation Correctness

Generate data-files then run `pytest`

## Create Testing data

Run `create_tests.py`

```bash
python3 create_tests.py <dataset>
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
python3 nyckel.py upload <dataset> <function_id> <ablation_size>
```

This will create classes from classes.txt and upload the images listed in `train_nyckel_{ablation_size}.csv`

Once upload has completed and the model trained, you can then run:

```bash
python nyckel.py invoke <dataset> <your_function_id> <ablation_size>
```

This will invoke the model endpoint against each image listed in `<dataset>_test_nyckel.csv` and give you a running accuracy score, as well as outputting `<dataset>-nyckel-results-{ablation_size}.csv` which has the format:

```bash
actual_class, predicted_class, confidence, invoke_time
```

## Huggingface

Create a ‘new project’ at [https://ui.autotrain.huggingface.co/projects](https://ui.autotrain.huggingface.co/projects). Give the project a name and select Task: Vision and Model choice: Automatic and ‘create project.’

Choose Use a .CSV or .JSONL file (Method 2) and then:

1. select the `data/<dataset>/ablations/train_hg_{ablation}.csv` for the ablation size you want to test.
2. Then add the images from the `data/<dataset>/training_uploads_{ablation}` folder.
3. Choose ‘Training’ as the split type.
4. Then map the data column names

Do the same for the corresponding `data/<dataset>/ablations/val_hg_{ablation}.csv` and `data/<dataset>/val_uploads_{ablation}`.

When uploaded, choose 'go to trainings,' select number of model candidates, and then 'start models training.' Once hte models have trained, choose the most accurate one and 'view on model hub' and copy the model name to use as you `inference_endpoint`.

Run:

```bash
python3 huggingface.py invoke <dataset> <inference_endpoint>
```

This will invoke the model endpoint against each image listed in `<dataset>_test_hg.csv` and give you a running accuracy score, as well as outputting `<dataset>-hg-results-{ablation_size}.csv` which has the format:

```json
actual_class, predicted_class, confidence, invoke_time
```

## Google Vertex AI

Go to [https://console.cloud.google.com/storage/create-bucket](https://console.cloud.google.com/storage/create-bucket) and create a new bucket the same as `google_bucket_name`. You will need to also create a credentials json file for your service account, then add that json to `ml-benchmarking`. Then run:

```bash
python3 vertex.py <dataset>
```

This will upload the images from the `data/<dataset>/training_uploads` and `data/<dataset>/val_uploads` folders into the bucket.

Once they have uploaded, go to [https://console.cloud.google.com/vertex-ai/datasets](https://console.cloud.google.com/vertex-ai/datasets) and:

- create a new dataset
- choose image classification
- choose “upload import files from your computer”
- upload the `data/<dataset>/ablations/train_vertex_{ablation}.csv` and `data/<dataset>/ablations/val_vertex_{ablation}.csv` for the ablation size you are testing along with `data/<dataset>/<dataset>_test_vertex.csv` and assign them to their corresponding data splits. Choose `google_bucket_name` as the storage path.

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
python3 azure_ml.py upload <dataset>
```

This will upload the training, validation, and testing data to separate data blobs. (note: do this before running `create_ablations.py` as you'll need the `azure_training|val|test_uploads` URIs for the image urls).

Once uploaded, go to [https://ml.azure.com/](https://ml.azure.com/) and choose your workspace. Choose 'Automated ML,' then 'New Automated ML job.' On the next page select 'Create' to add a new data asset, give it a name, select 'Next' and then 'From local files.' Choose a datastore then choose Upload on the next page. Choose you initial ablation training file (e.g. `data/<dataset>/ablations/train_azure_5.csv`). Click through the pages and 'Create.' Then do the same for the validation and testing files.

Once you have created your data assets, select the training dataset and click 'Next.' On the next page, choose 'label' as the target and either an existing compute cluster or create a new compute cluster. Select 'Classification' on the next page. On the next page choose 'User validation data' as the validation type and choose the validation file. Choose 'Provide a test data asset' as the Test data asset and choose the test file.

Click finish and the job will be created and start. You can find the acuuracy of the model on the Metrics page for the completed model.

## AWS Rekognition

To setup AWS Rekognition, you need:

- your access key and secret key from your security credentials under your account
- an S3 bucket for the images

When you have those, set them as environment variables:

```
export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>
export AWS_STORAGE_BUCKET_NAME=<AWS_STORAGE_BUCKET_NAME>
```

Then run:

```
python3 aws_rekognition.py upload <dataset> <ablation>
```

This will upload the images for `<ablation>` to `train` and `test` folders in the S3 bucket. The images within these folders will be organized into `class` folders.

Once uploaded, go to [Amazon Rekognition Custom Labels](https://us-east-2.console.aws.amazon.com/rekognition/custom-labels#/) and click Get Started. Choose 'Projects' in the left menu and then 'Create Project'. Name your project then choose 'Create Dataset.' On the next page:

- Choose 'Start with a training dataset and a test dataset'
- Choose 'Import images from S3 bucket' in Training dataset details and add your train folder S3 URI
- Choose 'Automatically assign image-level labels to images based on the folder name'
- Choose 'Import images from S3 bucket' in Test dataset details and add your test folder S3 URI
- Choose 'Automatically assign image-level labels to images based on the folder name'

Once the data is imported, select 'Train Model,' then 'Train Model' again on the next page.

Once the model has completed training, choose 'Check Metrics' to find the performance of the model.
