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

- 2 main files, `chest_xray_train.csv` and `chest_xray_test.csv` listing all images in the training and testing folders
- 5 training and 5 validation files (one for each ablation) formatted `filename, class`, named `train_{ablation}.csv` and `val_{ablation}.csv`
- 5 training and 5 validation files (one for each ablation) formatted `{google_bucket_name}/{training_uploads|val_uploads}/filename, class` for use with Vertex AI, named `train_vertex_{ablation}.csv` and `val_vertex_{ablation}.csv`
- 5 training and 5 validation files (one for each ablation) formatted `filename, class` for use with Huggingface, named `train_hg_{ablation}.csv` and `val_hg_{ablation}.csv`
- 5 training files (one for each ablation) formatted `filename, class` for use with nyckel, named `train_nyckel_{ablation}.csv`.
- 1 `classes.txt` file containing the class names.

2 folders are also generated, `training_uploads` and `val_uploads` that contain all the images to be used.

### Verify Ablation Correctness

Generate data-files then run `pytest`

## Nyckel

Create environment variables for your `client_id` and `client_secret` like so:

```bash
export NYCKEL_CLIENT_ID=<Your client ID>export NYCKEL_CLIENT_SECRET=<Your client secret>
```

Then run:

```json
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

```json
actual_class, predicted_class, confidence, invoke_time
```

## Huggingface

Create a ‘new project’ at [https://ui.autotrain.huggingface.co/projects](https://ui.autotrain.huggingface.co/projects). Give the project a name and select Task: Vision and Model choice: Automatic and ‘create project.’

Choose Use a .CSV or .JSONL file (Method 2) and then:

1. select the `train_hg_{ablation}.csv` for the ablation size you want to test.
2. Then add the images from the `training_uploads` folder.
3. Choose ‘Training’ as the split type.
4. Then map the data column names

Do the same for the corresponding `val_hg_{ablation}.csv`.

When uploaded, choose 'go to trainings,' select number of model candidates, and then 'start models training.' Once hte models have trained, choose the most accurate one and 'view on model hub' and copy the model name to use as you `inference_endpoint`.

Run:

```bash
python3 huggingface.py invoke <inference_endpoint>
```

This will invoke the model endpoint against each image listed in `chest_xray_test.csv` and give you a running accuracy score, as well as outputting `xray-results-{ablation_size}.csv` which has the format:

```json
actual_class, predicted_class, confidence, invoke_time
```

## Google Vertex AI

Go to [https://console.cloud.google.com/storage/create-bucket](https://console.cloud.google.com/storage/create-bucket) and create a new bucket the same as `google_bucket_name`. You will need to also create a credentials json file for your service account, then add that json to `ml-benchmarking`. Then run:

```json
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
