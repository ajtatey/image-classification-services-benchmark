import json
import csv
import os
import random
import shutil

training_list_filename = 'chest_xray_train.csv'
google_bucket_name = 'gs://argot-xrays'
azure_training_uploads = 'azureml://training_uploads/'
azure_val_uploads = 'azureml://val_uploads/'

if not os.path.exists('test/VIRAL'):
    os.makedirs('test/VIRAL')
for file in os.listdir('test/PNEUMONIA'):
    if 'virus' in file:
        shutil.move(f'test/PNEUMONIA/{file}', f'test/VIRAL/{file}')
os.rename('test/PNEUMONIA', 'test/BACTERIA')

if not os.path.exists('train/VIRAL'):
    os.makedirs('train/VIRAL')
for file in os.listdir('train/PNEUMONIA'):
    if 'virus' in file:
        shutil.move(f'train/PNEUMONIA/{file}', f'train/VIRAL/{file}')
os.rename('train/PNEUMONIA', 'train/BACTERIA')

# get the class names from the training folder
CLASSES = os.listdir('train')
# save the class names to a file
if ".DS_Store" in CLASSES:
    CLASSES.remove('.DS_Store')
with open('classes.txt', 'w') as f:
    f.write(','.join(CLASSES))
# create a training list from all the files in the training folder
training_list = []
for cls in CLASSES:
    training_list.extend([file, cls] for file in os.listdir(f'train/{cls}'))

# shuffle the training list
random.seed(0)
random.shuffle(training_list)
# write the training list to a csv
with open(training_list_filename, 'w') as f:
    a = csv.writer(f, delimiter=',')
    a.writerows(training_list)

ABLATIONS = [1280, 320, 80, 20, 5]

class_list = [[], []]
train_list = [[], []]
val_list = [[], []]
for cls in CLASSES:
    ablation_count = 0
    class_list = [[], []]
    vertex_train_list = [[], []]
    vertex_val_list = [[], []]
    azure_train_list = [['image_url'], ['label']]
    azure_val_list = [['image_url'], ['label']]
    hg_train_list = [['file'], ['label']]
    hg_val_list = [['file'], ['label']]
    nyckel_train_list = [[], []]
    with open(training_list_filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[1] == cls and ablation_count < ABLATIONS[0]:
                class_list[0].append(row[0])
                class_list[1].append(cls)
                ablation_count += 1
    train_list[0].extend(class_list[0][:int(ABLATIONS[0]*0.8)])
    train_list[1].extend(class_list[1][:int(ABLATIONS[0]*0.8)])
    for file_name, file_class in zip(train_list[0], train_list[1]):
        vertex_train_list[0].append(
            f'{google_bucket_name}/training_uploads/{file_name}')
        vertex_train_list[1].append(file_class)
    hg_train_list[0].extend(train_list[0])
    hg_train_list[1].extend(train_list[1])
    # concatenate the validation list and the training list into nyckel_train_list
    nyckel_train_list[0].extend(train_list[0])
    nyckel_train_list[1].extend(train_list[1])
    for file_name, file_class in zip(train_list[0], train_list[1]):
        azure_train_list[0].append(
            f'{azure_training_uploads}{file_name}')
        azure_train_list[1].append(file_class)

    val_list[0].extend(class_list[0][int(ABLATIONS[0]*0.8):])
    val_list[1].extend(class_list[1][int(ABLATIONS[0]*0.8):])
    for file_name, file_class in zip(val_list[0], val_list[1]):
        vertex_val_list[0].append(
            f'{google_bucket_name}/val_uploads/{file_name}')
        vertex_val_list[1].append(file_class)
    hg_val_list[0].extend(val_list[0])
    hg_val_list[1].extend(val_list[1])
    nyckel_train_list[0].extend(val_list[0])
    nyckel_train_list[1].extend(val_list[1])
    for file_name, file_class in zip(val_list[0], val_list[1]):
        azure_val_list[0].append(
            f'{azure_val_uploads}{file_name}')
        azure_val_list[1].append(file_class)


with open(f'train_{ABLATIONS[0]}.csv', 'w') as f:
    a = csv.writer(f, delimiter=',')
    a.writerows(zip(*train_list))
with open(f'train_vertex_{ABLATIONS[0]}.csv', 'w') as f:
    a = csv.writer(f, delimiter=',')
    a.writerows(zip(*vertex_train_list))
with open(f'train_hg_{ABLATIONS[0]}.csv', 'w') as f:
    a = csv.writer(f, delimiter=',')
    a.writerows(zip(*hg_train_list))
with open(f'train_nyckel_{ABLATIONS[0]}.csv', 'w') as f:
    a = csv.writer(f, delimiter=',')
    a.writerows(zip(*nyckel_train_list))
with open(f'train_azure_{ABLATIONS[0]}.csv', 'w') as f:
    a = csv.writer(f, delimiter=',')
    a.writerows(zip(*azure_train_list))
with open(f'val_{ABLATIONS[0]}.csv', 'w') as f:
    a = csv.writer(f, delimiter=',')
    a.writerows(zip(*val_list))
with open(f'val_vertex_{ABLATIONS[0]}.csv', 'w') as f:
    a = csv.writer(f, delimiter=',')
    a.writerows(zip(*vertex_val_list))
with open(f'val_hg_{ABLATIONS[0]}.csv', 'w') as f:
    a = csv.writer(f, delimiter=',')
    a.writerows(zip(*hg_val_list))
with open(f'val_azure_{ABLATIONS[0]}.csv', 'w') as f:
    a = csv.writer(f, delimiter=',')
    a.writerows(zip(*azure_val_list))

# copy all the files listed in the training list to a new hg_training folder
# create hg_training folder if it doesn't exist
if not os.path.exists('training_uploads'):
    os.makedirs('training_uploads')

for file_name, file_class in zip(train_list[0], train_list[1]):
    shutil.copyfile(f'train/{file_class}/{file_name}',
                    f'training_uploads/{file_name}')
if not os.path.exists('val_uploads'):
    os.makedirs('val_uploads')
for file_name, file_class in zip(val_list[0], val_list[1]):
    shutil.copyfile(f'train/{file_class}/{file_name}',
                    f'val_uploads/{file_name}')

for ablation in ABLATIONS[1:]:
    temp_train_list = [[], []]
    vertex_list = [[], []]
    hg_train_list = [['file'], ['label']]
    nyckel_train_list = [[], []]
    azure_train_list = [['image_url'], ['label']]

    for cls in CLASSES:
        ablation_count = 0
        for file_name, file_class in zip(train_list[0], train_list[1]):
            if file_class == cls and ablation_count < int(ablation*0.8):
                temp_train_list[0].append(file_name)
                temp_train_list[1].append(cls)
                ablation_count += 1
    train_list = temp_train_list
    for file_name, file_class in zip(train_list[0], train_list[1]):
        vertex_list[0].append(
            f'{google_bucket_name}/training_uploads/{file_name}')
        vertex_list[1].append(file_class)
    hg_train_list[0].extend(train_list[0])
    hg_train_list[1].extend(train_list[1])
    nyckel_train_list[0].extend(train_list[0])
    nyckel_train_list[1].extend(train_list[1])
    for file_name, file_class in zip(train_list[0], train_list[1]):
        azure_train_list[0].append(
            f'{azure_training_uploads}{file_name}')
        azure_train_list[1].append(file_class)

    with open(f'train_{ablation}.csv', 'w') as f:
        a = csv.writer(f, delimiter=',')
        a.writerows(zip(*train_list))
    with open(f'train_vertex_{ablation}.csv', 'w') as f:
        a = csv.writer(f, delimiter=',')
        a.writerows(zip(*vertex_list))
    with open(f'train_hg_{ablation}.csv', 'w') as f:
        a = csv.writer(f, delimiter=',')
        a.writerows(zip(*hg_train_list))
    with open(f'train_azure_{ablation}.csv', 'w') as f:
        a = csv.writer(f, delimiter=',')
        a.writerows(zip(*azure_train_list))

    temp_val_list = [[], []]
    vertex_list = [[], []]
    hg_val_list = [['file'], ['label']]
    azure_val_list = [['image_url'], ['label']]
    for cls in CLASSES:
        ablation_count = 0
        for file_name, file_class in zip(val_list[0], val_list[1]):
            if file_class == cls and ablation_count < ablation*0.2:
                temp_val_list[0].append(file_name)
                temp_val_list[1].append(cls)
                ablation_count += 1
    val_list = temp_val_list
    for file_name, file_class in zip(val_list[0], val_list[1]):
        vertex_list[0].append(
            f'{google_bucket_name}/val_uploads/{file_name}')
        vertex_list[1].append(file_class)
    hg_val_list[0].extend(val_list[0])
    hg_val_list[1].extend(val_list[1])
    nyckel_train_list[0].extend(val_list[0])
    nyckel_train_list[1].extend(val_list[1])
    for file_name, file_class in zip(val_list[0], val_list[1]):
        azure_val_list[0].append(
            f'{azure_val_uploads}{file_name}')
        azure_val_list[1].append(file_class)

    with open(f'val_{ablation}.csv', 'w') as f:
        a = csv.writer(f, delimiter=',')
        a.writerows(zip(*val_list))
    with open(f'val_vertex_{ablation}.csv', 'w') as f:
        a = csv.writer(f, delimiter=',')
        a.writerows(zip(*vertex_list))
    with open(f'val_hg_{ablation}.csv', 'w') as f:
        a = csv.writer(f, delimiter=',')
        a.writerows(zip(*hg_val_list))
    with open(f'train_nyckel_{ablation}.csv', 'w') as f:
        a = csv.writer(f, delimiter=',')
        a.writerows(zip(*nyckel_train_list))
    with open(f'val_azure_{ablation}.csv', 'w') as f:
        a = csv.writer(f, delimiter=',')
        a.writerows(zip(*azure_val_list))
