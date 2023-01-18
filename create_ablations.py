import json
import csv
import os
import random
import shutil
import sys

training_list_filename = 'chest_xray_train.csv'
google_bucket_name = 'gs://argot-xrays'
azure_training_uploads = 'azureml://training_uploads/'
azure_val_uploads = 'azureml://val_uploads/'


def get_ablations(dataset):
    ablations = [1280, 320, 80, 20, 5]

    train_folders = os.listdir(f'data/{dataset}/train')
    if ".DS_Store" in train_folders:
        train_folders.remove('.DS_Store')
    print(f'Number of classes in {dataset}: {len(train_folders)}')
    # find the number of images in each class and print the class with the fewest images
    min_images = 100000
    for class_ in train_folders:
        images = os.listdir(f'data/{dataset}/train/{class_}')
        if len(images) < min_images:
            min_images = len(images)
            min_class = class_
    print(
        f'Class with the fewest images: {min_class} with {min_images} images')
    # find the number and index in ABLATIONS that is the next lowest number from min_images
    for i, ablation in enumerate(ablations):
        if ablation < min_images:
            ablation_index = i
            break
    print(
        f'Number of images in the smallest class: {min_images}, next lowest number in ABLATIONS: {ablations[ablation_index]}')

    return ablations[ablation_index:]


def main(ablations):

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
        training_list.extend([file, cls]
                             for file in os.listdir(f'train/{cls}'))

    # shuffle the training list
    random.seed(0)
    random.shuffle(training_list)
    # write the training list to a csv
    with open(training_list_filename, 'w') as f:
        a = csv.writer(f, delimiter=',')
        a.writerows(training_list)
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
        aws_train_list = [[], []]
        aws_val_list = [[], []]
        with open(training_list_filename) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[1] == cls and ablation_count < ablations[0]:
                    class_list[0].append(row[0])
                    class_list[1].append(cls)
                    ablation_count += 1
        train_list[0].extend(class_list[0][:int(ablations[0]*0.8)])
        train_list[1].extend(class_list[1][:int(ablations[0]*0.8)])
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
        aws_train_list[0].extend(train_list[0])
        aws_train_list[1].extend(train_list[1])

        val_list[0].extend(class_list[0][int(ablations[0]*0.8):])
        val_list[1].extend(class_list[1][int(ablations[0]*0.8):])
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
        aws_val_list[0].extend(val_list[0])
        aws_val_list[1].extend(val_list[1])

    with open(f'train_{ablations[0]}.csv', 'w') as f:
        a = csv.writer(f, delimiter=',')
        a.writerows(zip(*train_list))
    with open(f'train_vertex_{ablations[0]}.csv', 'w') as f:
        a = csv.writer(f, delimiter=',')
        a.writerows(zip(*vertex_train_list))
    with open(f'train_hg_{ablations[0]}.csv', 'w') as f:
        a = csv.writer(f, delimiter=',')
        a.writerows(zip(*hg_train_list))
    with open(f'train_nyckel_{ablations[0]}.csv', 'w') as f:
        a = csv.writer(f, delimiter=',')
        a.writerows(zip(*nyckel_train_list))
    with open(f'train_azure_{ablations[0]}.csv', 'w') as f:
        a = csv.writer(f, delimiter=',')
        a.writerows(zip(*azure_train_list))
    with open(f'train_aws_{ablations[0]}.csv', 'w') as f:
        a = csv.writer(f, delimiter=',')
        a.writerows(zip(*aws_train_list))
    with open(f'val_{ablations[0]}.csv', 'w') as f:
        a = csv.writer(f, delimiter=',')
        a.writerows(zip(*val_list))
    with open(f'val_vertex_{ablations[0]}.csv', 'w') as f:
        a = csv.writer(f, delimiter=',')
        a.writerows(zip(*vertex_val_list))
    with open(f'val_hg_{ablations[0]}.csv', 'w') as f:
        a = csv.writer(f, delimiter=',')
        a.writerows(zip(*hg_val_list))
    with open(f'val_azure_{ablations[0]}.csv', 'w') as f:
        a = csv.writer(f, delimiter=',')
        a.writerows(zip(*azure_val_list))
    with open(f'val_aws_{ablations[0]}.csv', 'w') as f:
        a = csv.writer(f, delimiter=',')
        a.writerows(zip(*aws_val_list))

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

    for ablation in ablations[1:]:
        temp_train_list = [[], []]
        vertex_list = [[], []]
        hg_train_list = [['file'], ['label']]
        nyckel_train_list = [[], []]
        azure_train_list = [['image_url'], ['label']]
        aws_train_list = [[], []]

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
        aws_train_list[0].extend(train_list[0])
        aws_train_list[1].extend(train_list[1])

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
        with open(f'train_aws_{ablation}.csv', 'w') as f:
            a = csv.writer(f, delimiter=',')
            a.writerows(zip(*aws_train_list))

        temp_val_list = [[], []]
        vertex_list = [[], []]
        hg_val_list = [['file'], ['label']]
        azure_val_list = [['image_url'], ['label']]
        aws_val_list = [[], []]
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
        aws_val_list[0].extend(val_list[0])
        aws_val_list[1].extend(val_list[1])

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
        with open(f'val_aws_{ablation}.csv', 'w') as f:
            a = csv.writer(f, delimiter=',')
            a.writerows(zip(*aws_val_list))


if __name__ == '__main__':
    dataset = sys.argv[1]
    ablations = get_ablations(dataset)
    print(ablations)
    # main(ablations)
