import sys
import shutil
import os
import scipy.io
import pandas as pd
from mat4py import loadmat


def extract_dataset(training_set):
    if training_set == 'beans':
        for dataset in ['train', 'test', 'validation']:
            shutil.unpack_archive(
                f'data/{training_set}/{dataset}.zip', f'data/{training_set}')
    elif training_set == 'cars':
        shutil.unpack_archive(
            f'data/{training_set}/car_ims.tgz', f'data/{training_set}/')
        mat_to_csv(training_set)
    elif training_set == 'pets':
        for dataset in ['images', 'annotations']:
            shutil.unpack_archive(
                f'data/{training_set}/{dataset}.tar.gz', f'data/{training_set}')
    elif training_set == 'food':
        shutil.unpack_archive(
            f'data/{training_set}/food-101.tar.gz', f'data/{training_set}/')


def mat_to_csv(training_set):
    if training_set == 'cars':
        data = loadmat('data/cars/cars_annos.mat')
        df = pd.DataFrame(data['annotations'])
        df.to_csv('data/cars/cars_annos.csv', index=False)


def move_validation_to_train(training_set):
    if training_set == 'beans' and os.path.exists(f'data/{training_set}/validation/'):
        classes = ['healthy', 'angular_leaf_spot', 'bean_rust']
        for class_ in classes:
            for file in os.listdir(f'data/{training_set}/validation/{class_}'):
                shutil.move(f'data/{training_set}/validation/{class_}/{file}',
                            f'data/{training_set}/train/{class_}/{file}')
            os.rmdir(f'data/{training_set}/validation/{class_}')
        os.rmdir(f'data/{training_set}/validation/')
    if training_set == 'xrays' and os.path.exists(f'data/{training_set}/chest_xray/val'):
        classes = ['NORMAL', 'PNEUMONIA']
        for class_ in classes:
            for file in os.listdir(f'data/{training_set}/chest_xray/val/{class_}'):
                shutil.move(f'data/{training_set}/chest_xray/val/{class_}/{file}',
                            f'data/{training_set}/chest_xray/train/{class_}/{file}')
            os.rmdir(f'data/{training_set}/chest_xray/val/{class_}')
        os.rmdir(f'data/{training_set}/chest_xray/val/')


def create_train_and_test_dirs(training_set):
    if training_set in ['cars', 'food', 'pets', 'intel', 'xrays']:
        if not os.path.exists(f'data/{training_set}/train'):
            os.makedirs(f'data/{training_set}/train')
        if not os.path.exists(f'data/{training_set}/test'):
            os.makedirs(f'data/{training_set}/test')


def create_class_dirs(training_set):
    if training_set == 'cars':
        classes = pd.read_csv('data/cars/cars_annos.csv')['class'].unique()
        for class_ in classes:
            if not os.path.exists(f'data/cars/train/{class_}'):
                os.makedirs(f'data/cars/train/{class_}')
            if not os.path.exists(f'data/cars/test/{class_}'):
                os.makedirs(f'data/cars/test/{class_}')
    elif training_set == 'food':
        # read classes.txt
        classes = []
        with open('data/food/food-101/meta/classes.txt', 'r') as f:
            for line in f:
                classes.append(line.strip())
        for class_ in classes:
            if not os.path.exists(f'data/food/train/{class_}'):
                os.makedirs(f'data/food/train/{class_}')
            if not os.path.exists(f'data/food/test/{class_}'):
                os.makedirs(f'data/food/test/{class_}')
    elif training_set == 'pets':
        classes = []
        with open('data/pets/annotations/trainval.txt', 'r') as f:
            for line in f:
                image = line.strip().split(' ')[0]
                class_ = '_'.join(image.split('_')[:-1])
                classes.append(class_)
        for class_ in classes:
            if not os.path.exists(f'data/pets/train/{class_}'):
                os.makedirs(f'data/pets/train/{class_}')
            if not os.path.exists(f'data/pets/test/{class_}'):
                os.makedirs(f'data/pets/test/{class_}')


def move_images_to_class_dirs(training_set):
    if training_set == 'cars':
        df = pd.read_csv('data/cars/cars_annos.csv')
        for index, row in df.iterrows():
            if row['test']:
                shutil.move(f'data/cars/{row["relative_im_path"]}',
                            f'data/cars/test/{row["class"]}/{row["relative_im_path"].split("/")[-1]}')
            else:
                shutil.move(f'data/cars/{row["relative_im_path"]}',
                            f'data/cars/train/{row["class"]}/{row["relative_im_path"].split("/")[-1]}')
    elif training_set == 'food':
        with open('data/food/food-101/meta/train.txt', 'r') as f:
            for line in f:
                line = line.strip()
                shutil.move(f'data/food/food-101/images/{line}.jpg',
                            f'data/food/train/{line}.jpg')
        with open('data/food/food-101/meta/test.txt', 'r') as f:
            for line in f:
                line = line.strip()
                shutil.move(f'data/food/food-101/images/{line}.jpg',
                            f'data/food/test/{line}.jpg')
    elif training_set == 'pets':
        with open('data/pets/annotations/trainval.txt', 'r') as f:
            for line in f:
                image = line.strip().split(' ')[0]
                class_ = '_'.join(image.split('_')[:-1])

                shutil.move(f'data/pets/images/{image}.jpg',
                            f'data/pets/train/{class_}/{image}.jpg')
        with open('data/pets/annotations/test.txt', 'r') as f:
            for line in f:
                image = line.strip().split(' ')[0]
                class_ = '_'.join(image.split('_')[:-1])
                shutil.move(f'data/pets/images/{image}.jpg',
                            f'data/pets/test/{class_}/{image}.jpg')


def move_classes_folders_and_images_to_test_and_train(training_set):
    if training_set == 'intel':
        classes = os.listdir('data/intel/seg_train/seg_train')
        for class_ in classes:
            if not os.path.exists(f'data/intel/train/{class_}'):
                os.makedirs(f'data/intel/train/{class_}')
            images = os.listdir(f'data/intel/seg_train/seg_train/{class_}')
            for image in images:
                shutil.move(f'data/intel/seg_train/seg_train/{class_}/{image}',
                            f'data/intel/train/{class_}/{image}')
        classes = os.listdir('data/intel/seg_test/seg_test')
        for class_ in classes:
            if not os.path.exists(f'data/intel/test/{class_}'):
                os.makedirs(f'data/intel/test/{class_}')
            images = os.listdir(f'data/intel/seg_test/seg_test/{class_}')
            for image in images:
                shutil.move(f'data/intel/seg_test/seg_test/{class_}/{image}',
                            f'data/intel/test/{class_}/{image}')
    if training_set == 'xrays':
        classes = os.listdir('data/xrays/chest_xray/train')
        for class_ in classes:
            if not os.path.exists(f'data/xrays/train/{class_}'):
                os.makedirs(f'data/xrays/train/{class_}')
            images = os.listdir(f'data/xrays/chest_xray/train/{class_}')
            for image in images:
                shutil.move(f'data/xrays/chest_xray/train/{class_}/{image}',
                            f'data/xrays/train/{class_}/{image}')
        classes = os.listdir('data/xrays/chest_xray/test')
        for class_ in classes:
            if not os.path.exists(f'data/xrays/test/{class_}'):
                os.makedirs(f'data/xrays/test/{class_}')
            images = os.listdir(f'data/xrays/chest_xray/test/{class_}')
            for image in images:
                shutil.move(f'data/xrays/chest_xray/test/{class_}/{image}',
                            f'data/xrays/test/{class_}/{image}')


if __name__ == '__main__':
    training_set = sys.argv[1]
    extract_dataset(training_set)
    mat_to_csv(training_set)
    move_validation_to_train(training_set)
    create_train_and_test_dirs(training_set)
    create_class_dirs(training_set)
    move_images_to_class_dirs(training_set)
    move_classes_folders_and_images_to_test_and_train(training_set)
