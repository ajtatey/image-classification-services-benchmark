import csv
import os
import random
import shutil


ABLATIONS = [1280, 320, 80, 20, 5]


for ablation in ABLATIONS:
    if not os.path.exists(f'train_uploads_{ablation}'):
        os.makedirs(f'train_uploads_{ablation}')
    with open(f'train_hg_{ablation}.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[1] != 'label':
                shutil.copyfile(f'train/{row[1]}/{row[0]}',
                                f'train_uploads_{ablation}/{row[0]}')
    if not os.path.exists(f'val_uploads_{ablation}'):
        os.makedirs(f'val_uploads_{ablation}')
    with open(f'val_hg_{ablation}.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[1] != 'label':
                shutil.copyfile(f'train/{row[1]}/{row[0]}',
                                f'val_uploads_{ablation}/{row[0]}')
