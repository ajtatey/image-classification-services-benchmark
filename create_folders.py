import csv
import os
import random
import shutil
import sys
from create_ablations import get_ablations


def create_folders(dataset, ablations):
    for ablation in ablations:
        if not os.path.exists(f'data/{dataset}/train_uploads_{ablation}'):
            os.makedirs(f'data/{dataset}/train_uploads_{ablation}')
        with open(f'data/{dataset}/ablations/{dataset}_train_hg_{ablation}.csv') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[1] != 'label':
                    shutil.copyfile(f'data/{dataset}/train/{row[1]}/{row[0]}',
                                    f'data/{dataset}/train_uploads_{ablation}/{row[0]}')
        if not os.path.exists(f'data/{dataset}/val_uploads_{ablation}'):
            os.makedirs(f'data/{dataset}/val_uploads_{ablation}')
        with open(f'data/{dataset}/ablations/{dataset}_val_hg_{ablation}.csv') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[1] != 'label':
                    shutil.copyfile(f'data/{dataset}/train/{row[1]}/{row[0]}',
                                    f'data/{dataset}/val_uploads_{ablation}/{row[0]}')


if __name__ == '__main__':
    dataset = sys.argv[1]
    ablations = get_ablations(dataset)
    create_folders(dataset, ablations)
