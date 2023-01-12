import csv
import os
import random
import shutil

test_filename = 'chest_xray_test'
google_bucket_name = 'gs://argot-xrays'
azure_test_uploads = 'azureml://test_uploads/'
testing_list = []

# read CLASSES from file
with open('classes.txt', 'r') as f:
    CLASSES = f.read().split(',')

test_list = [[], []]
for cls in CLASSES:
    for file in os.listdir(f'test/{cls}'):
        test_list[0].append(file)
        test_list[1].append(cls)

if not os.path.exists('test_uploads'):
    os.makedirs('test_uploads')
for file_name, file_class in zip(test_list[0], test_list[1]):
    shutil.copyfile(f'test/{file_class}/{file_name}',
                    f'test_uploads/{file_name}')


vertex_list = [[], []]
for file_name, file_class in zip(test_list[0], test_list[1]):
    vertex_list[0].append(
        f'{google_bucket_name}/test_uploads/{file_name}')
    vertex_list[1].append(file_class)

with open(f'{test_filename}_vertex.csv', 'w') as f:
    a = csv.writer(f, delimiter=',')
    a.writerows(zip(*vertex_list))

azure_list = [['image_url'], ['label']]
for file_name, file_class in zip(test_list[0], test_list[1]):
    azure_list[0].append(
        f'{azure_test_uploads}{file_name}')
    azure_list[1].append(file_class)

with open(f'{test_filename}_azure.csv', 'w') as f:
    a = csv.writer(f, delimiter=',')
    a.writerows(zip(*azure_list))

with open(f'{test_filename}_nyckel.csv', 'w') as f:
    a = csv.writer(f, delimiter=',')
    a.writerows(zip(*test_list))

hg_list = [['image_url'], ['label']]
for file_name, file_class in zip(test_list[0], test_list[1]):
    hg_list[0].append(file_name)
    hg_list[1].append(file_class)

with open(f'{test_filename}_hg.csv', 'w') as f:
    a = csv.writer(f, delimiter=',')
    a.writerows(zip(*hg_list))
