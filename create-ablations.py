import shutil
import random
import os
import csv

training_list = 'chest_xray_train.csv'
CLASSES = ['NORMAL', 'PNEUMONIA']
ABLATIONS = [1280, 320, 80, 20, 5]

class_list = [[], []]
train_list = [[], []]
val_list = [[], []]
# split into classes
for cls in CLASSES:
    ablation_count = 0
    class_list = [[], []]
    with open(training_list) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[1] == cls and ablation_count < ABLATIONS[0]:
                class_list[0].append(row[0])
                class_list[1].append(cls)
                ablation_count += 1
    train_list[0].extend(class_list[0][:int(ABLATIONS[0]*0.8)])
    train_list[1].extend(class_list[1][:int(ABLATIONS[0]*0.8)])
    val_list[0].extend(class_list[0][int(ABLATIONS[0]*0.8):])
    val_list[1].extend(class_list[1][int(ABLATIONS[0]*0.8):])

# split into train and val
with open(f'train_test_{ABLATIONS[0]}.csv', 'w') as f:
    a = csv.writer(f, delimiter=',')
    a.writerows(zip(*train_list))
with open(f'val_test_{ABLATIONS[0]}.csv', 'w') as f:
    a = csv.writer(f, delimiter=',')
    a.writerows(zip(*val_list))

# iterate through the ablations
for ablation in ABLATIONS[1:]:
    print(ablation)
    temp_train_list = [[], []]
    for cls in CLASSES:
        ablation_count = 0
        # iterate through the train list
        for file_name, file_class in zip(train_list[0], train_list[1]):
            if file_class == cls and ablation_count < int(ablation*0.8):
                temp_train_list[0].append(file_name)
                temp_train_list[1].append(cls)
                ablation_count += 1
    train_list = temp_train_list
    with open(f'train_test_{ablation}.csv', 'w') as f:
        a = csv.writer(f, delimiter=',')
        a.writerows(zip(*train_list))


# iterate through the ablations
for ablation in ABLATIONS[1:]:
    temp_val_list = [[], []]
    for cls in CLASSES:
        ablation_count = 0
        for file_name, file_class in zip(val_list[0], val_list[1]):
            if file_class == cls and ablation_count < ablation*0.2:
                temp_val_list[0].append(file_name)
                temp_val_list[1].append(cls)
                ablation_count += 1
    val_list = temp_val_list

    with open(f'val_test_{ablation}.csv', 'w') as f:
        a = csv.writer(f, delimiter=',')
        a.writerows(zip(*val_list))
