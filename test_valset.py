import ast
import json
import numpy as np
import pandas as pd

file_path = 'data.csv'

# with open(file_path, 'r') as file:
#     data = [line.strip().replace(' ',',') for line in file]

# filtered_annotations = []
# for annotation in annotations:
#     annotation['gtboxes'] = [box for box in annotation['gtboxes'] if box['tag'] == 'person']
#     filtered_annotations.append(annotation)

# with open('CrowdHuman_val/annotation_person.odgt', 'w') as file:
#     for annotation in filtered_annotations:
#         file.write(json.dumps(annotation) + '\n')

# with open(file_path, 'w') as file:
#     for line in data:
#         file.write(line + '\n')

# boxes = []

# for ann in annotations:
#     for box in ann['gtboxes']:

#         boxes.append([box['ıd'],box['fbox']])

# print(boxes[0])

data = pd.read_csv(file_path)

data.sort_values(by=['iou'], inplace=True, ascending=False)

# get dummies
data.info()
data = pd.get_dummies(data, columns=['value'], prefix='', prefix_sep='')
data = data.astype(float)
data[data.columns.drop('iou')] = data[data.columns.drop('iou')].cumsum(axis=0)

# recall and precision calculation accorting to TP, FP, FN

data['recall'] = data['TP'] / (data['TP'] + data['FN'])
data['precision'] = data['TP'] / (data['TP'] + data['FP'])

print(data)

import matplotlib.pyplot as plt

plt.plot(data['recall'], data['precision'])
plt.show()