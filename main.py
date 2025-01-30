import pandas as pd
from ultralytics import RTDETR
from eval import results_to_boxes
from image import image#ros
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import torch
import subprocess
import psutil

model = RTDETR('rtdetr-l.pt')
annotation_file_path = 'CrowdHuman_val/annotation_person.odgt'
with open(annotation_file_path, 'r') as file:
    annotations = [json.loads(line) for line in file]

images_path = 'CrowdHuman_val/Images/'

time_data = []
fps_data = []
iou_data = []
confision_matrix_data = []
gpu_data = []
ram_data = []
watt_data = []
truth_boxes_data = 0

for i, annotation in enumerate(annotations):

    if i % 1 == 0:    
        print(annotation['ID'])
        print(f"{i+1}/{len(annotations)}")
        img_id = annotation['ID']
        img = image(img_id, images_path + img_id + '.jpg')
        
        result = model(img.image)# prediction
        allocated_memory = torch.cuda.memory_allocated() / 1e6# GPU memory
        _result = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'])# GPU watt
        memory_info = psutil.virtual_memory()# RAM 
        
        used_ram = memory_info.used / 1e6
        ms = result[0].speed['inference']
        fps = 1000 / ms
        power_values = [float(x) for x in _result.decode('utf-8').strip().split('\n')]

        time_data.append(ms)
        fps_data.append(fps)
        watt_data.append(sum(power_values) / len(power_values))
        gpu_data.append(allocated_memory)
        ram_data.append(used_ram)


        mask = result[0].boxes.data[:, -1] == 0
        results = result[0].boxes.data[mask]
        results = results_to_boxes(results)  # [x1, y1, x2, y2, score, pred_class]
        img.load_pred_boxes(results)
        img.find_truth_box(threshold=0.5)
        ious, confision_matrix, truth_boxes = img.confision_matrix(iou_threshold=0.5)
        print(f"{[iou for iou in ious]}, {confision_matrix}")

        iou_data.extend(ious)
        confision_matrix_data.extend(confision_matrix)
        truth_boxes_data += truth_boxes


data = np.array([iou_data, confision_matrix_data]).T
sorted_indices = np.argsort(data[:, 0].astype(float))[::-1]  # Azalan sıra için [::-1]
sorted_data = data[sorted_indices]

data = pd.DataFrame(sorted_data, columns=['iou', 'value'])

# get dummies
data.info()
data = pd.get_dummies(data, columns=['value'], prefix='', prefix_sep='')
data = data.astype(float)
data[data.columns.drop('iou')] = data[data.columns.drop('iou')].cumsum(axis=0)

# recall and precision calculation accorting to TP, FP, FN

data['recall'] = data['TP'] / truth_boxes_data
data['precision'] = data['TP'] / (data['TP'] + data['FP'])

recall = data['recall'].to_numpy()
precision = data['precision'].to_numpy()

# average calculation
average_fps = np.mean(fps_data)
average_latency = np.mean(time_data)
avg_gpu_usage = np.mean(gpu_data)
avg_ram_usage = np.mean(ram_data)
avg_watt_usage = np.mean(watt_data)
average_precision = np.trapz(precision, recall)

os.system("clear")
print(f"Average FPS: {average_fps:.4f}")
print(f"Average Latency: {average_latency:.4f} ms")
print(f"Average GPU Usage: {avg_gpu_usage:.4f} MB")
print(f"Average RAM Usage: {avg_ram_usage:.4f} MB")
print(f"Average Watt Usage: {avg_watt_usage:.4f}")
print(f"AP: {average_precision:.4f}")

import matplotlib.pyplot as plt

plt.plot(data['recall'], data['precision'], label=f"AP: {average_precision:.4f}")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.savefig('precision_recall_curve.png')
plt.show()

data.to_csv('data.csv', index=False)