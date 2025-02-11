import pandas as pd
from ultralytics import RTDETR
from eval import results_to_boxes, get_truth_boxes
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import torch
import subprocess
import psutil, cv2

model_path = "rtdetr-l.pt"
images_path = 'CrowdHuman_val/Images/'
annotation_file_path = 'CrowdHuman_val/annotation_person.odgt'

model = RTDETR(model_path)

with open(annotation_file_path, 'r') as file:
    annotations = [json.loads(line) for line in file]

for i, annotation in enumerate(annotations):

    if i % 1 == 0:    
        print(annotation['ID'])
        print(f"{i+1}/{len(annotations)}")
        img_id = annotation['ID']
        img = images_path + img_id + '.jpg'
        
        truth_boxes = get_truth_boxes(img_id)
        
        result = model(img)# prediction
        
        allocated_memory = torch.cuda.memory_allocated() / 1e6# GPU memory
        _result = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'])# GPU watt
        process = psutil.Process(os.getpid())# RAM 
        
        used_ram = process.memory_info().rss / 1e6

        preproces_ms = result[0].speed['preprocess']
        inference_ms = result[0].speed['inference']
        postprocess_ms = result[0].speed['postprocess']
        total_ms = preproces_ms + inference_ms + postprocess_ms
        
        power_values = [float(x) for x in _result.decode('utf-8').strip().split('\n')]     

        mask = result[0].boxes.data[:, -1] == 0
        results = result[0].boxes.data[mask]
        
        odgt_line = {
            'ID': img_id,
            'truth_boxes': get_truth_boxes(img_id),# [x1, y1, x2, y2, class_id]
            'pred_boxes': results_to_boxes(results),  # [x1, y1, x2, y2, score, pred_class]
            'preprocess_ms': preproces_ms,
            'inference_ms': inference_ms,
            'postprocess_ms': postprocess_ms,
            'gpu_watt_usage': sum(power_values) / len(power_values),
            'gpu_memory_usage': allocated_memory,
            'ram_memory_usage': used_ram
        }
        
        with open(f'{model_path}_results.odgt', 'a') as f:
            f.write(json.dumps(odgt_line) + '\n')
        