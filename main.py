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

model = RTDETR('rtdetr-l.pt')

annotation_file_path = 'CrowdHuman_val/annotation_person.odgt'

with open(annotation_file_path, 'r') as file:
    annotations = [json.loads(line) for line in file]

images_path = 'CrowdHuman_val/Images/'

total_ms_data = []
preproces_ms_data = []
inference_ms_data = []
postprocess_ms_data = []
fps_data = []
iou_data = []
confision_matrix_data = []
gpu_data = []
ram_data = []
watt_data = []

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
        
        fps = 1000 / total_ms
        power_values = [float(x) for x in _result.decode('utf-8').strip().split('\n')]

        total_ms_data.append(total_ms)
        preproces_ms_data.append(preproces_ms)        
        inference_ms_data.append(inference_ms)        
        postprocess_ms_data.append(postprocess_ms)
        fps_data.append(fps)
        watt_data.append(sum(power_values) / len(power_values))
        gpu_data.append(allocated_memory)
        ram_data.append(used_ram)        


        mask = result[0].boxes.data[:, -1] == 0
        results = result[0].boxes.data[mask]
        
        odgt_line = {
            'ID': img_id,
            'truth_boxes': get_truth_boxes(img_id),# [x1, y1, x2, y2, class_id]
            'pred_boxes': results_to_boxes(results)  # [x1, y1, x2, y2, score, pred_class]
        }
        
        with open('pred_results.odgt', 'a') as f:
            f.write(json.dumps(odgt_line) + '\n')
        
        
            
    
        
        
        

        




# average calculation

average_fps = np.mean(fps_data)
average_preprocess_latency = np.mean(preproces_ms_data)
average_inference_latency = np.mean(inference_ms_data)
average_postprocess_latency = np.mean(postprocess_ms_data)
average_total_latency = np.mean(total_ms_data)
avg_gpu_usage = np.mean(gpu_data)
avg_ram_usage = np.mean(ram_data)
avg_watt_usage = np.mean(watt_data)

os.system("clear")
print(f"Average FPS: {average_fps:.4f}")
print(f"Average Preprocess Latency: {average_preprocess_latency:.4f} ms")
print(f"Average Inference Latency: {average_inference_latency:.4f} ms")
print(f"Average Postprocess Latency: {average_postprocess_latency:.4f} ms")
print(f"Average Total Latency: {average_total_latency:.4f} ms")
print(f"Average GPU Usage: {avg_gpu_usage:.4f} MB")
print(f"Average RAM Usage: {avg_ram_usage:.4f} MB")
print(f"Average Watt Usage: {avg_watt_usage:.4f} Watt")
