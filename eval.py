import numpy as np
import json

def find_truth_box(truth_boxes, pred_box):
    ious = []

    for truth_box in truth_boxes:
        ious.append(iou(truth_box, pred_box))

    truthBox_index = np.argmax(ious)

    return truth_boxes[truthBox_index], ious[truthBox_index]

def get_truth_boxes(image, file_path='CrowdHuman_val/annotation_person.odgt'):
    boxes = []

    with open(file_path, 'r') as file:
        annotations = [json.loads(line) for line in file]

    for annotation in annotations:
        if annotation['ID'] == image:
            for box in annotation['gtboxes']:
                x1, y1, width, height = box['vbox']
                boxes.append([x1, y1, x1 + width, y1 + height, box['tag']])

    return np.array(boxes)

def iou(truth_box, pred_box):
    tx1, ty1, tx2, ty2 = truth_box[0], truth_box[1], truth_box[2], truth_box[3]
    px1, py1, px2, py2 = pred_box[0], pred_box[1], pred_box[2], pred_box[3]

    intersection = (min(tx2, px2) - max(tx1, px1)) * (min(ty2, py2) - max(ty1, py1))
    union = (tx2 - tx1) * (ty2 - ty1) + (px2 - px1) * (py2 - py1) - intersection

    iou = intersection / union

    return iou

def results_to_boxes(results):
    boxes = []

    for result in results:
        boxes.append([int(result[0].item()), int(result[1].item()), int(result[2].item()), int(result[3].item()), float(result[4].item()), int(result[5].item())])

    return np.array(boxes)
    
def mAP(truth_boxes, pred_boxes, iou_threshold=0.5):
    pass