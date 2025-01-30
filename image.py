from matplotlib.pylab import f
from eval import get_truth_boxes
from box import truth_box, pred_box
import numpy as np
import cv2

class image:
    def __init__(self, image_id, image_path):
        self.image_id = image_id
        self.image_path = image_path
        self.truth_boxes = []
        self.pred_boxes = []
        self.load_image()
        self.load_truth_boxes()  

    def __str__(self):
        return f'Image ID: {self.image_id}'\
                f'\nImage Path: {self.image_path}'\
                f'\nTruth Boxes: {np.asarray([truth_box.__str__() for truth_box in self.truth_boxes])}'\
                f'\nPred Boxes: {np.asarray([pred_box.__str__() for pred_box in self.pred_boxes])}'\
                

    def load_truth_boxes(self):
        truth_boxes = get_truth_boxes(self.image_id) # x1,y1, x2, y2, class_id
        for box_ in truth_boxes:
            new_box = truth_box(x1=box_[0],
                                y1=box_[1],
                                x2=box_[2],
                                y2=box_[3],
                                class_id=box_[4])
            self.truth_boxes.append(new_box)

    def load_pred_boxes(self, results): # x1, y1, x2, y2, score, class_id
        for result in results:
            new_box = pred_box(x1=result[0],
                               y1=result[1],
                               x2=result[2],
                               y2=result[3],
                               score=result[4],
                               class_id=result[5])
            self.pred_boxes.append(new_box)

    def find_truth_box(self, threshold=0.5):
        for truth_box in self.truth_boxes:
            best_iou = 0
            best_pred_box = None

            for pred_box in self.pred_boxes:
                if pred_box.matched:
                    continue
                
                current_iou = self.iou(truth_box, pred_box)

                if current_iou > best_iou:
                    best_iou = current_iou
                    best_pred_box = pred_box
            
            if best_pred_box is not None:
                best_pred_box.iou = best_iou
                if best_iou > threshold:
                    truth_box.matched_box = best_pred_box
                    best_pred_box.matched_box = truth_box
                    best_pred_box.matched = True
                    truth_box.matched = True


    def iou(self, truth_box, pred_box):

        common_width = max(0, min(truth_box.x2, pred_box.x2) - max(truth_box.x1, pred_box.x1))
        common_height = max(0, min(truth_box.y2, pred_box.y2) - max(truth_box.y1, pred_box.y1))
        
        intersection = common_width * common_height
        
        truth_area = (truth_box.x2 - truth_box.x1) * (truth_box.y2 - truth_box.y1)
        pred_area = (pred_box.x2 - pred_box.x1) * (pred_box.y2 - pred_box.y1)
        
        union = truth_area + pred_area - intersection
        
        iou = intersection / union
        return iou

    def load_image(self):
        self.image = cv2.imread(self.image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def confision_matrix(self, iou_threshold=0.5):
        iou_data = []
        confision_matrix_data = []

        # Tahmin kutularını kontrol et
        for pred_box in self.pred_boxes:
            iou_data.append(pred_box.iou)
            if pred_box.matched:  # Eşleşmiş tahmin kutusu
                if pred_box.iou >= iou_threshold:
                    confision_matrix_data.append('TP')  # True Positive
                else:
                    confision_matrix_data.append('FP')  # False Positive (IoU düşük)
            else:
                confision_matrix_data.append('FP')  # False Positive (hiçbir eşleşme yok)

        # Gerçek kutuları kontrol et
        for truth_box in self.truth_boxes:
            if not truth_box.matched:  # Hiçbir tahminle eşleşmemiş gerçek kutular
                iou_data.append(0)
                confision_matrix_data.append('FN')  # False Negative

        return iou_data, confision_matrix_data, len(self.truth_boxes)
            
