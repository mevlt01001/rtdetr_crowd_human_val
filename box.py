from typing import Optional


class box:
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.matched = False
        self.area = (self.x2 - self.x1) * (self.y2 - self.y1)

    def __str__(self):
        return f'x1: {self.x1}, y1: {self.y1}, x2: {self.x2}, y2: {self.y2}'

class truth_box(box):
    def __init__(self, x1: int, y1: int, x2: int, y2: int, class_id: int):
        super().__init__(x1, y1, x2, y2)
        if class_id == 'person':
            self.class_id = 0
        else:
            self.class_id = 1


class pred_box(box):
    def __init__(self, x1: int, y1: int, x2: int, y2: int, score: float, class_id: int):
        super().__init__(x1, y1, x2, y2)
        self.score = float(score)
        self.class_id = int(class_id)
        self.matched_box = None
        self.iou = 0    
