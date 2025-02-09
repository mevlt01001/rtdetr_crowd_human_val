from ultralytics import RTDETR
import cv2
from image import image

model = RTDETR('rtdetr-l.pt')

img = image('273275,5d0330005d00a4c8', 'CrowdHuman_val/Images/273275,5d0330005d00a4c8.jpg')
result = model(img.image)
mask = result[0].boxes.data[:,-1] == 0
results = result[0]
print(results)
quit(1)

img.load_pred_boxes(results)

print(img.__str__())

import matplotlib.pyplot as plt

boxed_img = img.image.copy()

for result in results:
    boxed_img = cv2.rectangle(img.image, (int(result[0]), int(result[1])), (int(result[2]), int(result[3])), (0, 255, 0), 2)
    boxed_img = cv2.putText(boxed_img, f'x1: {result[0]:.0f}, y1: {result[1]:.0f}', (int(result[0]), int(result[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    boxed_img = cv2.putText(boxed_img, f'x2: {result[2]:.0f}, y2: {result[3]:.0f}', (int(result[2]), int(result[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

rgb_image = cv2.cvtColor(boxed_img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_image)
plt.show()
