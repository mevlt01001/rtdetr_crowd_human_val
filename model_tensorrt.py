from ultralytics import RTDETR
import cv2

model = RTDETR('rtdetr-l.pt')

model.export(format='tensorrt', int8=True)

model = RTDETR('rtdetr-l.engine')

img = cv2.imread('CrowdHuman_val/Images/273275,5d0330005d00a4c8.jpg')
result = model(img)
mask = result[0].boxes.data[:,-1] == 0
results = result[0].boxes.data[mask]
print(result)

