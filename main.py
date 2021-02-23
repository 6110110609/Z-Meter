import cv2.cv2 as cv
import numpy as np
import glob

# Load Yolo
net = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")
counter = 0
classes = []

# List File JPG
for list_file in glob.glob(r'*.jpg'):
    print(list_file)

# File Text
filenames = glob.glob(list_file)
filetext = open('data_person.txt','a')

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
color_number = (0, 0, 255)
color_box = (0, 255, 0)

# Loading image
img = cv.imread(list_file)
img = cv.resize(img, None, fx = 0.6, fy = 0.6)
height, width, channels = img.shape

# Detecting objects
blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop = False)

net.setInput(blob)
outs = net.forward(output_layers)
# print(outs)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
# print(indexes)
font = cv.FONT_HERSHEY_COMPLEX_SMALL
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        if label == 'person':
            counter += 1
            cv.rectangle(img, (x, y), (x + w, y + h), color_box, 2)
            cv.putText(img, str(counter) + '', (x, y + 30), font, 2, color_number, 3)
            

print(counter)
for filename in filenames:
    # print(filename[:-4])
    print(filename[19:33])
filetext.write('\nDate:' + filename[19:23] + '-' + filename[23:25] + '-'+ filename[25:27] + 
                ' Time:' + filename[27:29] + ':' + filename[29:31] + ':' + filename[31:33] + '\nPerson: ' + str(counter))

cv.imshow("Image", img)
cv.waitKey(0)
cv.destroyAllWindows()
