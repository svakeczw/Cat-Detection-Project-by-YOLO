import cv2 as cv
import numpy as np


net = net = cv.dnn.readNet('weights/yolov3_training_final.weights', 'weights/yolov3_testing.cfg')


def predict_img(img_file_path):
    img = cv.imread(img_file_path)
    height, width, _ = img.shape
    blob = cv.dnn.blobFromImage(img, 1 / 255, (416, 416), swapRB=True)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()  # get output layers' names
    layerOutputs = net.forward(output_layers_names)  # compute forward and get out put from output layer
    # print(layerOutputs)  # 4 bounding boxes, 1 box confidence score, 1
    return layerOutputs, height, width, img


def generate_bounding_box(layeroutputs, height, width, img):
    classes = []
    boxes = []  # extract bounding boxes
    confidences = []  # store confidences
    class_ids = []  # store class ids
    with open('data/classes.txt', 'r') as f:
        classes = f.read().splitlines()
    for output in layeroutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.9:  # score threshold
                print(scores)
                # get the center and resize back to original size
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # in yolo, the upper left the the original point, so assign x,y to the new original point
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    # get rid of redundant box bu Non-max surpass
    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.9, 0.4)
    font = cv.FONT_HERSHEY_PLAIN  # select a font
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))  # select a color
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = label + " " + confidence
            cv.putText(img, text=text, org=(x, y), fontFace=font, fontScale=2, color=color, thickness=2)
    else:
        text = 'None cat doven detected!'
        color = np.random.uniform(0, 255, size=(1, 3)).flatten()
        cv.putText(img, text,(int(width/2), int(height/2 + 20)), fontFace=font, fontScale=2, color=color, thickness=3)
    return img


def detect(image_file):
    for img in image_file:
        layerOutputs, height, width, img = predict_img(img)
        img_pred = generate_bounding_box(layerOutputs, height, width, img)
        cv.imshow('Cat_Doven', img_pred)
        cv.waitKey(0)
        cv.destroyAllWindows()
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    img_file = ['data/doven1.jpeg', 'data/doven2.jpeg', 'data/doven3.jpeg', 'data/dog1.jpeg']
    detect(img_file)
