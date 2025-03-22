import cv2
import time
import numpy as np
import os
import urllib.request

# Скачивание файлов, если их нет
if not os.path.exists("yolov4-tiny.weights"):
    print("Downloading weights...")
    urllib.request.urlretrieve(
        "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
        "yolov4-tiny.weights"
    )

if not os.path.exists("yolov4-tiny.cfg"):
    print("Downloading config...")
    urllib.request.urlretrieve(
        "https://github.com/AlexeyAB/darknet/raw/master/cfg/yolov4-tiny.cfg",
        "yolov4-tiny.cfg"
    )

if not os.path.exists("coco.names"):
    print("Downloading class names...")
    urllib.request.urlretrieve(
        "https://github.com/pjreddie/darknet/raw/master/data/coco.names",
        "coco.names"
    )

# Загрузка модели YOLOv4-Tiny
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
layer_names = net.getLayerNames()

# Исправление для OpenCV 4.x и выше
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Загрузка классов
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Захват видеопотока с камеры
cap = cv2.VideoCapture(0)

# Переменные для замера FPS
fps_start_time = time.time()
fps_frame_count = 0
fps = 0

while True:
    ret, frame = cap.read()
    # if not ret:
    #     break

    # Подготовка изображения для YOLO
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Обработка результатов детекции
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Применение Non-Max Suppression для удаления дублирующих bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Отображение результатов в консоль
    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        label = f"{classes[class_ids[i]]} {confidences[i]:.1f}"
        print(f"Detected: {label}, Bounding Box: {x}, {y}, {w}, {h}")

    # Замер FPS
    fps_frame_count += 1
    if fps_frame_count >= 10:
        fps = fps_frame_count / (time.time() - fps_start_time)
        fps_frame_count = 0
        fps_start_time = time.time()

    print(f"FPS: {fps:.1f}")

    # # Отображение кадра с bounding boxes (опционально)
    # for i in indices:
    #     box = boxes[i]
    #     x, y, w, h = box
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     cv2.putText(frame, f"{classes[class_ids[i]]} {confidences[i]:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #
    # cv2.imshow("Frame", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()