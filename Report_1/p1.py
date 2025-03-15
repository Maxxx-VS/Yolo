import cv2
import numpy as np
from ultralytics import YOLO
import time


# Функция для экспорта модели в INT8 формат OpenVINO
def export_int8_model():
    model = YOLO('yolov8n.pt')
    model.export(format='openvino', int8=True, imgsz=640, data='coco.yaml')

model = YOLO('yolov8n_openvino_model/', task='detect')

# Захват с камеры
cap = cv2.VideoCapture(0)

# Переменные для FPS
frame_count = 0
start_time = time.time()
avg_fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Замер времени обработки
    inference_start = time.time()

    # Детекция
    results = model(frame, imgsz=640, verbose=False, half=False)

    # Расчет FPS
    # inference_time = time.time() - inference_start
    frame_count += 1

    # Обновление FPS
    if frame_count % 10 == 0:
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        print(f"FPS: {fps:.2f}")
        frame_count = 0
        start_time = time.time()

    # Отрисовка результатов
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confs, classes):
            if cls == 0 and conf > 0.7:  # Класс 0 = 'person', порог уверенности поставил 70%
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Person {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # Показ результата
    cv2.imshow('YOLOv8', frame)

    # Выход по нажатию на 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()