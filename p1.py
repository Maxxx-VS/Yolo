import cv2
import numpy as np
from datetime import datetime
import time

import onnxruntime as ort
import torch
import torch.nn as nn
import torchvision.transforms as transforms


# Загрузка модели YOLOv8
def load_model(model_path):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    return session, input_name


# Функция для предобработки изображений перед подачей в модель
def preprocess_image(image, target_size=(640, 640)):
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1)) / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image


# Функция для постобработки результатов детекции
def postprocess_predictions(outputs, threshold=0.5):
    # Обработка bounding box'ов и классов
    detections = []
    for output in outputs:
        for detection in output:
            if detection[4] > threshold:
                x1, y1, x2, y2 = detection[:4]
                cls_id = int(detection[5])
                score = float(detection[4])
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'class_id': cls_id,
                    'score': score
                })
    return detections


# Основная функция для обработки кадров
def process_frame(frame, model_session, input_name):
    start_time = time.time()

    # Преобразуем кадр в нужный формат
    preprocessed_img = preprocess_image(frame)

    # Выполняем инференс
    outputs = model_session.run(None, {input_name: preprocessed_img})

    # Постобрабатываем результаты
    detections = postprocess_predictions(outputs)

    end_time = time.time()
    fps = 1 / (end_time - start_time)

    print(f"FPS: {fps:.2f}")
    print("Detected objects:")
    for detection in detections:
        bbox = detection['bbox']
        class_id = detection['class_id']
        score = detection['score']
        print(f"\tClass ID: {class_id}, Score: {score:.2f}, BBox: {bbox}")

    return frame


if __name__ == "__main__":
    # Настройки
    model_path = "yolov8n.onnx"  # Путь к файлу модели
    camera_index = 0  # Индекс веб-камеры
    resolution = (640, 480)  # Разрешение входного изображения

    # Загружаем модель
    model_session, input_name = load_model(model_path)

    # Открываем камеру
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        processed_frame = process_frame(frame, model_session, input_name)

        # Показываем обработанный кадр
        cv2.imshow('Frame', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Закрываем камеру и окно отображения
    cap.release()
    cv2.destroyAllWindows()
