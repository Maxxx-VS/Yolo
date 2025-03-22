import cv2
import numpy as np
import time
import onnxruntime as ort

# Загрузка квантованной модели ONNX
onnx_model_path = 'yolov8n.onnx'  # Путь к экспортированной модели
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])  # Используем CPU

# Установка разрешения (320x320, как при экспорте)
input_resolution = (320, 320)

# Функция для обработки кадра и вывода результатов
def detect_and_display(frame):
    # Изменение размера кадра
    resized_frame = cv2.resize(frame, input_resolution)

    # Преобразование кадра в тензор и нормализация
    input_tensor = resized_frame.transpose(2, 0, 1)  # Изменение порядка осей (HWC -> CHW)
    input_tensor = input_tensor / 255.0  # Нормализация [0, 1]
    input_tensor = input_tensor[np.newaxis, ...].astype(np.float32)  # Добавление batch dimension

    # Выполнение инференса
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    # Получение bounding boxes, классов и confidence scores
    predictions = outputs[0]  # Выходные данные модели (num_predictions, 84)
    num_predictions = predictions.shape[0]

    # Преобразование выходных данных в формат [x1, y1, x2, y2, conf, class_id]
    boxes = []
    confidences = []
    for i in range(num_predictions):
        prediction = predictions[i]
        confidence = prediction[4]  # Confidence score для bounding box
        class_scores = prediction[5:]  # Confidence scores для всех классов
        class_id = np.argmax(class_scores)  # Класс с максимальной вероятностью
        if class_id == 0 and confidence > 0.25:  # Фильтрация только класса "человек" (класс 0)
            x1, y1, x2, y2 = prediction[:4]  # Координаты bounding box
            boxes.append([x1, y1, x2, y2])
            confidences.append(confidence)

    # Применение Non-Maximum Suppression (NMS) для удаления дублирующих bounding boxes
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.25, nms_threshold=0.45)

        # Масштабирование bounding boxes обратно к исходному разрешению кадра
        scale_x = frame.shape[1] / input_resolution[0]
        scale_y = frame.shape[0] / input_resolution[1]

        # Вывод bounding boxes и confidence scores
        for i in indices:
            i = i[0] if isinstance(i, np.ndarray) else i  # Обработка формата вывода NMSBoxes
            x1, y1, x2, y2 = boxes[i]
            x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
            confidence = confidences[i]
            label = f"Person {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Вывод FPS
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Отображение кадра
    cv2.imshow("YOLOv8n INT8 Human Detection", frame)

# Захват видео с камеры
cap = cv2.VideoCapture(0)

# Установка разрешения камеры (для увеличения FPS)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Обработка кадра
    detect_and_display(frame)

    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()