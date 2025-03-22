import torch
import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import time

# Экспорт модели в ONNX
def export_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
    dummy_input = torch.randn(1, 3, 320, 320)  # Новое разрешение
    onnx_path = "yolov5n_320.onnx"
    torch.onnx.export(
        model.model,  # Используем только модель, без постобработки
        dummy_input,
        onnx_path,
        opset_version=12,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    return onnx_path

# Квантование модели
def quantize_model(onnx_path):
    quantized_model_path = "yolov5n_320_int8.onnx"
    quantize_dynamic(onnx_path, quantized_model_path, weight_type=QuantType.QUInt8)
    return quantized_model_path

# Детекция
def detect_objects(model_path):
    # Инициализация ONNX Runtime
    session = ort.InferenceSession(model_path)

    # Функция для предобработки кадра
    def preprocess(frame):
        frame = cv2.resize(frame, (320, 320))  # Новое разрешение
        frame = frame / 255.0  # Нормализация [0, 1]
        frame = frame.transpose(2, 0, 1)  # Изменение порядка осей (HWC -> CHW)
        frame = np.expand_dims(frame, axis=0)  # Добавление batch dimension
        frame = frame.astype(np.float32)  # Преобразование в float32
        return frame

    # Функция для постобработки результатов
    def postprocess(output, frame_shape, conf_threshold=0.5, iou_threshold=0.5):
        detections = output[0]  # Берем первый (и единственный) batch
        detections = detections[detections[:, 5] == 0]  # Фильтрация по классу "человек"
        detections = detections[detections[:, 4] > conf_threshold]  # Фильтрация по уверенности
        boxes = detections[:, :4]
        scores = detections[:, 4]
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
        return detections[indices]

    # Открытие видеопотока с камеры
    cap = cv2.VideoCapture(0)

    # Переменные для расчета FPS
    prev_time = 0
    curr_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Предобработка кадра
        input_tensor = preprocess(frame)

        # Выполнение инференса
        output = session.run(None, {"input": input_tensor})

        # Постобработка результатов
        results = postprocess(output[0], frame.shape)

        # Масштабирование bounding boxes
        scale_x = frame.shape[1] / 320
        scale_y = frame.shape[0] / 320
        results[:, [0, 2]] *= scale_x
        results[:, [1, 3]] *= scale_y

        # Отрисовка bounding boxes
        for detection in results:
            x1, y1, x2, y2, conf, class_id = detection
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Расчет FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        print(f"FPS: {fps:.2f}")

        # Отображение кадра
        # cv2.imshow('YOLOv5 Nano INT8 Human Detection', frame)

        # Выход по нажатию клавиши 'q'
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()

# Основной код
if __name__ == "__main__":
    onnx_path = export_model()
    quantized_model_path = quantize_model(onnx_path)
    detect_objects(quantized_model_path)