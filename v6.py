import cv2
import torch
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import time
import sys


def export_yolov6_to_onnx():
    try:
        print("Attempting to load YOLOv6 model...")

        # Загрузка YOLOv6 Nano (без аргумента pretrained)
        model = torch.hub.load('meituan/YOLOv6', 'yolov6n')

        print("Model loaded successfully!")

        # Создание dummy input
        dummy_input = torch.randn(1, 3, 640, 640)
        onnx_path = "yolov6n.onnx"
        print(f"Exporting model to {onnx_path}...")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=12,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print("ONNX export completed!")
        return onnx_path
    except Exception as e:
        print(f"Error in export: {str(e)}")
        sys.exit(1)


def quantize_onnx_model(onnx_path):
    try:
        quantized_model_path = "yolov6n_int8.onnx"
        print(f"Quantizing model to {quantized_model_path}...")
        quantize_dynamic(onnx_path, quantized_model_path, weight_type=QuantType.QUInt8)
        print("Quantization completed!")
        return quantized_model_path
    except Exception as e:
        print(f"Error in quantization: {str(e)}")
        sys.exit(1)


def detect_objects(model_path):
    try:
        print(f"Initializing ONNX Runtime session with {model_path}...")
        session = ort.InferenceSession(model_path)
        print("Session created successfully!")

        # Открытие видеопотока с камеры
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera!")
            sys.exit(1)
        print("Camera opened successfully!")

        prev_time = 0
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame from camera!")
                break

            # Предобработка кадра
            input_tensor = cv2.resize(frame, (640, 640))  # Изменение размера до 640x640
            input_tensor = input_tensor.transpose(2, 0, 1).astype(np.float32) / 255.0  # Нормализация
            input_tensor = np.expand_dims(input_tensor, axis=0)  # Добавление batch dimension

            # Выполнение инференса
            try:
                output = session.run(None, {"input": input_tensor})
                print(f"Inference completed! Output shape: {output[0].shape}")
            except Exception as e:
                print(f"Inference error: {str(e)}")
                continue

            # Постобработка результатов
            detections = output[0][0]  # Берем первый batch
            detections = detections[detections[:, 4] > 0.5]  # Фильтрация по confidence score
            detections = detections[detections[:, 5] == 0]  # Фильтрация по классу "человек"

            # Масштабирование bounding boxes
            scale_x = frame.shape[1] / 640
            scale_y = frame.shape[0] / 640
            detections[:, [0, 2]] *= scale_x
            detections[:, [1, 3]] *= scale_y

            # Вывод результатов в консоль
            if len(detections) > 0:
                for detection in detections:
                    x1, y1, x2, y2, conf, class_id = detection
                    print(
                        f"Class: Person, Confidence: {conf:.2f}, Bounding Box: [x1: {x1:.0f}, y1: {y1:.0f}, x2: {x2:.0f}, y2: {y2:.0f}]")
            else:
                print("No detections")

            # Отрисовка bounding boxes
            for detection in detections:
                x1, y1, x2, y2, conf, class_id = detection
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Расчет FPS
            frame_count += 1
            curr_time = time.time()
            fps = frame_count / (curr_time - start_time)
            sys.stdout.write(f"\rFPS: {fps:.2f} | Frame: {frame_count} ")
            sys.stdout.flush()

            # Отображение кадра
            cv2.imshow('YOLOv6 Nano INT8 Human Detection', frame)

            # Выход по нажатию клавиши 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("\nExecution completed!")

    except Exception as e:
        print(f"Detection error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    print("===== Starting YOLOv6 Demo =====")
    onnx_path = export_yolov6_to_onnx()
    quantized_path = quantize_onnx_model(onnx_path)
    detect_objects(quantized_path)
    print("===== Program Finished =====")
