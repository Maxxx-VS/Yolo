import cv2
import torch
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Инициализация устройства
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Загрузка модели YOLOv5n
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.conf = 0.25  # Порог уверенности для детекции
model.classes=[0]

# Применение оптимизаций
model = model.to(device)
if device == 'cuda':
    model = model.half()  # Используем FP16 на GPU
else:
    model = model.float()  # Используем FP32 на CPU

# Открытие видеопотока
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ошибка открытия видеопотока")
    exit()

# Параметры обработки
input_size = 320  # Уменьшение разрешения
prev_time = time.time()


while True:
    ret, frame = cap.read()
    if not ret:
            break

    # Предобработка кадра
    frame_resized = cv2.resize(frame, (input_size, input_size))
    img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Инференс
    start_time = time.time()
    results = model(img_rgb, size=input_size)
    inference_time = 1 / (time.time() - start_time)

    # Вывод результатов детекции
    detections = results.pandas().xyxy[0]
    for _, det in detections.iterrows():
        print(f"КЛАСС: {det['name']:15} УВЕРЕННОСТЬ: {det['confidence']:.2f} | "
                  f"КОНТУРЫ: {det['xmin']:.1f}, {det['ymin']:.1f}, {det['xmax']:.1f}, {det['ymax']:.1f}")

    # Вывод статистики производительности
    print(f"Текущий FPS: {round(inference_time)}")
    print("-" * 60)



cap.release()
cv2.destroyAllWindows()