import cv2
import torch
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Игнорируем лишние предупреждения
from picamera2 import Picamera2

# Инициализация камеры
picam2 = Picamera2()

# Настройка параметров камеры
config = picam2.create_preview_configuration(
    main={"size": (1920, 1080), "format": "RGB888"}
)
picam2.configure(config)

# Запуск камеры
picam2.start()
time.sleep(1)  # Прогрев камеры

# Инициализация устройства
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Расчеты на: {device}")

# Загрузка модели YOLOv5n
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.conf = 0.25  # Порог уверенности для детекции
model.classes=[0]  # Оставляю только класс person

# Применение оптимизаций
model = model.to(device)
if device == 'cuda':
    model = model.half()  # Используем FP16 на GPU
else:
    model = model.float()  # Используем FP32 на CPU

# Функция уменьшения разрешения
#def resize_image(image, scale_percent):
#    width = int(image.shape[1] * scale_percent / 100)
#    height = int(image.shape[0] * scale_percent / 100)
#    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


# Отключение аппаратного ускорения
#cv2.ocl.setUseOpenCL(False)

# Для RTSP-потока
#stream_url = "rtsp://admin:Admin123@10.4.0.31:554/ISAPI/Streaming/Channels/101"

# Открытие видеопотока
#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(f"{stream_url}?tcp&timeout=5000000", cv2.CAP_V4L2)

#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Уменьшение ширины
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Уменьшение высоты
#cap.set(cv2.CAP_PROP_FPS, 15)            # Ограничение FPS
#cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # Буфер в 1 кадр

#if not cap.isOpened():
   #print("Ошибка открытия видеопотока")
   #exit()

# Параметры обработки
input_size = 180  # Уменьшение разрешения
prev_time = time.time()

while True:
   # ret, frame = cap.read()
    frame = picam2.capture_array()

    #if not ret:
    #        break

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
    print("-" * 80)

# Высвобождение ресурсов
picam2.stop()
cv2.destroyAllWindows()
