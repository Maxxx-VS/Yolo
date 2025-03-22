import cv2
import torch
import time

# Загрузка модели YOLOv7 с force_reload=True
model = torch.hub.load('WongKinYiu/yolov7', 'yolov7', force_reload=True)  # Обновление кэша

# Установка устройства (CPU или GPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device).eval()

# Класс "person" в COCO имеет индекс 0
CLASS_PERSON = 0

# Захват видеопотока с камеры
cap = cv2.VideoCapture(0)  # 0 — индекс камеры (обычно это встроенная камера)

# Переменные для расчета FPS
fps_start_time = time.time()
fps_frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование кадра в формат, подходящий для модели
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

    # Детекция объектов
    with torch.no_grad():
        results = model(img)

    # Обработка результатов
    detections = results.xyxy[0].cpu().numpy()  # Получение bounding boxes, классов и confidence scores

    # Фильтрация детекций по классу "person"
    person_detections = [det for det in detections if int(det[5]) == CLASS_PERSON]

    # Вывод результатов в консоль
    for detection in person_detections:
        x1, y1, x2, y2, conf, cls = detection
        print(f"Class: person, Confidence: {conf:.2f}, Bounding Box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

    # Расчет FPS
    fps_frame_count += 1
    if fps_frame_count >= 10:  # Обновляем FPS каждые 10 кадров
        fps_end_time = time.time()
        fps = fps_frame_count / (fps_end_time - fps_start_time)
        print(f"FPS: {fps:.2f}")
        fps_start_time = time.time()
        fps_frame_count = 0

    # Отображение кадра с bounding boxes (опционально)
    for detection in person_detections:
        x1, y1, x2, y2, conf, cls = detection
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"person {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('YOLOv7 Person Detection', frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()