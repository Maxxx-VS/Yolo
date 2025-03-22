import cv2
import torch
import time
import numpy as np

# Загрузка модели YOLOv5n
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

# Переключение в режим оценки (важно для производительности)
model.eval()


def detect_humans(frame, conf_threshold=0.5):
    # Изменение размера кадра
    frame_resized = cv2.resize(frame, (640, 480))

    # Преобразование BGR -> RGB и нормализация
    img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.ascontiguousarray(img)  # Устранение отрицательных шагов
    img = torch.from_numpy(img).float() / 255.0
    img = img.unsqueeze(0)  # Добавление оси батча

    # Детекция
    with torch.no_grad():
        pred = model(img)[0]  # Получаем "сырой" тензор предсказаний

    # Постобработка: формат [x1, y1, x2, y2, conf, class]
    detections = non_max_suppression(pred, conf_threshold, 0.45)[0].numpy()

    # Фильтрация людей (класс 0)
    humans = [0]

    return humans, frame_resized


# Вспомогательная функция для подавления немаксимумов (NMS)
def non_max_suppression(prediction, conf_threshold=0.25, iou_threshold=0.45):
    from torchvision.ops import nms
    prediction = prediction[prediction[:, 4] >= conf_threshold]
    if prediction.size(0) == 0:
        return torch.tensor([])

    # Конвертируем xywh в xyxy
    box = xywh2xyxy(prediction[:, :4])
    scores = prediction[:, 4]
    classes = prediction[:, 5]

    # Применяем NMS
    keep = nms(box, scores, iou_threshold)
    return prediction[keep]


def xywh2xyxy(x):
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # left
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # right
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom
    return y


# Инициализация видеопотока
cap = cv2.VideoCapture(0)
frame_count = 0
fps_start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    humans, frame_resized = detect_humans(frame)

    # Отрисовка боксов
    for det in humans:
        x1, y1, x2, y2 = int(det // 1000), int((det % 1000) // 100), int((det % 100) // 10), int(det % 10)
        #conf = det[:4]
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #cv2.putText(frame_resized, f"Human {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print(x1)

    # Расчет FPS
    frame_count += 1
    fps = frame_count / (time.time() - fps_start)
    cv2.putText(frame_resized, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLOv5 Human Detection", frame_resized)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()