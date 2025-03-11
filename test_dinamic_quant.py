import cv2
import time
from ultralytics import YOLO

input_width, input_height = 180, 180              # Установка размера изображения


model = YOLO("yolov8n.pt")                        # Загрузка модели YOLOv8n
model.export(format="onnx",
             half=True)                           # Экспорт модели onnx с FP16 (половинная точность)

cap = cv2.VideoCapture(0)                         # Захват видеопотока с веб-камеры
if not cap.isOpened():                            # Проверка отсутсвия видеопотока
    raise IOError("Нет подключения к камере")     # Вызываем исключение + инфо в консоль

fps_count = 0                                     # Начальная позиция счетчика для FPS
start_time = time.time()                          # Время начала процесса

while True:                                       # Главный цикл скрипта
    ret, frame = cap.read()                       # Захват кадра из видеопотока

    frame = cv2.resize(frame,                     # Уменьшение разрешения изображения
            (input_width, input_height))

    results = model(frame,
                    classes=[0],                  # Класс "person" это индекс 0
                    verbose=False,                # verbose = False для минимизации выводов
                    conf=0.85)                     # пороговое значение точности

    fps_count += 1                                # Увеличиваем счетчик итераций
    if fps_count % 15 == 0:                       # Выводим FPS каждые ... итераций
        end_time = time.time()
        fps = fps_count / (end_time - start_time) # Формула расчета FPS
        print(f"FPS: {fps:.2f}")                  # Вывод в консоль
        fps_count = 0                             # Сброс счетчика
        start_time = time.time()

        for result in results:                    # Циклом перебираем найденные объекты
            boxes = result.boxes                  # Cодержит predict (результаты детекции)
            for box in boxes:
                class_id = int(box.cls)           # Индекс класса обнаруженного объекта
                confidence = float(box.conf)      # Вероятность по объекту
                bbox = box.xyxy[0].tolist()
                print(f"КЛАСС: {class_id}, ВЕРОЯТНОСТЬ: {confidence:.2f}, РАМКА: {bbox}")

    annotated_frame = results[0].plot()           # Визуализация результатов
    cv2.imshow('Video stream',            # Вывод на экран
               annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):         # Прерывание цикла по кнопке
        break

cap.release()                                     # Освобождение ресурсов
cv2.destroyAllWindows()