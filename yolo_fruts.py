import cv2
import torch
from collections import defaultdict

def count_objects_with_yolov5(image_path, model_name='yolov5s', conf_threshold=0.2):

    # Загрузка модели YOLOv5
    model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
    
    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    
    # Детекция объектов
    results = model(image)
    
    # Получение предсказаний
    preds = results.pandas().xyxy[0]  # DataFrame с результатами
    
    # Фильтрация по порогу уверенности
    filtered_preds = preds[preds['confidence'] >= conf_threshold]
    
    # Подсчёт объектов по классам
    object_counts = defaultdict(int)
    for _, row in filtered_preds.iterrows():
        class_name = row['name']
        object_counts[class_name] += 1
    
    # Отрисовка bounding boxes на изображении
    image_with_boxes = results.render()[0]  # Возвращает список изображений (в данном случае одно)
    # image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR)  # Конвертация RGB → BGR
    
    return dict(object_counts), image_with_boxes

# Пример использования
if __name__ == "__main__":
    image_path = "frut1.jpg"  # Замените на свой путь
    counts, image_with_boxes = count_objects_with_yolov5(image_path)
    
    print("Результаты подсчёта объектов:")
    for obj, count in counts.items():
        print(f"{obj}: {count}")
    
    # Сохранение изображения с bounding boxes
    output_path = "detected_objects.jpg"
    cv2.imwrite(output_path, image_with_boxes)
    print(f"Изображение с детекцией сохранено в: {output_path}")
