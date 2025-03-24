#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <chrono>
#include <iostream>

// Функция для предобработки изображения перед подачей в модель
torch::Tensor preprocess(cv::Mat &image, int target_size = 640) {
    // Изменение размера изображения
    cv::resize(image, image, cv::Size(target_size, target_size));
    
    // Конвертация BGR -> RGB и нормализация [0, 255] -> [0, 1]
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);
    
    // Конвертация cv::Mat в torch::Tensor
    torch::Tensor tensor_image = torch::from_blob(
        image.data, 
        {1, image.rows, image.cols, 3}, 
        torch::kFloat32
    );
    
    // Перестановка размерностей (NHWC -> NCHW)
    tensor_image = tensor_image.permute({0, 3, 1, 2});
    
    return tensor_image;
}

// Функция для постобработки выхода модели YOLOv5
void postprocess(
    torch::Tensor &output, 
    float conf_threshold = 0.5, 
    const std::string& target_class = "person"
) {
    // Словарь классов COCO (80 классов)
    std::vector<std::string> classes = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", 
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", 
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    // Получаем размеры выхода модели
    auto detections = output.squeeze(0);  // Удаляем batch-размерность

    // Фильтруем детекции по confidence и классу
    for (int i = 0; i < detections.size(0); ++i) {
        float confidence = detections[i][4].item<float>();
        
        if (confidence < conf_threshold) 
            continue;

        // Находим класс с максимальной вероятностью
        auto class_probs = detections[i].slice(5, 5 + classes.size());
        int class_id = class_probs.argmax().item<int>();
        float class_score = class_probs[class_id].item<float>();
        std::string class_name = classes[class_id];

        // Пропускаем, если класс не "person"
        if (class_name != target_class)
            continue;

        // Координаты bounding box (x1, y1, x2, y2)
        float x1 = detections[i][0].item<float>();
        float y1 = detections[i][1].item<float>();
        float x2 = detections[i][2].item<float>();
        float y2 = detections[i][3].item<float>();

        // Выводим информацию в консоль
        std::cout << "Detected " << class_name 
                  << " (Confidence: " << confidence * class_score 
                  << ") at [" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << "]" << std::endl;
    }
}

int main() {
    // Загрузка модели YOLOv5n
    torch::jit::script::Module model;
    try {
        model = torch::jit::load("yolov5n.torchscript.pt");
        model.eval();
    } catch (const c10::Error &e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return -1;
    }

    // Открытие видеопотока с камеры
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream" << std::endl;
        return -1;
    }

    // Переменные для замера FPS
    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error reading frame" << std::endl;
            break;
        }

        // Уменьшение разрешения (например, до 640x480)
        cv::resize(frame, frame, cv::Size(640, 480));

        // Предобработка кадра
        torch::Tensor input_tensor = preprocess(frame);
        
        // Запуск модели
        torch::Tensor output;
        try {
            output = model.forward({input_tensor}).toTensor();
        } catch (const c10::Error &e) {
            std::cerr << "Error running the model: " << e.what() << std::endl;
            break;
        }

        // Постобработка и вывод результатов
        postprocess(output, 0.5, "person");

        // Подсчет FPS
        frame_count++;
        if (frame_count % 10 == 0) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            double fps = 10000.0 / duration;  // 10 кадров / время в секундах
            std::cout << "FPS: " << fps << std::endl;
            start_time = end_time;
            frame_count = 0;
        }

        // Выход по нажатию 'q'
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    return 0;
}