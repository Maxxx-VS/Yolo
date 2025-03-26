#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <fstream>  // Добавлен этот заголовочный файл

// Загрузка классов объектов COCO
std::vector<std::string> loadClassNames() {
    std::vector<std::string> classNames;
    std::ifstream classNamesFile("coco.names");
    if (classNamesFile.is_open()) {
        std::string className;
        while (std::getline(classNamesFile, className)) {
            classNames.push_back(className);
        }
    }
    return classNames;
}

// Обработка выходных слоев YOLO
void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, const std::vector<std::string>& classNames) {
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (const auto& out : outs) {
        float* data = (float*)out.data;
        for (int i = 0; i < out.rows; ++i, data += out.cols) {
            cv::Mat scores = out.row(i).colRange(5, out.cols);
            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);
            
            if (confidence > 0.5) {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    // Применяем non-maximum suppression для устранения дублирующих детекций
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.5, 0.4, indices);

    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        std::string label = cv::format("%s: %.2f", classNames[classIds[idx]].c_str(), confidences[idx]);
        cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, label, cv::Point(box.x, box.y - 10), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
}

int main() {
    // Загрузка модели YOLO
    cv::dnn::Net net = cv::dnn::readNet("yolov4.weights", "yolov4.cfg");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    
    std::vector<std::string> classNames = loadClassNames();
    if (classNames.empty()) {
        std::cerr << "Не удалось загрузить файл coco.names" << std::endl;
        return -1;
    }

    // Настройка камеры
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Ошибка: не удалось открыть камеру!" << std::endl;
        return -1;
    }

    // Выбор разрешения
    std::vector<cv::Size> resolutions = {
        cv::Size(640, 480),
        cv::Size(1280, 720),
        cv::Size(1920, 1080)
    };

    std::cout << "Доступные разрешения:\n";
    for (size_t i = 0; i < resolutions.size(); ++i) {
        std::cout << i + 1 << ". " << resolutions[i].width << "x" << resolutions[i].height << "\n";
    }

    int choice;
    std::cout << "Выберите разрешение (1-" << resolutions.size() << "): ";
    std::cin >> choice;

    if (choice < 1 || choice > resolutions.size()) {
        std::cerr << "Неверный выбор, используется 640x480\n";
        choice = 1;
    }

    cv::Size selectedResolution = resolutions[choice - 1];
    cap.set(cv::CAP_PROP_FRAME_WIDTH, selectedResolution.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, selectedResolution.height);

    cv::namedWindow("YOLO Object Detection", cv::WINDOW_NORMAL);
    cv::resizeWindow("YOLO Object Detection", selectedResolution.width, selectedResolution.height);

    // Получаем имена выходных слоев YOLO
    std::vector<std::string> outputLayers = net.getUnconnectedOutLayersNames();

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // Подготовка изображения для YOLO
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1/255.0, cv::Size(416, 416), 
                                             cv::Scalar(0, 0, 0), true, false);
        net.setInput(blob);
        
        // Прямой проход через сеть
        std::vector<cv::Mat> outs;
        net.forward(outs, outputLayers);

        // Обработка результатов
        postprocess(frame, outs, classNames);

        cv::imshow("YOLO Object Detection", frame);
        if (cv::waitKey(10) == 27) break; // ESC для выхода
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}