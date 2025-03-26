#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <fstream>

// Параметры детекции
const float CONFIDENCE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.4;
const int INPUT_WIDTH = 416;
const int INPUT_HEIGHT = 416;

// Цвета для отрисовки
const cv::Scalar GREEN(0, 255, 0);
const cv::Scalar RED(0, 0, 255);

std::vector<std::string> loadClassNames(const std::string& filename) {
    std::vector<std::string> classes;
    std::ifstream ifs(filename);
    std::string line;
    while (std::getline(ifs, line)) {
        classes.push_back(line);
    }
    return classes;
}

void drawPrediction(cv::Mat& frame, int classId, float confidence, 
                   int left, int top, int right, int bottom, 
                   const std::vector<std::string>& classes) {
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), GREEN, 2);
    
    std::string label = cv::format("%s: %.2f", classes[classId].c_str(), confidence);
    cv::putText(frame, label, cv::Point(left, top - 5), 
               cv::FONT_HERSHEY_SIMPLEX, 0.5, RED, 1);
}

int main() {
    // Загрузка модели YOLOv3
    cv::dnn::Net net = cv::dnn::readNetFromDarknet("yolov3.cfg", "yolov3.weights");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Загрузка классов COCO
    std::vector<std::string> classes = loadClassNames("coco.names");
    if (classes.empty()) {
        std::cerr << "Не удалось загрузить файл классов!" << std::endl;
        return -1;
    }

    // Открытие камеры
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Не удалось открыть камеру!" << std::endl;
        return -1;
    }

    cv::namedWindow("YOLO Object Detection", cv::WINDOW_NORMAL);

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // Подготовка изображения
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), 
                             cv::Scalar(), true, false);
        net.setInput(blob);

        // Получение результатов
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // Обработка результатов
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        for (const auto& output : outputs) {
            float* data = (float*)output.data;
            for (int i = 0; i < output.rows; i++, data += output.cols) {
                cv::Mat scores = output.row(i).colRange(5, output.cols);
                cv::Point classIdPoint;
                double confidence;
                cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                
                if (confidence > CONFIDENCE_THRESHOLD) {
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

        // Применение Non-Maximum Suppression
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

        // Отрисовка результатов
        for (int idx : indices) {
            cv::Rect box = boxes[idx];
            drawPrediction(frame, classIds[idx], confidences[idx], 
                         box.x, box.y, box.x + box.width, box.y + box.height, 
                         classes);
        }

        cv::imshow("YOLO Object Detection", frame);
        if (cv::waitKey(1) == 27) break; // ESC для выхода
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}