#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <fstream>

const float CONF_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.4;
const int INPUT_SIZE = 640;

std::vector<std::string> loadClasses(const std::string& file) {
    std::vector<std::string> classes;
    std::ifstream ifs(file);
    std::string line;
    while (std::getline(ifs, line)) classes.push_back(line);
    return classes;
}

int main() {
    // Загрузка модели (используем .weights вместо ONNX)
    cv::dnn::Net net = cv::dnn::readNetFromDarknet("yolov5n.cfg", "yolov5n.weights");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    auto classes = loadClasses("coco.names");
    if (classes.empty()) {
        std::cerr << "Не удалось загрузить классы!" << std::endl;
        return -1;
    }

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Не удалось открыть камеру!" << std::endl;
        return -1;
    }

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // Подготовка изображения
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(INPUT_SIZE, INPUT_SIZE), cv::Scalar(), true, false);
        net.setInput(blob);

        // Получение результатов
        std::vector<cv::Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());

        // Обработка результатов
        for (auto& out : outs) {
            float* data = (float*)out.data;
            for (int i = 0; i < out.rows; i++, data += out.cols) {
                cv::Mat scores = out.row(i).colRange(5, out.cols);
                cv::Point classIdPoint;
                double confidence;
                cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                
                if (confidence > CONF_THRESHOLD) {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;
                    
                    cv::rectangle(frame, cv::Rect(left, top, width, height), cv::Scalar(0, 255, 0), 2);
                    std::string label = cv::format("%s: %.2f", classes[classIdPoint.x].c_str(), confidence);
                    cv::putText(frame, label, cv::Point(left, top - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
                }
            }
        }

        cv::imshow("YOLOv5n Detection", frame);
        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}