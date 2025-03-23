#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <iostream>

// Функция для предобработки изображения
cv::Mat preprocess_image(cv::Mat& image, int input_width, int input_height) {
    cv::Mat resized_image, blob;
    cv::resize(image, resized_image, cv::Size(input_width, input_height));
    cv::dnn::blobFromImage(resized_image, blob, 1.0 / 255.0, cv::Size(input_width, input_height), cv::Scalar(0, 0, 0), true, false);
    return blob;
}

// Функция для постобработки результатов YOLO
void postprocess_results(const std::vector<float>& output, int input_width, int input_height, int original_width, int original_height) {
    // Пример постобработки (зависит от формата вывода модели)
    // Здесь нужно распарсить выход модели и отфильтровать результаты по классу 0
    // Вывести bounding boxes, классы и confidence scores в консоль
}

int main() {
    // Параметры модели
    const std::string model_path = "yolov8n.onnx";
    const int input_width = 640;
    const int input_height = 640;

    // Инициализация ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8n");
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path.c_str(), session_options);

    // Инициализация видеозахвата
    cv::VideoCapture cap(0); // 0 - индекс камеры
    if (!cap.isOpened()) {
        std::cerr << "Ошибка открытия камеры!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Предобработка изображения
        cv::Mat blob = preprocess_image(frame, input_width, input_height);

        // Подготовка входных данных для модели
        std::vector<float> input_tensor_values(blob.begin<float>(), blob.end<float>());
        std::vector<int64_t> input_shape = {1, 3, input_height, input_width};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

        // Запуск модели
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, nullptr, &input_tensor, 1);

        // Постобработка результатов
        std::vector<float> output(output_tensors[0].GetTensorMutableData<float>(), output_tensors[0].GetTensorMutableData<float>() + output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount());
        postprocess_results(output, input_width, input_height, frame.cols, frame.rows);

        // Расчет и вывод FPS
        frame_count++;
        if (frame_count % 10 == 0) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            double fps = 1000.0 * frame_count / duration;
            std::cout << "FPS: " << fps << std::endl;
            frame_count = 0;
            start_time = std::chrono::high_resolution_clock::now();
        }

        // Отображение изображения с результатами
        cv::imshow("YOLOv8n Detection", frame);
        if (cv::waitKey(1) == 27) break; // Выход по нажатию ESC
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}