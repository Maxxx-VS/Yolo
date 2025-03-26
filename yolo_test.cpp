#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap(0); // Открываем веб-камеру (устройство 0)
    
    if (!cap.isOpened()) {
        std::cerr << "Ошибка: не удалось открыть камеру!" << std::endl;
        return -1;
    }

    // Выводим доступные разрешения (опционально)
    std::cout << "Доступные разрешения камеры:" << std::endl;
    std::vector<cv::Size> resolutions = {
        cv::Size(640, 480),   // VGA
        cv::Size(1280, 720),  // HD
        cv::Size(1920, 1080)  // Full HD
    };
    
    for (size_t i = 0; i < resolutions.size(); ++i) {
        std::cout << i + 1 << ". " << resolutions[i].width << "x" << resolutions[i].height << std::endl;
    }

    // Запрашиваем у пользователя выбор разрешения
    int choice;
    std::cout << "Выберите разрешение (1-" << resolutions.size() << "): ";
    std::cin >> choice;

    if (choice < 1 || choice > resolutions.size()) {
        std::cerr << "Неверный выбор, используется разрешение по умолчанию (640x480)" << std::endl;
        choice = 1;
    }

    cv::Size selectedResolution = resolutions[choice - 1];
    
    // Устанавливаем выбранное разрешение
    cap.set(cv::CAP_PROP_FRAME_WIDTH, selectedResolution.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, selectedResolution.height);

    // Проверяем, применилось ли разрешение (не все камеры поддерживают все режимы)
    double actualWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double actualHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    
    std::cout << "Установленное разрешение: " << actualWidth << "x" << actualHeight << std::endl;

    if (actualWidth != selectedResolution.width || actualHeight != selectedResolution.height) {
        std::cerr << "Предупреждение: камера не поддерживает выбранное разрешение!" << std::endl;
    }

    cv::namedWindow("Webcam", cv::WINDOW_NORMAL);
    cv::resizeWindow("Webcam", selectedResolution.width, selectedResolution.height);

    while (true) {
        cv::Mat frame;
        cap >> frame; // Захватываем кадр
        
        if (frame.empty()) {
            std::cerr << "Ошибка: пустой кадр!" << std::endl;
            break;
        }

        cv::imshow("Webcam", frame);
        
        if (cv::waitKey(10) == 27) { // Выход по ESC
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}