#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <fstream>  // Добавьте эту строку
#include <string>  // Для std::string

using namespace cv;
using namespace dnn;
using namespace std;
using namespace std::chrono;

// Функция для отрисовки bounding boxes и вывода информации в консоль
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, vector<string> classes)
{
    // Отрисовка bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 3);
    
    // Получение метки класса и confidence
    string label = format("%.2f", conf);
    if (!classes.empty() && classId < (int)classes.size())
    {
        label = classes[classId] + ": " + label;
        
        // Вывод информации в консоль
        cout << "Detected: " << classes[classId] << " | Confidence: " << conf 
             << " | Box: [" << left << ", " << top << ", " << right << ", " << bottom << "]" << endl;
    }
    
    // Отображение label на frame
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height), Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}

int main()
{
    // Загрузка модели YOLOv8n
    string modelPath = "yolov8n.onnx"; // Убедитесь, что модель находится в этой директории
    Net net = readNet(modelPath);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // Загрузка классов COCO
    vector<string> classes;
    string classesFile = "coco.names"; // Файл с именами классов COCO
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    // Открытие видеопотока с камеры
    VideoCapture cap(0); // 0 для встроенной камеры, или укажите путь к видеофайлу
    if (!cap.isOpened())
    {
        cerr << "Error: Could not open camera/video." << endl;
        return -1;
    }

    // Переменные для расчета FPS
    auto start = high_resolution_clock::now();
    int frameCount = 0;
    float fps = 0;

    Mat frame, blob;
    while (true)
    {
        cap >> frame;
        if (frame.empty())
        {
            cerr << "Error: Blank frame grabbed." << endl;
            break;
        }

        // Подготовка кадра для нейронной сети
        blobFromImage(frame, blob, 1/255.0, Size(640, 640), Scalar(), true, false);
        net.setInput(blob);

        // Прямой проход (forward pass) для получения детекций
        vector<Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // Размеры кадра
        int rows = outputs[0].size[1];
        int dimensions = outputs[0].size[2];

        // Проверка формата вывода (должен быть 1x84x8400 для YOLOv8)
        if (dimensions != 84)
        {
            cerr << "Error: Unexpected output dimensions. Make sure you're using YOLOv8 model." << endl;
            return -1;
        }

        // Преобразование выхода в удобный формат (1x84x8400 -> 8400x84)
        outputs[0] = outputs[0].reshape(1, dimensions);
        Mat output = outputs[0].t();

        // Размеры оригинального кадра
        float x_factor = frame.cols / 640.0;
        float y_factor = frame.rows / 640.0;

        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes;

        // Обработка выхода
        for (int i = 0; i < output.rows; i++)
        {
            Mat scores = output.row(i).colRange(4, 84);
            Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

            if (confidence > 0.5) // Порог confidence
            {
                float cx = output.at<float>(i, 0);
                float cy = output.at<float>(i, 1);
                float w = output.at<float>(i, 2);
                float h = output.at<float>(i, 3);

                int left = int((cx - w/2) * x_factor);
                int top = int((cy - h/2) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }

        // Применение NMS для устранения дублирующих bounding boxes
        vector<int> indices;
        NMSBoxes(boxes, confidences, 0.5, 0.4, indices);

        // Отрисовка результатов
        for (size_t i = 0; i < indices.size(); i++)
        {
            int idx = indices[i];
            Rect box = boxes[idx];
            drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame, classes);
        }

        // Расчет и отображение FPS
        frameCount++;
        if (frameCount >= 10) // Обновляем FPS каждые 10 кадров
        {
            auto end = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(end - start);
            fps = 1000.0 * frameCount / duration.count();
            cout << "FPS: " << fps << endl;
            frameCount = 0;
            start = high_resolution_clock::now();
        }

        // Отображение кадра
        imshow("YOLOv8 Object Detection", frame);

        // Выход по нажатию ESC
        if (waitKey(1) == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}