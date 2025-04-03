#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <sys/mman.h>
#include <fcntl.h>
#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/stream.h>
#include <libcamera/formats.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

using namespace libcamera;
using namespace std;
using namespace cv;
using namespace dnn;

// Явные псевдонимы для избежания конфликтов
using CV_Size = cv::Size;
using CV_Point = cv::Point;

// YOLO parameters
const float CONFIDENCE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.4;
const int INPUT_WIDTH = 224;  // Оптимальный размер для скорости
const int INPUT_HEIGHT = 224; // Оптимальный размер для скорости
const string CLASSES_FILE = "coco.names";
const string MODEL_CONFIG = "yolov4-tiny.cfg";
const string MODEL_WEIGHTS = "yolov4-tiny.weights";

// Поддерживаемые разрешения
const vector<CV_Size> SUPPORTED_RESOLUTIONS = {
    {224, 224},   // Квадратное для YOLO
    {320, 240},   // Низкое разрешение
    {640, 480}    // Среднее разрешение
};

class Detector {
public:
    Detector() {
        // Load YOLO model
        try {
            net = readNetFromDarknet(MODEL_CONFIG, MODEL_WEIGHTS);
            net.setPreferableBackend(DNN_BACKEND_OPENCV);
            net.setPreferableTarget(DNN_TARGET_CPU);
        } catch (const Exception& e) {
            cerr << "Error loading YOLO model: " << e.what() << endl;
            exit(1);
        }

        // Load class names
        ifstream ifs(CLASSES_FILE);
        string line;
        while (getline(ifs, line)) {
            classes.push_back(line);
        }
    }

    void detect(Mat& frame) {
        Mat blob;
        // Быстрая предобработка с уменьшением размера
        blobFromImage(frame, blob, 1./255., CV_Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

        net.setInput(blob);

        vector<Mat> outputs;
        try {
            net.forward(outputs, getOutputsNames());
        } catch (const Exception& e) {
            cerr << "Detection error: " << e.what() << endl;
            return;
        }

        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes;

        for (const auto& output : outputs) {
            const float* data = (float*)output.data;
            for (int i = 0; i < output.rows; ++i, data += output.cols) {
                Mat scores = output.row(i).colRange(5, output.cols);
                CV_Point classIdPoint;
                double confidence;
                minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);

                // Фильтрация только людей на раннем этапе
                if (confidence > CONFIDENCE_THRESHOLD && classes[classIdPoint.x] == "person") {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }

        vector<int> indices;
        NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

        for (int idx : indices) {
            Rect box = boxes[idx];
            cout << "Person detected | Confidence: " << confidences[idx]
                 << " | Box: [" << box.x << ", " << box.y << ", "
                 << box.width << ", " << box.height << "]" << endl;
        }
    }

private:
    Net net;
    vector<string> classes;

    vector<String> getOutputsNames() {
        static vector<String> names;
        if (names.empty()) {
            vector<int> outLayers = net.getUnconnectedOutLayers();
            vector<String> layersNames = net.getLayerNames();
            names.resize(outLayers.size());
            for (size_t i = 0; i < outLayers.size(); ++i) {
                names[i] = layersNames[outLayers[i] - 1];
            }
        }
        return names;
    }
};

class FrameRateCalculator {
public:
    void tick() {
        frameCount_++;
        auto now = chrono::steady_clock::now();
        auto elapsed = chrono::duration_cast<chrono::milliseconds>(now - lastTime_).count();

        if (elapsed >= 1000) {
            fps_ = frameCount_ / (elapsed / 1000.0);
            frameCount_ = 0;
            lastTime_ = now;
            cout << "FPS: " << fps_ << endl;
        }
    }

private:
    int frameCount_ = 0;
    double fps_ = 0;
    chrono::steady_clock::time_point lastTime_ = chrono::steady_clock::now();
};

CV_Size selectResolution() {
    cout << "Available resolutions (higher = lower FPS):" << endl;
    for (size_t i = 0; i < SUPPORTED_RESOLUTIONS.size(); ++i) {
        cout << i+1 << ". " << SUPPORTED_RESOLUTIONS[i].width << "x" << SUPPORTED_RESOLUTIONS[i].height << endl;
    }

    int choice;
    do {
        cout << "Select resolution (1-" << SUPPORTED_RESOLUTIONS.size() << "): ";
        cin >> choice;
    } while (choice < 1 || choice > SUPPORTED_RESOLUTIONS.size());

    return SUPPORTED_RESOLUTIONS[choice-1];
}

int main() {
    // Выбор разрешения
    CV_Size selectedResolution = selectResolution();
    cout << "Selected resolution: " << selectedResolution.width << "x" << selectedResolution.height << endl;

    Detector detector;
    FrameRateCalculator fpsCounter;

    unique_ptr<CameraManager> cm = make_unique<CameraManager>();
    cm->start();

    if (cm->cameras().empty()) {
        cerr << "No cameras found!" << endl;
        return 1;
    }

    shared_ptr<Camera> camera = cm->get(cm->cameras()[0]->id());
    if (!camera || camera->acquire()) {
        cerr << "Failed to acquire camera" << endl;
        return 1;
    }

    unique_ptr<CameraConfiguration> config = camera->generateConfiguration({StreamRole::VideoRecording});
    if (!config) {
        cerr << "Failed to generate camera configuration" << endl;
        return 1;
    }

    // Устанавливаем выбранное разрешение
    StreamConfiguration &streamConfig = config->at(0);
    streamConfig.pixelFormat = PixelFormat::fromString("YUV420");
    streamConfig.size = {static_cast<uint32_t>(selectedResolution.width),
                        static_cast<uint32_t>(selectedResolution.height)};
    streamConfig.bufferCount = 4;  // Оптимальное количество буферов

    if (config->validate() == CameraConfiguration::Invalid) {
        cerr << "Failed to validate stream configuration" << endl;
        return 1;
    }

    if (camera->configure(config.get())) {
        cerr << "Failed to configure camera" << endl;
        return 1;
    }

    FrameBufferAllocator allocator(camera);
    if (allocator.allocate(config->at(0).stream()) < 0) {
        cerr << "Failed to allocate buffers" << endl;
        return 1;
    }

    vector<unique_ptr<Request>> requests;
    for (unsigned int i = 0; i < config->at(0).bufferCount; i++) {
        unique_ptr<Request> request = camera->createRequest(i);
        if (!request || request->addBuffer(config->at(0).stream(), allocator.buffers(config->at(0).stream())[i].get())) {
            cerr << "Failed to create request" << endl;
            return 1;
        }
        requests.push_back(move(request));
    }

    camera->requestCompleted.connect(camera.get(), [&](Request *request) {
        FrameBuffer *buffer = request->buffers().begin()->second;
        size_t size = buffer->metadata().planes()[0].bytesused;
        int fd = buffer->planes()[0].fd.get();

        void *data = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
        if (data == MAP_FAILED) {
            perror("mmap failed");
            return;
        }

        // Используем выбранное разрешение
        Mat yuv(selectedResolution.height * 3/2, selectedResolution.width, CV_8UC1, data);
        Mat rgb;
        cvtColor(yuv, rgb, COLOR_YUV2BGR_I420);

        // Обнаружение людей
        detector.detect(rgb);
        fpsCounter.tick();

        munmap(data, size);
        request->reuse(Request::ReuseBuffers);
        camera->queueRequest(request);
    });

    if (camera->start()) {
        cerr << "Failed to start camera" << endl;
        return 1;
    }

    for (auto &request : requests) {
        camera->queueRequest(request.get());
    }

    cout << "Detection started. Press Ctrl+C to stop." << endl;

    while (true) {
        this_thread::sleep_for(chrono::seconds(1));
    }

    camera->stop();
    allocator.free(config->at(0).stream());
    camera->release();
    cm->stop();

    return 0;
}