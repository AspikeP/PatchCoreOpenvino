#include <string>
#include <numeric>
#include <vector>
#include <Windows.h>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include"inference.h"

using namespace std;


int main() {
    // patchcore模型训练配置文件调整center_crop为 `center_crop: null`
    string model_path ="D:/Unsupervised/anomalib/results/pdweights/onnx/model.onnx";
    string image_path = "D://1//1644647_1.bmp";
    string device = "CPU";
    bool openvino_preprocess = true;    // 是否使用openvino图片预处理,使用dynamic shape必须要用openvino_preprocess

    cv::Mat Image = cv::imread(image_path);
    Inference infer;
    infer.InitNet(model_path, device, openvino_preprocess);
    infer.infer(Image);

    return 0;
}