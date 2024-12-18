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
    // patchcoreģ��ѵ�������ļ�����center_cropΪ `center_crop: null`
    string model_path ="D:/Unsupervised/anomalib/results/pdweights/onnx/model.onnx";
    string image_path = "D://1//1644647_1.bmp";
    string device = "CPU";
    bool openvino_preprocess = true;    // �Ƿ�ʹ��openvinoͼƬԤ����,ʹ��dynamic shape����Ҫ��openvino_preprocess

    cv::Mat Image = cv::imread(image_path);
    Inference infer;
    infer.InitNet(model_path, device, openvino_preprocess);
    infer.infer(Image);

    return 0;
}