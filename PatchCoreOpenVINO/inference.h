#pragma once
#include <string>
#include <numeric>
#include <vector>
#include <Windows.h>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/dnn.hpp>

using namespace std;



class Inference {
private:
    bool openvino_preprocess;                   // 是否使用openvino图片预处理
    ov::CompiledModel compiled_model;           // 编译好的模型
    ov::InferRequest infer_request;             // 推理请求
    vector<ov::Output<const ov::Node>> inputs;  // 模型的输入列表名称
    vector<ov::Output<const ov::Node>> outputs; // 模型的输出列表名称

public:
    /**
     * @param model_path    模型路径
     * @param meta_path     超参数路径
     * @param device        CPU or GPU 推理
     * @param openvino_preprocess   是否使用openvino图片预处理
     */
    void InitNet(string& model_path, string& device, bool openvino_preprocess);
    /**
     * 推理单张图片
     * @param image 原始图片
     * @return      标准化的并所放到原图热力图和得分
     */
    void infer(cv::Mat& image);

   // cv::Mat preprocess(cv::Mat& image);

   // void postprocess();
};