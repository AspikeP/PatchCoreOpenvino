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
    bool openvino_preprocess;                   // �Ƿ�ʹ��openvinoͼƬԤ����
    ov::CompiledModel compiled_model;           // ����õ�ģ��
    ov::InferRequest infer_request;             // ��������
    vector<ov::Output<const ov::Node>> inputs;  // ģ�͵������б�����
    vector<ov::Output<const ov::Node>> outputs; // ģ�͵�����б�����

public:
    /**
     * @param model_path    ģ��·��
     * @param meta_path     ������·��
     * @param device        CPU or GPU ����
     * @param openvino_preprocess   �Ƿ�ʹ��openvinoͼƬԤ����
     */
    void InitNet(string& model_path, string& device, bool openvino_preprocess);
    /**
     * ������ͼƬ
     * @param image ԭʼͼƬ
     * @return      ��׼���Ĳ����ŵ�ԭͼ����ͼ�͵÷�
     */
    void infer(cv::Mat& image);

   // cv::Mat preprocess(cv::Mat& image);

   // void postprocess();
};