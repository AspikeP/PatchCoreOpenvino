#include"inference.h"


void Inference::InitNet(string& model_path, string& device, bool openvino_preprocess)
{
    bool efficient_ad = true;
    // 2.创建模型
        // Step 1. Initialize OpenVINO Runtime core
        ov::Core core;
        // Step 2. Read a Model from a Drive
        std::shared_ptr<ov::Model> model = core.read_model(model_path);

        if (openvino_preprocess)
        {
            vector<float> mean;
            vector<float> std;
            if (!efficient_ad) {
                mean = { 0.485 * 255, 0.456 * 255, 0.406 * 255 };
                std = { 0.229 * 255, 0.224 * 255, 0.225 * 255 };
            }
            else {
                mean = { 0., 0., 0. };
                std = { 255., 255., 255. };
            }

            // Step 3. Inizialize Preprocessing for the model
            ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);

            // Specify input image format
            ppp.input(0).tensor()
                .set_color_format(ov::preprocess::ColorFormat::RGB)     // BGR -> RGB
                .set_element_type(ov::element::u8)
                .set_layout(ov::Layout("HWC"));                         // HWC NHWC NCHW

            // Specify preprocess pipeline to input image without resizing
            ppp.input(0).preprocess()
                  //.convert_color(ov::preprocess::ColorFormat::RGB)
                .convert_element_type(ov::element::f32)
                .mean(mean)
                .scale(std);

            // Specify model's input layout
            ppp.input(0).model().set_layout(ov::Layout("NCHW"));

            // Specify output results format
            for (size_t i = 0; i < model->outputs().size(); i++)
            {
                ppp.output(i).tensor().set_element_type(ov::element::f32);
            }

            // Embed above steps in the graph
            model = ppp.build();
        }
        // Step 4. Load the Model to the Device
        compiled_model = core.compile_model(model, device);

    // 3.获取模型的输入输出
    this->inputs = this->compiled_model.inputs();
    this->outputs = this->compiled_model.outputs();

    // 打印输入输出形状
    //dynamic shape model without openvino_preprocess coundn't print input and output shape
    for (auto input : this->inputs) {
        cout << "Input: " << input.get_any_name() << ": [ ";
        for (auto j : input.get_shape()) {
            cout << j << " ";
        }
        cout << "] ";
        cout << "dtype: " << input.get_element_type() << endl;
    }

    for (auto output : this->outputs) {
        cout << "Output: " << output.get_any_name() << ": [ ";
        for (auto j : output.get_shape()) {
            cout << j << " ";
        }
        cout << "] ";
        cout << "dtype: " << output.get_element_type() << endl;
    }

    // 4.创建推理请求
    this->infer_request = this->compiled_model.create_infer_request();
}

void Inference::infer(cv::Mat& image)
{
    while (true)
    {
    // 2.图片预处理
    //cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(512, 512), cv::Scalar(0, 0, 0), true, false);
    cv::Mat blob;

    

    cv::resize(image, blob, { 512, 512 });

    // 3.从图像创建tensor
    ov::Tensor input_tensor = ov::Tensor(this->compiled_model.input(0).get_element_type(),
        this->compiled_model.input(0).get_shape(), (float*)blob.data);
    this->infer_request.set_input_tensor(input_tensor);

    auto start = std::chrono::system_clock::now();
    // 4.推理
    
    this->infer_request.infer();

    auto end = std::chrono::system_clock::now();
    int cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Image: " << " cost: " << cost << " ms." << std::endl;
   
    // 5.获取热力图
    ov::Tensor result1,result2;
    result1 = this->infer_request.get_output_tensor(0);
    result2 = this->infer_request.get_output_tensor(1);
    // cout << result1.get_shape() << endl;    //{1, 1, 224, 224}
    float* out1= result1.data<float>();
    float* out2= result2.data<float>();
    cv::Mat anomaly_map = cv::Mat(cv::Size(448, 448), CV_32FC1, result1.data<float>());

    cv::Mat threshold;
    cv::threshold(anomaly_map, threshold, 45, 255, CV_THRESH_BINARY);

    

    

    

    cv::imwrite("D://1.bmp", anomaly_map);
    cv::imwrite("D://2.bmp", threshold);
    }

}