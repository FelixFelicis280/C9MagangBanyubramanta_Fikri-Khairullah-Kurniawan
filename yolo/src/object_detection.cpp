#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include <opencv2/opencv.hpp>
// #include <opencv2/dnn.hpp>
#include "cv_bridge/cv_bridge.h"
#include <iostream>
#include <openvino/openvino.hpp>

class ObjectDetection : public rclcpp::Node{
    public:
    ObjectDetection() : Node("object_detection"){
        // Openvino Initialization
        ov::Core core;
        model = core.read_model("/home/felix/Desktop/kpp_yolo_ws/src/yolo/resources/openvino/best.xml");
        
        // Preprocessing Configuration
        ov::preprocess::PrePostProcessor ppp(model);
        ppp.input().tensor()
            .set_element_type(ov::element::u8)
            .set_layout("NHWC")
            .set_color_format(ov::preprocess::ColorFormat::BGR);
        ppp.input().preprocess()
            .convert_element_type(ov::element::f32)
            .convert_color(ov::preprocess::ColorFormat::BGR)
            .scale(255.0f);
        ppp.input().model().set_layout("NCHW");
        model = ppp.build();
        
        // Compile Model
        compiled_model = core.compile_model(model, "CPU");
        infer_request = compiled_model.create_infer_request();

        // ROS2
        detection_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("detections", 10);
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("raw_video", 10);
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&ObjectDetection::timerCallback, this)
        );
        cap_.open("/home/felix/Desktop/kpp_yolo_ws/src/yolo/resources/fourth.mp4");
    }

    private:
    std::vector<size_t> applyNMS(
        const std::vector<vision_msgs::msg::Detection2D>& detections, 
        float iou_threshold = 0.5){
            std::vector<size_t> indices;
            std::vector<bool> used(detections.size(), false);
            
            for (size_t i = 0; i < detections.size(); i++){
                if (used[i]) continue;
                
                indices.push_back(i);
                
                for (size_t j = i + 1; j < detections.size(); j++) {
                    if (used[j]) continue;
                    
                    // Calculate IoU (overlap)
                    auto& det1 = detections[i];
                    auto& det2 = detections[j];
                    
                    float x1 = std::max(det1.bbox.center.position.x - det1.bbox.size_x/2, 
                                    det2.bbox.center.position.x - det2.bbox.size_x/2);
                    float y1 = std::max(det1.bbox.center.position.y - det1.bbox.size_y/2, 
                                    det2.bbox.center.position.y - det2.bbox.size_y/2);
                    float x2 = std::min(det1.bbox.center.position.x + det1.bbox.size_x/2, 
                                    det2.bbox.center.position.x + det2.bbox.size_x/2);
                    float y2 = std::min(det1.bbox.center.position.y + det1.bbox.size_y/2, 
                                    det2.bbox.center.position.y + det2.bbox.size_y/2);
                    
                    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
                    float area1 = det1.bbox.size_x * det1.bbox.size_y;
                    float area2 = det2.bbox.size_x * det2.bbox.size_y;
                    float iou = intersection / (area1 + area2 - intersection);
                    
                    if (iou > iou_threshold) {
                        used[j] = true;
                    }
                }
        }
        return indices;
    }

    
    void timerCallback(){
        cv::Mat frame;
        cap_ >> frame;
        if(frame.empty()) return;

        // Publish raw image
        auto img_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
        img_msg->header.stamp = this->now();
        image_pub_->publish(*img_msg);

        // Preprocess image for model
        cv::Mat resized;
        const int INPUT_WIDTH = 512;
        const int INPUT_HEIGHT = 512;
        cv::resize(frame, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));

        // Run inference (run new video through model)
        ov::Tensor input_tensor(ov::element::u8, {1, INPUT_HEIGHT, INPUT_WIDTH, 3}, resized.data);
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();

        // Put output in tensor (Openvino data structure)
        auto output_tensor = infer_request.get_output_tensor();
        const float* data = output_tensor.data<const float>();
        // Example data [x0, y0, w0, h0, conf0, class0, unknown0, x1, y1, w1, h1, conf1, class1, unknown1, ...]
        auto output_shape = output_tensor.get_shape();
        size_t num_detections = output_shape[1];

        // Debugging
        RCLCPP_INFO(this->get_logger(), "=== Checking confidence locations ===");
        for (size_t i = 0; i < std::min(num_detections, (size_t)10); i++) {
            RCLCPP_INFO(this->get_logger(), 
                "Det[%ld]: xywh=[%.3f,%.3f,%.3f,%.3f] confs=[%.3f,%.3f,%.3f]", 
                i,
                data[i * 7 + 0], data[i * 7 + 1], data[i * 7 + 2], data[i * 7 + 3],  // xywh
                data[i * 7 + 4], data[i * 7 + 5], data[i * 7 + 6]  // potential confidence scores
            );
        }
        
        vision_msgs::msg::Detection2DArray detections_msg;
        detections_msg.header.stamp = this->now();
        detections_msg.header.frame_id = "camera_frame";

        float scale_x = static_cast<float>(frame.cols) / INPUT_WIDTH;
        float scale_y = static_cast<float>(frame.rows) / INPUT_HEIGHT;
        
        std::vector<vision_msgs::msg::Detection2D> all_detections;

        for (size_t i = 0; i < num_detections; i++) {
            
            float x = data[i* 7 + 0];
            float y = data[i* 7 + 1];
            float w = data[i* 7 + 2];
            float h = data[i* 7 + 3];
            float conf = data[i * 7 + 4];
            int classId = static_cast<int>(data[i* 7 + 5]);
            
            // Filter low confidence
            if (conf < 0.5) continue;
            
            // Filter invalid detections
            if (x > 1 || x < 0 || y > 1 || y < 0 || w > 1 || w <= 0 || h > 1 || h <= 0) {
                continue;
            }

            // Filter small setections
            if (w * scale_x < 20 || h * scale_y < 20) {
                continue;
            }
            
            // Create detection messages
            vision_msgs::msg::Detection2D det;
            det.bbox.center.position.x = (x + w/2) * scale_x;
            det.bbox.center.position.y = (y + h/2) * scale_y;
            det.bbox.size_x = w * scale_x;
            det.bbox.size_y = h * scale_y;
        
            vision_msgs::msg::ObjectHypothesisWithPose hyp;
            hyp.hypothesis.class_id = std::to_string(classId);
            hyp.hypothesis.score = conf;
            det.results.push_back(hyp);

            all_detections.push_back(det);
        }

        auto kept_indices = applyNMS(all_detections, 0.5f);
        for (auto idx : kept_indices) {
            detections_msg.detections.push_back(all_detections[idx]);
        }
        
        detection_pub_->publish(detections_msg);
        RCLCPP_INFO(this->get_logger(), "Published %zu detections", detections_msg.detections.size());
    }
    
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    cv::VideoCapture cap_;

};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ObjectDetection>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}