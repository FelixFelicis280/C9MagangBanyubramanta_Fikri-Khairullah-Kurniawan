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

        // // Put output in tensor (Openvino data structure)
        auto output_tensor = infer_request.get_output_tensor();
        const float* data = output_tensor.data<const float>();
        // [cx, cy, w, h, objectness, class0_prob, class1_prob]
        //  0   1   2  3      4           5            6x   
        
        vision_msgs::msg::Detection2DArray detections_msg;
        detections_msg.header.stamp = this->now();
        detections_msg.header.frame_id = "camera_frame";

        float scale_x = static_cast<float>(frame.cols) / INPUT_WIDTH;
        float scale_y = static_cast<float>(frame.rows) / INPUT_HEIGHT;
        // Frame size 1280 x 720
        
        std::vector<vision_msgs::msg::Detection2D> all_detections;

        auto output_shape = output_tensor.get_shape(); // [1, N, D]
        size_t num_detections = output_shape[1];
        size_t num_classes = output_shape[2]; // D = 5 + num_classes

        for (size_t i = 0; i < num_detections; i++) {
            const float* det = data + i * num_classes;
            float x_center = det[0];
            float y_center = det[1];
            float w = det[2];
            float h = det[3];
            float obj_conf = det[4];
            
            float conf_threshold = 0.5;
            // Filter low confidence
            if (obj_conf < conf_threshold) continue;
            
            // find best class
            int best_class = -1;
            float best_class_score = 0.0f;
            for (size_t c = 5; c < num_classes; ++c) {
                if (det[c] > best_class_score) {
                    best_class_score = det[c];
                    best_class = static_cast<int>(c - 5);
                }
            }

            float conf = obj_conf * best_class_score;
            
            // Debugging
            // RCLCPP_INFO(this->get_logger(), 
            //     "xywh=[%.3f,%.3f,%.3f,%.3f] confs=[%.3f,%.3f,%.3f,%i]",
            //     x_center, y_center, w, h,  // xywh
            //     data[i * 7 + 4], data[i * 7 + 5], data[i * 7 + 6], best_class  // objectness, class0_prob, class1_prob
            // );

            // Create detection messages
            vision_msgs::msg::Detection2D detection;
            detection.bbox.center.position.x = x_center * scale_x;
            detection.bbox.center.position.y = y_center * scale_y;
            detection.bbox.size_x = w * scale_x;
            detection.bbox.size_y = h * scale_y;
        
            vision_msgs::msg::ObjectHypothesisWithPose hyp;
            hyp.hypothesis.class_id = std::to_string(best_class);
            hyp.hypothesis.score = conf;
            detection.results.push_back(hyp);

            all_detections.push_back(detection);
        }

        auto kept_indices = applyNMS(all_detections, 0.5f);
        for (auto idx : kept_indices) {
            detections_msg.detections.push_back(all_detections[idx]);
        }
        
        // RCLCPP_INFO(this->get_logger(), "=== Detections: %zu ===", detections_msg.detections.size());
        
        // for (size_t i = 0; i < detections_msg.detections.size(); i++) {
        //     const auto& det = detections_msg.detections[i];
            
        //     RCLCPP_INFO(this->get_logger(),
        //         "Detection %zu: center=[%.1f, %.1f] size=[%.1f, %.1f] class=%s score=%.3f",
        //         i,
        //         det.bbox.center.position.x,
        //         det.bbox.center.position.y,
        //         det.bbox.size_x,
        //         det.bbox.size_y,
        //         det.results.empty() ? "N/A" : det.results[0].hypothesis.class_id.c_str(),
        //         det.results.empty() ? 0.0 : det.results[0].hypothesis.score
        //     );
        // }
        
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