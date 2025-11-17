#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include <opencv2/opencv.hpp>
// #include <cv_bridge/cv_bridge.h>

class DisplayMask : public rclcpp::Node{
  public:
  DisplayMask() : Node("display_mask"){
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "raw_video", 10,
      std::bind(&DisplayMask::imageCallback, this, std::placeholders::_1)
    );
    
    detection_sub_ = this->create_subscription<vision_msgs::msg::Detection2DArray>(
      "detections", 10,
      std::bind(&DisplayMask::detectionCallback, this, std::placeholders::_1)
    );
  }
  
  private:
  cv::Mat current_frame_;
  vision_msgs::msg::Detection2DArray last_detections_;
  
  cv::Mat rosImageToCvMat(const sensor_msgs::msg::Image::SharedPtr msg){
    int type = 0;
    if (msg->encoding == "rgb8") type = CV_8UC3;
    else if (msg->encoding == "bgr8") type = CV_8UC3;
    else if (msg->encoding == "mono8") type = CV_8UC1;
    else {
        RCLCPP_WARN(this->get_logger(), "Unsupported image encoding: %s", msg->encoding.c_str());
        return cv::Mat();
    }
  
    cv::Mat mat(msg->height, msg->width, type, const_cast<unsigned char*>(msg->data.data()), msg->step);
    if (msg->encoding == "rgb8") cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
    return mat.clone();
  }

  void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg){
    // try{
    //   current_frame_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
    // }catch (cv_bridge::Exception &e){
    //   return;
    // }
    current_frame_ = rosImageToCvMat(msg);
    drawDetections();
  }

  void detectionCallback(const vision_msgs::msg::Detection2DArray::SharedPtr msg){
    last_detections_ = *msg;
  }

  void drawDetections(){
    if(current_frame_.empty()){
      return;
    }
    
    cv::Mat display = current_frame_.clone();
  
    for(const auto &det : last_detections_.detections){
      for (const auto &hyp : det.results){
          double x = det.bbox.center.position.x * current_frame_.cols;
          double y = det.bbox.center.position.y * current_frame_.rows;
          double w = det.bbox.size_x * current_frame_.cols;
          double h = det.bbox.size_y * current_frame_.rows;
          
          cv::Rect box(x - w / 2, y - h / 2, w, h);
          cv::rectangle(display, box, cv::Scalar(0, 255, 0), 2);
  
          std::string label = "ID: " + hyp.hypothesis.class_id +
                              " (" + std::to_string(hyp.hypothesis.score).substr(0, 4) + ")";
                              cv::putText(display, label, cv::Point(box.x, box.y - 5),
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
      }
    }
  
    cv::imshow("Display Mask", display);
    cv::waitKey(1);
  }
  
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detection_sub_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DisplayMask>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
