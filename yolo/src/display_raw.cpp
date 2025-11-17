#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include <opencv2/opencv.hpp>
// #include <cv_bridge/cv_bridge.h>

class DisplayRaw : public rclcpp::Node{
    public:
    DisplayRaw() : Node("display_raw"){
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "raw_video", 10,
            std::bind(&DisplayRaw::imageCallback, this, std::placeholders::_1)
        );
    }
    
    private:
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
        // cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        cv::Mat frame = rosImageToCvMat(msg);
    
        cv::imshow("Raw Video", frame);
        cv::waitKey(1);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DisplayRaw>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}