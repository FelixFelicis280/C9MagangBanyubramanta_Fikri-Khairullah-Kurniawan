#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "interfaces/msg/controller.hpp"

const int X_POS_UPPER = 250;
const int X_POS_LOWER = -250;
const int Y_POS_UPPER = 250;
const int Y_POS_LOWER = -250;
const int YAW_UPPER = 180;
const int YAW_LOWER = -180;
const double DEPTH_UPPER = 10;
const double DEPTH_LOWER = 0;

class Controller : public rclcpp::Node{
    public:
    Controller() : Node("controller")
    {
        joy_sub_ = this->create_subscription<sensor_msgs::msg::Joy>(
            "joy",
            10,
            std::bind(&Controller::joy_callback, this, std::placeholders::_1)
        );
        log_pub_ = this->create_publisher<interfaces::msg::Controller>("/cmd_vel", 10);
    }
    
    private:
    double x_pos = 0;
    double y_pos = 0;
    double yaw = 0;
    double depth = 0;

    // Create publisher and subscriber
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
    rclcpp::Publisher<interfaces::msg::Controller>::SharedPtr log_pub_;

    // Function called every interval
    void joy_callback(const sensor_msgs::msg::Joy::SharedPtr joy)
    {
        int x_input = joy->axes[0];
        int y_input = joy->axes[1];
        int yaw_input = joy->axes[3];
        double depth_input = joy->axes[4];
        int xy_speed = 5;
        int xy_return_speed = 3;
        int yaw_speed = 4;
        int yaw_return_speed = 2;

        double joy_threshold = 0.1;
        int xy_threshold = 3;

        if(x_input > joy_threshold and x_pos < X_POS_UPPER){
            x_pos += xy_speed;
        }
        if(x_input < -joy_threshold and x_pos > X_POS_LOWER){
            x_pos -= xy_speed;
        }
        if(x_input < joy_threshold and x_input > -joy_threshold){
            if(x_pos < xy_threshold and x_pos > -xy_threshold){
                x_pos = 0;
            }else if(x_pos > 0){
                x_pos -= xy_return_speed;
            }else if(x_pos < 0){
                x_pos += xy_return_speed;
            }
        }

        if(y_input > joy_threshold and y_pos < Y_POS_UPPER){
            y_pos += xy_speed;
        }
        if(y_input < -joy_threshold and y_pos > Y_POS_LOWER){
            y_pos -= xy_speed;
        }
        if(y_input < joy_threshold and y_input > -joy_threshold){
            if(y_pos < xy_threshold and y_pos > -xy_threshold){
                y_pos = 0;
            }else if(y_pos > 0){
                y_pos -= xy_return_speed;
            }else if(y_pos < 0){
                y_pos += xy_return_speed;
            }
        }

        if(yaw_input > joy_threshold and yaw < YAW_UPPER){
            yaw += yaw_speed;
        }
        if(yaw_input < -joy_threshold and yaw > YAW_LOWER){
            yaw -= yaw_speed;
        }

        if(depth_input > joy_threshold and depth < DEPTH_UPPER){
            depth += 0.1;
        }else if(depth > DEPTH_UPPER){
            depth = DEPTH_UPPER;
        }

        if(depth_input < -joy_threshold and depth > DEPTH_LOWER){
            depth -= 0.1;
        }else if(depth < DEPTH_LOWER){
            depth = DEPTH_LOWER;
        }

        // Publish as interface
        interfaces::msg::Controller msg;
        msg.x = x_pos;
        msg.y = y_pos;
        msg.yaw = yaw;
        msg.depth = depth;
        log_pub_->publish(msg);
    }
    
};

int main(int argc, char **argv){
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Controller>());
    rclcpp::shutdown();
    return 0;
}
