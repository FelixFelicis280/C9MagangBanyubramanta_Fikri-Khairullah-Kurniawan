#include <cmath>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include <ignition/transport.hh>
#include <ignition/msgs.hh>

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
        
        // Initialize Ignition Transport
        // Topic for applying forces: /model/robot/link/base_link/wrench
        wrench_pub_ = ign_node_.Advertise<ignition::msgs::Wrench>(
            "/world/world1/model/robot/link/base_link/wrench"
        );
        
        RCLCPP_INFO(this->get_logger(), "Controller initialized with Ignition Transport");
        
        if (!wrench_pub_) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create Ignition publisher!");
        }
    }
    
    private:
    double x_pos = 0;
    double y_pos = 0;
    double yaw = 0;
    double depth = 0;
    
    double prev_x_pos = 0;
    double prev_y_pos = 0;
    double prev_yaw = 0;
    double prev_depth = 0;
    
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
    
    // Ignition Transport
    ignition::transport::Node ign_node_;
    ignition::transport::Node::Publisher wrench_pub_;
    
    const double kp_linear = 50.0;  // Proportional gain for linear motion
    const double kd_linear = 10.0;  // Derivative gain for linear motion
    const double kp_angular = 20.0; // Proportional gain for angular motion
    const double kd_angular = 5.0;  // Derivative gain for angular motion
    
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
        
        // X position control
        if(x_input > joy_threshold && x_pos < X_POS_UPPER){
            x_pos += xy_speed;
        }
        if(x_input < -joy_threshold && x_pos > X_POS_LOWER){
            x_pos -= xy_speed;
        }
        if(x_input < joy_threshold && x_input > -joy_threshold){
            if(x_pos < xy_threshold && x_pos > -xy_threshold){
                x_pos = 0;
            }else if(x_pos > 0){
                x_pos -= xy_return_speed;
            }else if(x_pos < 0){
                x_pos += xy_return_speed;
            }
        }
        
        // Y position control
        if(y_input > joy_threshold && y_pos < Y_POS_UPPER){
            y_pos += xy_speed;
        }
        if(y_input < -joy_threshold && y_pos > Y_POS_LOWER){
            y_pos -= xy_speed;
        }
        if(y_input < joy_threshold && y_input > -joy_threshold){
            if(y_pos < xy_threshold && y_pos > -xy_threshold){
                y_pos = 0;
            }else if(y_pos > 0){
                y_pos -= xy_return_speed;
            }else if(y_pos < 0){
                y_pos += xy_return_speed;
            }
        }
        
        // Yaw control
        if(yaw_input > joy_threshold && yaw < YAW_UPPER){
            yaw += yaw_speed;
        }
        if(yaw_input < -joy_threshold && yaw > YAW_LOWER){
            yaw -= yaw_speed;
        }
        
        // Depth control
        if(depth_input > joy_threshold && depth < DEPTH_UPPER){
            depth += 0.1;
        }else if(depth > DEPTH_UPPER){
            depth = DEPTH_UPPER;
        }
        if(depth_input < -joy_threshold && depth > DEPTH_LOWER){
            depth -= 0.1;
        }else if(depth < DEPTH_LOWER){
            depth = DEPTH_LOWER;
        }
        
        // Calculate derivatives (velocity)
        double dx = x_pos - prev_x_pos;
        double dy = y_pos - prev_y_pos;
        double dyaw = yaw - prev_yaw;
        double ddepth = depth - prev_depth;
        
        // PD control to calculate forces
        double force_x = kp_linear * x_pos + kd_linear * dx;
        double force_y = kp_linear * y_pos + kd_linear * dy;
        double force_z = kp_linear * depth + kd_linear * ddepth;
        double torque_z = kp_angular * (yaw * M_PI / 180.0) + kd_angular * (dyaw * M_PI / 180.0);
        
        // Create and publish Ignition wrench message
        ignition::msgs::Wrench wrench_msg;
        ignition::msgs::Vector3d *force = wrench_msg.mutable_force();
        ignition::msgs::Vector3d *torque = wrench_msg.mutable_torque();
        
        force->set_x(force_x);
        force->set_y(force_y);
        force->set_z(force_z);
        
        torque->set_x(0.0);
        torque->set_y(0.0);
        torque->set_z(torque_z);
        
        if (wrench_pub_) {
            wrench_pub_.Publish(wrench_msg);
        }
        
        // Update previous values
        prev_x_pos = x_pos;
        prev_y_pos = y_pos;
        prev_yaw = yaw;
        prev_depth = depth;
        
        // RCLCPP_INFO(this->get_logger(), 
        //     "Position: x=%.2f, y=%.2f, yaw=%.2f, depth=%.2f | Forces: fx=%.2f, fy=%.2f, fz=%.2f, tz=%.2f",
        //     x_pos, y_pos, yaw, depth, force_x, force_y, force_z, torque_z);
    }
};

int main(int argc, char **argv){
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Controller>());
    rclcpp::shutdown();
    return 0;
}