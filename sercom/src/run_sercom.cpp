#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <asio.hpp>

#include <memory>
#include <string>
#include <thread>
#include <functional>

class SercomNode : public rclcpp::Node {
private:
    // ROS2 Subscriber
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr subscription_;
    
    // ASIO Serial Port
    asio::io_context io_context_;
    asio::serial_port serial_port_;
    std::string port_;
    unsigned int baud_rate_;
    std::thread io_thread_;
    bool serial_connected_;
    
public:
    SercomNode(const std::string& port, unsigned int baud_rate = 115200)
        : Node("run_sercom"), port_(port), baud_rate_(baud_rate), serial_port_(io_context_), serial_connected_(false) {
        
        // Initialize serial connection
        if (initializeSerial()) {
            subscription_ = this->create_subscription<geometry_msgs::msg::Twist>(
                "/cmd_vel", 10,
                std::bind(&SercomNode::cmdVelCallback, this, std::placeholders::_1));
            
            RCLCPP_INFO(this->get_logger(), "Serial port: %s, Baud rate: %d", port.c_str(), baud_rate);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize serial port. Subscriber not created.");
        }
    }
    
    ~SercomNode() {
        // Stop the IO context
        io_context_.stop();
        
        // Join the IO thread if it's joinable
        if (io_thread_.joinable()) {
            io_thread_.join();
        }
        
        // Close serial port
        if (serial_port_.is_open()) {
            serial_port_.close();
            RCLCPP_INFO(this->get_logger(), "Serial port closed");
        }
    }

private:
    bool initializeSerial() {
        try {
            // Open serial port
            serial_port_.open(port_);
            
            // Set serial port options
            serial_port_.set_option(asio::serial_port_base::baud_rate(baud_rate_));
            serial_port_.set_option(asio::serial_port_base::character_size(8));
            serial_port_.set_option(asio::serial_port_base::stop_bits(asio::serial_port_base::stop_bits::one));
            serial_port_.set_option(asio::serial_port_base::parity(asio::serial_port_base::parity::none));
            serial_port_.set_option(asio::serial_port_base::flow_control(asio::serial_port_base::flow_control::none));
            
            serial_connected_ = true;
            
            // Start IO context in separate thread
            io_thread_ = std::thread([this]() {
                try {
                    io_context_.run();
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(this->get_logger(), "IO context error: %s", e.what());
                }
            });
            
            return true;
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize serial port %s: %s", port_.c_str(), e.what());
            serial_connected_ = false;
            return false;
        }
    }
    
    std::string formatCmdVelMessage(const geometry_msgs::msg::Twist& msg) {
        // Format message based on expected data
        char buffer[512];
        snprintf(buffer, sizeof(buffer), "V,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                 msg.linear.x, msg.linear.y, msg.linear.z,
                 msg.angular.x, msg.angular.y, msg.angular.z);
        return std::string(buffer);
    }

    void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        if (!serial_connected_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
                                "Serial port not connected.");
            return;
        }
        
        try {
            std::string command = formatCmdVelMessage(*msg);
            
            // Send command over serial
            asio::write(serial_port_, asio::buffer(command.c_str(), command.length()));
            
            RCLCPP_DEBUG(this->get_logger(), "Sent cmd_vel: lin.x=%.2f, ang.z=%.2f", 
                        msg->linear.x, msg->angular.z);
                        
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error sending serial command: %s", e.what());
            serial_connected_ = false;
        }
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    
    // Create node with configurable serial port and baud rate
    std::string serial_port = "/dev/ttyACM0";
    unsigned int baud_rate = 115200;
    
    // Parse command line arguments for serial port and baud rate
    if (argc >= 2) {
        serial_port = argv[1];
    }
    if (argc >= 3) {
        baud_rate = std::stoi(argv[2]);
    }
    
    auto node = std::make_shared<SercomNode>(serial_port, baud_rate);
    
    rclcpp::spin(node);
    
    rclcpp::shutdown();
    
    return 0;
}