#include<opencv2/opencv.hpp>
#include<iostream>

int hue_value = 10;
const int hue_slider_max = 180;
int thresh_value = 15;
const int thresh_slider_max = 180;

int main(){
    // Initialize video
    cv::VideoCapture cap("second.mp4");
    if(!cap.isOpened()){
        return -1;
    }
    
    cv::Mat frame;

    // Initialize slider
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Hue Slider", "Original", &hue_value, hue_slider_max);
    cv::createTrackbar("Threshold Slider", "Original", &thresh_value, thresh_slider_max);

    
    // Main loop
    while (true) {
        // Display Video
        cap.read(frame);
        if (frame.empty()) {
            break;
        }
        
        // Apply mask
        cv::Mat frameHSV;
        cv::cvtColor(frame, frameHSV, cv::COLOR_BGR2HSV);
        
        cv::Scalar minHSV(hue_value - thresh_value, 100, 100);
        cv::Scalar maxHSV(hue_value + thresh_value, 255, 255);
        
        cv::Mat maskHSV, resultHSV;
        cv::inRange(frameHSV, minHSV, maxHSV, maskHSV);
        
        cv::imshow("Original", frame);
        cv::imshow("Mask", maskHSV);

        // Terminate
        if (cv::waitKey(30) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
}