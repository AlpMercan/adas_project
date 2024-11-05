#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <visualization_msgs/Marker.h>
#include "simple_lane_changer.hpp"

class MotionController {
private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
    
    ros::Publisher cmd_vel_pub_;
    ros::Publisher marker_pub_;
    ros::Publisher lane_status_pub_;
    ros::Publisher distance_pub_;
    
    ros::Subscriber person_in_lane_sub_;
    ros::Subscriber person_distance_sub_;
    ros::Subscriber ttc_sub_;
    
    SimpleLaneChanger lane_changer_;

    // Motion parameters
    double linear_speed_;
    double max_angular_speed_;
    
    // PID parameters
    double Kp_;
    double Ki_;
    double Kd_;
    
    // PID variables
    double last_error_;
    double integral_;
    double max_integral_;
    
    // Safety parameters
    double safety_distance_threshold_;
    bool person_detected_;
    double person_distance_;
    
    // State variables
    int no_detection_counter_;
    int max_no_detection_;
    bool first_;
    bool debug_;

public:
    MotionController() 
        : it_(nh_)
        , linear_speed_(1.0)
        , max_angular_speed_(0.5)
        , Kp_(0.005)
        , Ki_(0.0001)
        , Kd_(0.001)
        , last_error_(0.0)
        , integral_(0.0)
        , max_integral_(50.0)
        , safety_distance_threshold_(1.0)
        , person_detected_(false)
        , person_distance_(std::numeric_limits<double>::infinity())
        , no_detection_counter_(0)
        , max_no_detection_(5)
        , first_(true)
        , debug_(true)
    {
        // Initialize subscribers
        image_sub_ = it_.subscribe("/processed_image", 1, 
            &MotionController::imageCallback, this);
        person_in_lane_sub_ = nh_.subscribe("/person_in_lane", 1,
            &MotionController::personInLaneCallback, this);
        person_distance_sub_ = nh_.subscribe("/person_distance", 1,
            &MotionController::personDistanceCallback, this);
        ttc_sub_ = nh_.subscribe("/ttc", 1,
            &MotionController::ttcCallback, this);

        // Initialize publishers
        cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
        marker_pub_ = nh_.advertise<visualization_msgs::Marker>("/detected_object", 10);
        lane_status_pub_ = nh_.advertise<std_msgs::Bool>("/person_in_lane", 10);
        distance_pub_ = nh_.advertise<std_msgs::Float32>("/person_distance", 10);
        image_pub_ = it_.advertise("/detection_visualization", 10);

        ROS_INFO("Lane following motion controller initialized");
    }

    void ttcCallback(const std_msgs::Float32::ConstPtr& msg) {
        double ttc_value = msg->data;
    }

    void personInLaneCallback(const std_msgs::Bool::ConstPtr& msg) {
        person_detected_ = msg->data;
        if (person_detected_) {
            ROS_WARN("Person detected in lane!");
            if (person_distance_ < safety_distance_threshold_) {
                ROS_FATAL("Person too close! Emergency shutdown initiated!");
                emergencyShutdown();
            }
        }
    }

    void personDistanceCallback(const std_msgs::Float32::ConstPtr& msg) {
        person_distance_ = msg->data;
        if (person_detected_ && person_distance_ < safety_distance_threshold_) {
            ROS_FATAL_STREAM("Person at unsafe distance: " << person_distance_ << "m!");
            emergencyShutdown();
        }
    }

    void emergencyShutdown() {
        try {
            publishZeroVelocity();
            ROS_WARN("Emergency stop initiated - person detected in unsafe zone");
        } catch (const std::exception& e) {
            ROS_ERROR_STREAM("Error during emergency shutdown: " << e.what());
        }
    }

    cv::Mat detectLaneMarkers(const cv::Mat& image) {
        if (image.empty()) {
            return cv::Mat();
        }

        cv::Mat hsv;
        cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

        cv::Mat blue_mask, green_mask, combined_mask;
        cv::inRange(hsv, cv::Scalar(100, 50, 50), cv::Scalar(130, 255, 255), blue_mask);
        cv::inRange(hsv, cv::Scalar(40, 50, 50), cv::Scalar(80, 255, 255), green_mask);
        cv::bitwise_or(blue_mask, green_mask, combined_mask);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(combined_mask, combined_mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(combined_mask, combined_mask, cv::MORPH_CLOSE, kernel);

        return combined_mask;
    }

    double calculateCenterOffset(const cv::Mat& image) {
        if (image.empty()) {
            return 0.0;
        }

        int height = image.rows;
        int width = image.cols;
        int roi_height = static_cast<int>(height * 0.6);
        int roi_top = height - roi_height;

        cv::Mat roi = image(cv::Range(roi_top, height), cv::Range::all());
        cv::Mat path_mask = detectLaneMarkers(roi);

        if (path_mask.empty() || cv::countNonZero(path_mask) == 0) {
            return 0.0;
        }

        std::vector<cv::Point> nonzero;
        cv::findNonZero(path_mask, nonzero);

        if (nonzero.size() < 100) {
            return 0.0;
        }

        cv::Scalar points_mean = cv::mean(nonzero);
        double center_x = points_mean[0];
        double offset = -(center_x - width / 2.0);

        return offset;
    }

    double pidControl(double error) {
        if (std::isnan(error)) {
            integral_ = 0.0;
            last_error_ = 0.0;
            return 0.0;
        }

        double proportional = Kp_ * error;
        integral_ = std::max(-max_integral_, std::min(integral_ + error, max_integral_));
        double integral = Ki_ * integral_;
        double derivative = Kd_ * (error - last_error_);
        
        double angular_velocity = proportional + integral + derivative;
        angular_velocity = std::max(-max_angular_speed_, 
                                  std::min(angular_velocity, max_angular_speed_));
        last_error_ = error;
        
        return angular_velocity;
    }

    void publishVelocity(double linear, double angular) {
        try {
            if (person_detected_ && first_) {
                publishZeroVelocity();
                ROS_INFO("Lane changing...");
                lane_changer_.changeToLeftLane();
                ROS_INFO("Lane changed...");
                first_ = false;
                return;
            }

            geometry_msgs::Twist twist;
            if (std::abs(angular) > max_angular_speed_ / 2) {
                linear *= 0.5;
            }
            twist.linear.x = linear;
            twist.angular.z = angular;
            cmd_vel_pub_.publish(twist);
        } catch (const std::exception& e) {
            ROS_ERROR_STREAM("Error publishing velocity: " << e.what());
            publishZeroVelocity();
        }
    }

    void publishZeroVelocity() {
        try {
            geometry_msgs::Twist twist;
            cmd_vel_pub_.publish(twist);
            ROS_INFO("Emergency stop - Publishing zero velocity");
        } catch (const std::exception& e) {
            ROS_ERROR_STREAM("Error publishing zero velocity: " << e.what());
        }
    }

    void imageCallback(const sensor_msgs::Image::ConstPtr& msg) {
        try {
            if (person_detected_ && person_distance_ < safety_distance_threshold_) {
                emergencyShutdown();
                return;
            }

            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            if (cv_ptr->image.empty()) {
                publishZeroVelocity();
                return;
            }

            double offset = calculateCenterOffset(cv_ptr->image);
            if (!std::isnan(offset)) {
                no_detection_counter_ = 0;
                double angular_velocity = pidControl(offset);
                publishVelocity(linear_speed_, angular_velocity);
            } else {
                no_detection_counter_++;
                if (no_detection_counter_ >= max_no_detection_) {
                    ROS_WARN("Lost lane markers or emergency condition");
                    publishZeroVelocity();
                }
            }
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("CV bridge exception: %s", e.what());
            publishZeroVelocity();
        } catch (const std::exception& e) {
            ROS_ERROR_STREAM("Error in imageCallback: " << e.what());
            publishZeroVelocity();
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "motion_controller_node");
    MotionController controller;
    ros::spin();
    return 0;
}
