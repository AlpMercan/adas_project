#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <geometry_msgs/Twist.h>
#include <deque>
#include <limits>
#include <cmath>

class TTCCalculator {
private:
    ros::NodeHandle nh_;
    
    // Publishers
    ros::Publisher ttc_pub_;
    ros::Publisher object_motion_pub_;
    
    // Subscribers
    ros::Subscriber distance_sub_;
    ros::Subscriber in_lane_sub_;
    ros::Subscriber velocity_sub_;
    
    // Parameters
    double min_ttc_threshold_;
    int distance_buffer_size_;
    double time_window_;
    double min_velocity_threshold_;
    
    // Buffers
    std::deque<double> distance_buffer_;
    std::deque<ros::Time> time_buffer_;
    
    // State variables
    bool is_in_lane_;
    double ego_velocity_;
    double current_distance_;

public:
    TTCCalculator() : 
        min_ttc_threshold_(2.0),
        distance_buffer_size_(10),
        time_window_(0.5),
        min_velocity_threshold_(0.3),
        is_in_lane_(false),
        ego_velocity_(0.0),
        current_distance_(std::numeric_limits<double>::infinity())
    {
        // Initialize publishers
        ttc_pub_ = nh_.advertise<std_msgs::Float32>("/ttc", 1);
        object_motion_pub_ = nh_.advertise<std_msgs::String>("/object_motion_status", 1);
        
        // Initialize subscribers
        distance_sub_ = nh_.subscribe("/person_distance", 1, 
            &TTCCalculator::distanceCallback, this);
        in_lane_sub_ = nh_.subscribe("/person_in_lane", 1, 
            &TTCCalculator::inLaneCallback, this);
        velocity_sub_ = nh_.subscribe("/cmd_vel", 1, 
            &TTCCalculator::velocityCallback, this);
        
        ROS_INFO("TTC Calculator initialized");
    }

    void velocityCallback(const geometry_msgs::Twist::ConstPtr& msg) {
        ego_velocity_ = msg->linear.x;
    }

    void inLaneCallback(const std_msgs::Bool::ConstPtr& msg) {
        is_in_lane_ = msg->data;
        if (!is_in_lane_) {
            publishInfinityTTC("OUT_OF_LANE");
        } else {
            analyzeMotionAndPublish();
        }
    }

    void distanceCallback(const std_msgs::Float32::ConstPtr& msg) {
        current_distance_ = msg->data;
        
        // Update buffers
        distance_buffer_.push_back(current_distance_);
        time_buffer_.push_back(ros::Time::now());
        
        if (distance_buffer_.size() > distance_buffer_size_) {
            distance_buffer_.pop_front();
            time_buffer_.pop_front();
        }
        
        if (is_in_lane_) {
            analyzeMotionAndPublish();
        }
    }

    void publishInfinityTTC(const std::string& motion_type) {
        std_msgs::Float32 ttc_msg;
        ttc_msg.data = std::numeric_limits<double>::infinity();
        ttc_pub_.publish(ttc_msg);
        
        std_msgs::String motion_msg;
        motion_msg.data = motion_type;
        object_motion_pub_.publish(motion_msg);
    }

    void analyzeMotionAndPublish() {
        if (distance_buffer_.size() < 3) {
            publishInfinityTTC("INSUFFICIENT_DATA");
            return;
        }

        // Calculate velocity using simple differentiation
        double dt = (time_buffer_.back() - time_buffer_.front()).toSec();
        if (dt <= 0) {
            publishInfinityTTC("INVALID_TIME");
            return;
        }

        double velocity = (distance_buffer_.back() - distance_buffer_.front()) / dt;
        double relative_velocity = velocity - ego_velocity_;
        
        // Calculate TTC
        double ttc;
        std::string motion_type;

        if (std::abs(relative_velocity) < min_velocity_threshold_) {
            ttc = std::numeric_limits<double>::infinity();
            motion_type = "STATIONARY";
        } else if (relative_velocity < 0) {  // Object is approaching
            ttc = -current_distance_ / relative_velocity;
            motion_type = "APPROACHING";
        } else {
            ttc = std::numeric_limits<double>::infinity();
            motion_type = "MOVING_AWAY";
        }
        
        // Publish results
        std_msgs::Float32 ttc_msg;
        ttc_msg.data = ttc;
        ttc_pub_.publish(ttc_msg);
        
        std_msgs::String motion_msg;
        motion_msg.data = motion_type;
        object_motion_pub_.publish(motion_msg);
        
        // Log warning if TTC is low
        if (ttc < min_ttc_threshold_ && ttc != std::numeric_limits<double>::infinity()) {
            ROS_WARN_STREAM("Low TTC detected: " << ttc << "s, Distance: " << current_distance_ 
                          << "m, Relative Velocity: " << relative_velocity << "m/s");
        }
    }

    void run() {
        ros::spin();
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "ttc_calculator_node");
    TTCCalculator calculator;
    calculator.run();
    return 0;
}
