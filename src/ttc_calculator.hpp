#ifndef TTC_CALCULATOR_HPP
#define TTC_CALCULATOR_HPP

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <geometry_msgs/Twist.h>
#include <deque>
#include <Eigen/Dense>

class TTCCalculator {
private:
    ros::NodeHandle nh_;
    
    // Parameters
    double min_ttc_threshold_;
    int distance_buffer_size_;
    double time_window_;
    double min_velocity_threshold_;
    
    // Publishers
    ros::Publisher ttc_pub_;
    ros::Publisher object_motion_pub_;
    
    // Subscribers
    message_filters::Subscriber<std_msgs::Float32> distance_sub_;
    message_filters::Subscriber<std_msgs::Bool> in_lane_sub_;
    ros::Subscriber velocity_sub_;
    std::shared_ptr<message_filters::TimeSynchronizer<std_msgs::Float32, std_msgs::Bool>> sync_;
    
    // Data buffers
    std::deque<double> distance_buffer_;
    std::deque<ros::Time> time_buffer_;
    std::deque<double> velocity_buffer_;
    
    // Motion state
    bool is_moving_;
    int motion_direction_;  // -1: approaching, 0: stationary, 1: moving away
    ros::Time last_motion_update_;
    double ego_velocity_;

    // Callback functions
    void velocityCallback(const geometry_msgs::Twist::ConstPtr& msg);
    void synchronizedCallback(const std_msgs::Float32::ConstPtr& distance_msg,
                            const std_msgs::Bool::ConstPtr& in_lane_msg);
    
    // Helper functions
    std::tuple<bool, double, std::string> analyzeMotion();
    double calculateTTC(double distance, double relative_velocity, const std::string& motion_type);
    
    // Parameter loading
    void loadParameters();

public:
    TTCCalculator();
    void run();
};

#endif // TTC_CALCULATOR_HPP
