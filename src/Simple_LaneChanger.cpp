#include "simple_lane_changer.hpp"
#include <ros/console.h>

SimpleLaneChanger::SimpleLaneChanger() 
    : rate_(10), // 10Hz
      is_changing_lane_(false) {
    ros::NodeHandle nh;
    cmd_vel_pub_ = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
}

void SimpleLaneChanger::changeToLeftLane() {
    if (is_changing_lane_) {
        return;
    }
    is_changing_lane_ = true;
    geometry_msgs::Twist cmd;

    // First move forward a bit
    ROS_INFO("Moving forward...");
    cmd.linear.x = 0.5;  // Forward speed
    cmd.angular.z = 0.0;
    for (int i = 0; i < 20; ++i) {  // 2 seconds
        cmd_vel_pub_.publish(cmd);
        rate_.sleep();
    }

    // Execute left turn
    ROS_INFO("Changing to left lane...");
    cmd.linear.x = 0.3;   // Slower during turn
    cmd.angular.z = 0.5;  // Positive for left turn
    for (int i = 0; i < 30; ++i) {  // 3 seconds
        cmd_vel_pub_.publish(cmd);
        rate_.sleep();
    }

    // Straighten out
    ROS_INFO("Straightening...");
    cmd.linear.x = 0.3;
    cmd.angular.z = -0.5;  // Negative to straighten
    for (int i = 0; i < 30; ++i) {  // 1.5 seconds
        cmd_vel_pub_.publish(cmd);
        rate_.sleep();
    }

    // Stop
    ROS_INFO("Stopping...");
    cmd.linear.x = 0.0;
    cmd.angular.z = 0.0;
    cmd_vel_pub_.publish(cmd);

    is_changing_lane_ = false;
    ROS_INFO("Left lane change completed!");
}

void SimpleLaneChanger::changeToRightLane() {
    if (is_changing_lane_) {
        return;
    }
    is_changing_lane_ = true;
    geometry_msgs::Twist cmd;

    // First move forward a bit
    ROS_INFO("Moving forward...");
    cmd.linear.x = 0.5;  // Forward speed
    cmd.angular.z = 0.0;
    for (int i = 0; i < 20; ++i) {  // 2 seconds
        cmd_vel_pub_.publish(cmd);
        rate_.sleep();
    }

    // Execute right turn
    ROS_INFO("Changing to right lane...");
    cmd.linear.x = 0.3;    // Slower during turn
    cmd.angular.z = -0.5;  // Negative for right turn
    for (int i = 0; i < 30; ++i) {  // 3 seconds
        cmd_vel_pub_.publish(cmd);
        rate_.sleep();
    }

    // Straighten out
    ROS_INFO("Straightening...");
    cmd.linear.x = 0.3;
    cmd.angular.z = 0.5;  // Positive to straighten
    for (int i = 0; i < 30; ++i) {  // 1.5 seconds
        cmd_vel_pub_.publish(cmd);
        rate_.sleep();
    }

    // Stop
    ROS_INFO("Stopping...");
    cmd.linear.x = 0.0;
    cmd.angular.z = 0.0;
    cmd_vel_pub_.publish(cmd);

    is_changing_lane_ = false;
    ROS_INFO("Right lane change completed!");
}

bool SimpleLaneChanger::isBusy() const {
    return is_changing_lane_;
}
