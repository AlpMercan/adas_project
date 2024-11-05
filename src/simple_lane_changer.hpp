#ifndef SIMPLE_LANE_CHANGER_HPP
#define SIMPLE_LANE_CHANGER_HPP

#include <ros/ros.h>
#include <geometry_msgs/Twist.h>

class SimpleLaneChanger {
private:
    ros::Publisher cmd_vel_pub_;
    ros::Rate rate_;
    bool is_changing_lane_;

public:
    SimpleLaneChanger();
    void changeToLeftLane();
    void changeToRightLane();
    bool isBusy() const;
};

#endif 
