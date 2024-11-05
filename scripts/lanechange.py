#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
import time

class SimpleLaneChanger:
    def __init__(self):
        # Node'u main scriptte başlatacağımız için init_node'u kaldırıyoruz
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.rate = rospy.Rate(10)  # 10Hz
        self.is_changing_lane = False  # Şerit değiştirme durumunu takip et

    def change_to_left_lane(self):
        if self.is_changing_lane:
            return
        
        self.is_changing_lane = True
        cmd = Twist()
        
        # First move forward a bit
        print("Moving forward...")
        cmd.linear.x = 0.5  # Forward speed
        cmd.angular.z = 0.0
        for _ in range(20):  # 2 seconds
            self.cmd_vel_pub.publish(cmd)
            self.rate.sleep()

        # Execute left turn
        print("Changing to left lane...")
        cmd.linear.x = 0.3  # Slower during turn
        cmd.angular.z = 0.5  # Positive for left turn
        for _ in range(30):  # 3 seconds
            self.cmd_vel_pub.publish(cmd)
            self.rate.sleep()

        # Straighten out
        print("Straightening...")
        cmd.linear.x = 0.3
        cmd.angular.z = -0.5  # Negative to straighten
        for _ in range(30):  # 1.5 seconds
            self.cmd_vel_pub.publish(cmd)
            self.rate.sleep()

        # Stop
        print("Stopping...")
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
        
        self.is_changing_lane = False
        print("Left lane change completed!")
        
    def change_to_right_lane(self):
        if self.is_changing_lane:
            return
            
        self.is_changing_lane = True
        cmd = Twist()
        
        # First move forward a bit
        print("Moving forward...")
        cmd.linear.x = 0.5  # Forward speed
        cmd.angular.z = 0.0
        for _ in range(20):  # 2 seconds
            self.cmd_vel_pub.publish(cmd)
            self.rate.sleep()

        # Execute right turn
        print("Changing to right lane...")
        cmd.linear.x = 0.3  # Slower during turn
        cmd.angular.z = -0.5  # Negative for right turn
        for _ in range(30):  # 3 seconds
            self.cmd_vel_pub.publish(cmd)
            self.rate.sleep()

        # Straighten out
        print("Straightening...")
        cmd.linear.x = 0.3
        cmd.angular.z = 0.5  # Positive to straighten
        for _ in range(30):  # 1.5 seconds
            self.cmd_vel_pub.publish(cmd)
            self.rate.sleep()

        # Stop
        print("Stopping...")
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
        
        self.is_changing_lane = False
        print("Right lane change completed!")

    def is_busy(self):
        return self.is_changing_lane
        
    ## bu maine ekle    


