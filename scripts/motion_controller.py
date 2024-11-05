#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
from std_msgs.msg import Bool, Float32
from lanechange import SimpleLaneChanger  # lane_changer.py olarak kaydedin


class MotionController:
    def __init__(self):
        rospy.init_node('motion_controller_node', anonymous=True)
        self.lane_changer = SimpleLaneChanger()
        self.bridge = CvBridge()
        # Motion parameters
        self.linear_speed = 1
        self.max_angular_speed = 0.5
        # PID parameters
        self.Kp = 0.005
        self.Ki = 0.0001
        self.Kd = 0.001
        # PID variables
        self.last_error = 0
        self.integral = 0
        self.max_integral = 50
        
        # Safety parameters
        self.safety_distance_threshold = 1.0  # meters
        self.person_detected = False
        self.person_distance = float('inf')
        
        # Publishers and subscribers
        self.image_sub = rospy.Subscriber('/processed_image', Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # New publishers
        self.marker_pub = rospy.Publisher("/detected_object", Marker, queue_size=10)
        self.image_pub = rospy.Publisher("/detection_visualization", Image, queue_size=10)
        self.lane_status_pub = rospy.Publisher("/person_in_lane", Bool, queue_size=10)
        self.distance_pub = rospy.Publisher("/person_distance", Float32, queue_size=10)
        
        # New subscribers for safety
        self.person_in_lane_sub = rospy.Subscriber("/person_in_lane", Bool, self.person_in_lane_callback)
        self.person_distance_sub = rospy.Subscriber("/person_distance", Float32, self.person_distance_callback)
        
        # State variables
        self.no_detection_counter = 0
        self.max_no_detection = 5
        self.first= True
        # For debugging
        self.debug = True
        rospy.Subscriber('/ttc', Float32, self.ttc_callback)
        
    
        rospy.loginfo("Lane following motion controller initialized")
    def ttc_callback(self, msg):
        ttc_value = msg.data
        #rospy.loginfo(f'Received TTC value: {ttc_value}')

    def person_in_lane_callback(self, msg):
        """Handle person detection status"""
        self.person_detected = msg.data
        if self.person_detected:
            rospy.logwarn("Person detected in lane!")
            if self.person_distance < self.safety_distance_threshold:
                rospy.logfatal("Person too close! Emergency shutdown initiated!")
                self.emergency_shutdown()

    def person_distance_callback(self, msg):
        """Handle person distance updates"""
        self.person_distance = msg.data
        if self.person_detected and self.person_distance < self.safety_distance_threshold:
            rospy.logfatal(f"Person at unsafe distance: {self.person_distance}m! Emergency shutdown initiated!")
            self.emergency_shutdown()

    def emergency_shutdown(self):
        """Handle emergency shutdown procedure"""
        try:
            # First, stop the robot
            self.publish_zero_velocity()
            rospy.logwarn("Emergency stop initiated - person detected in unsafe zone")
            # Then shutdown for debugging
            ##rospy.signal_shutdown("Emergency stop: Person detected in unsafe zone")
        except Exception as e:
            rospy.logerr(f"Error during emergency shutdown: {str(e)}")

    def detect_lane_markers(self, image):
        """Detect blue or green lane area"""
        try:
            # Convert BGR to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Check for empty image
            if np.sum(image) == 0:
                rospy.logwarn("Empty image received - emergency stop condition")
                return None
            # HSV ranges for blue and green
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])
            # Create masks
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            combined_mask = cv2.bitwise_or(blue_mask, green_mask)
            # Reduce noise
            kernel = np.ones((5, 5), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            if self.debug:
                cv2.imshow('Path Mask', combined_mask)
                cv2.waitKey(1)
            return combined_mask
        except Exception as e:
            rospy.logerr(f"Error in detect_lane_markers: {str(e)}")
            return None

    def calculate_center_offset(self, image):
        """Calculate offset from lane center"""
        try:
            if np.sum(image) == 0:
                rospy.logwarn("Empty image - stopping vehicle")
                return None
            height, width = image.shape[:2]
            roi_height = int(height * 0.6)
            roi_top = height - roi_height
            roi = image[roi_top:height, :]
            path_mask = self.detect_lane_markers(roi)
            if path_mask is None or np.sum(path_mask) == 0:
                rospy.logwarn("No path detected")
                return None
            nonzero = cv2.findNonZero(path_mask)
            if nonzero is None or len(nonzero) < 100:
                rospy.logwarn("Not enough path points detected")
                return None
            points_mean = np.mean(nonzero, axis=0)
            center_x = points_mean[0][0]
            image_center = width / 2
            offset = -(center_x - image_center)
            if self.debug:
                debug_image = cv2.cvtColor(path_mask, cv2.COLOR_GRAY2BGR)
                cv2.circle(debug_image, (int(center_x), int(roi_height / 2)), 5, (0, 0, 255), -1)
                cv2.line(debug_image, (int(width / 2), 0), (int(width / 2), roi_height), (0, 255, 0), 1)
                cv2.imshow('Path Detection Debug', debug_image)
                cv2.waitKey(1)
                rospy.loginfo(f"Path center: {center_x:.1f}, Offset: {offset:.1f}")
            return offset
        except Exception as e:
            rospy.logerr(f"Error in calculate_center_offset: {str(e)}")
            return None

    def pid_control(self, error):
        """Apply PID control"""
        if error is None:
            self.integral = 0
            self.last_error = 0
            return 0
        proportional = self.Kp * error
        self.integral = np.clip(self.integral + error, -self.max_integral, self.max_integral)
        integral = self.Ki * self.integral
        derivative = self.Kd * (error - self.last_error)
        angular_velocity = proportional + integral + derivative
        angular_velocity = np.clip(angular_velocity, -self.max_angular_speed, self.max_angular_speed)
        self.last_error = error
        return angular_velocity

    def publish_velocity(self, linear, angular):
        """Publish velocity commands"""
        try:
            # Check safety conditions before publishing velocity
            print(self.person_in_lane_sub)
            
            if self.person_detected == True and self.first == True:
                self.publish_zero_velocity()
                print("Lane changing...")
                alp=True
                while alp:
                    self.lane_changer.change_to_left_lane()  # veya change_to_right_lane()
                    alp=False
                print("Lane changed...")
                self.first=False
                return
                
            twist = Twist()
            if abs(angular) > self.max_angular_speed / 2:
                linear *= 0.5
            twist.linear.x = linear
            twist.angular.z = angular
            self.cmd_vel_pub.publish(twist)
            rospy.loginfo(f"Velocities - Linear: {linear:.2f}, Angular: {angular:.2f}")
        except Exception as e:
            rospy.logerr(f"Error publishing velocity: {str(e)}")
            self.publish_zero_velocity()

    def publish_zero_velocity(self):
        """Publish zero velocity"""
        try:
            twist = Twist()
            self.cmd_vel_pub.publish(twist)
            rospy.loginfo("Emergency stop - Publishing zero velocity")
        except Exception as e:
            rospy.logerr(f"Error publishing zero velocity: {str(e)}")

    def image_callback(self, msg):
        """Process incoming image and control robot"""
        try:
            # Check safety conditions first
            if self.person_detected and self.person_distance < self.safety_distance_threshold:
                self.emergency_shutdown()
                print("image callback")
                return
                
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if np.sum(cv_image) == 0:
                rospy.logwarn("Emergency stop condition - empty image received")
                self.publish_zero_velocity()
                return
            offset = self.calculate_center_offset(cv_image)
            if offset is not None:
                self.no_detection_counter = 0
                angular_velocity = self.pid_control(offset)
                self.publish_velocity(self.linear_speed, angular_velocity)
            else:
                self.no_detection_counter += 1
                if self.no_detection_counter >= self.max_no_detection:
                    rospy.logwarn("Lost lane markers or emergency condition")
                    self.publish_zero_velocity()
        except Exception as e:
            rospy.logerr(f"Error in image_callback: {str(e)}")
            self.publish_zero_velocity()

def main():
    try:
        controller = MotionController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        controller.publish_zero_velocity()

if __name__ == '__main__':
    main()