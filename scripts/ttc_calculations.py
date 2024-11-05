#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Twist
import message_filters
from collections import deque
from std_msgs.msg import String

class TTCCalculator:
    def __init__(self):
        rospy.init_node('ttc_calculator', anonymous=True)
        
        # Parameters
        self.min_ttc_threshold = rospy.get_param('~min_ttc_threshold', 2.0)
        self.distance_buffer_size = rospy.get_param('~distance_buffer_size', 10)
        self.time_window = rospy.get_param('~time_window', 0.5)
        self.min_velocity_threshold = rospy.get_param('~min_velocity_threshold', 0.3)  # m/s
        
        # Data structures for motion analysis
        self.distance_buffer = deque(maxlen=self.distance_buffer_size)
        self.time_buffer = deque(maxlen=self.distance_buffer_size)
        self.velocity_buffer = deque(maxlen=self.distance_buffer_size)
        
        # Motion state tracking
        self.is_moving = False
        self.motion_direction = 0  # -1: approaching, 0: stationary, 1: moving away
        self.last_motion_update = rospy.Time.now()
        
        # Store ego vehicle velocity
        self.ego_velocity = 0.0
        
        # Publishers
        self.ttc_pub = rospy.Publisher('/ttc', Float32, queue_size=1)
        self.object_motion_pub = rospy.Publisher('/object_motion_status', String, queue_size=1)
        
        # Subscribers
        self.distance_sub = message_filters.Subscriber('/person_distance', Float32)
        self.in_lane_sub = message_filters.Subscriber('/person_in_lane', Bool)
        
        # Create a time synchronizer
        ts = message_filters.TimeSynchronizer([self.distance_sub, self.in_lane_sub], 10)
        ts.registerCallback(self.synchronized_callback)
        
        rospy.Subscriber('/cmd_vel', Twist, self.velocity_callback)
        
        rospy.loginfo("TTC Calculator with motion analysis initialized")

    def velocity_callback(self, msg):
        """Store ego vehicle velocity"""
        self.ego_velocity = msg.linear.x

    def analyze_motion(self):
        """
        Analyze object motion patterns
        Returns: (is_moving, velocity, motion_type)
        """
        if len(self.distance_buffer) < 3:
            return False, 0.0, "INSUFFICIENT_DATA"

        # Calculate velocities over recent measurements
        times = np.array([(t - self.time_buffer[0]).to_sec() 
                         for t in self.time_buffer])
        distances = np.array(self.distance_buffer)
        
        # Use linear regression to get velocity trend
        coefficients = np.polyfit(times, distances, 1)
        velocity = coefficients[0]  # Slope represents velocity
        
        # Calculate R-squared to determine if motion is linear
        y_pred = np.polyval(coefficients, times)
        r_squared = 1 - np.sum((distances - y_pred) ** 2) / np.sum((distances - np.mean(distances)) ** 2)
        
        # Relative velocity (accounting for ego vehicle)
        relative_velocity = velocity - self.ego_velocity
        
        # Determine motion state
        is_moving = abs(relative_velocity) > self.min_velocity_threshold
        
        if is_moving:
            if relative_velocity < 0:
                motion_type = "APPROACHING"
            else:
                motion_type = "MOVING_AWAY"
        else:
            motion_type = "STATIONARY"
            
        return is_moving, relative_velocity, motion_type

    def calculate_ttc(self, distance, relative_velocity, motion_type):
        """
        Calculate Time To Collision based on motion analysis
        """
        if motion_type == "STATIONARY":
            # For stationary objects, use simple distance/velocity formula
            if self.ego_velocity <= 0:
                return float('inf')
            return distance / self.ego_velocity
            
        elif motion_type == "APPROACHING":
            # For approaching objects, use relative velocity
            if relative_velocity >= 0:  # Sanity check
                return float('inf')
            return abs(distance / relative_velocity)
            
        elif motion_type == "MOVING_AWAY":
            return float('inf')
            
        return float('inf')  # Default case

    def synchronized_callback(self, distance_msg, in_lane_msg):
        """Process synchronized distance and lane status messages"""
        current_time = rospy.Time.now()
        current_distance = distance_msg.data
        is_in_lane = in_lane_msg.data
        
        # Only process if person is in lane
        if not is_in_lane:
            self.ttc_pub.publish(Float32(float('inf')))
            self.object_motion_pub.publish(String("OUT_OF_LANE"))
            return
        
        # Update buffers
        self.distance_buffer.append(current_distance)
        self.time_buffer.append(current_time)
        
        # Analyze motion
        is_moving, relative_velocity, motion_type = self.analyze_motion()
        
        # Calculate TTC based on motion analysis
        ttc = self.calculate_ttc(current_distance, relative_velocity, motion_type)
        
        # Publish results
        self.ttc_pub.publish(Float32(ttc))
        self.object_motion_pub.publish(String(motion_type))
        
        # Log detailed information
        if ttc < self.min_ttc_threshold and ttc != float('inf'):
            rospy.logwarn(
                f"Low TTC detected: {ttc:.2f}s\n"
                f"Distance: {current_distance:.2f}m\n"
                f"Relative Velocity: {relative_velocity:.2f}m/s\n"
                f"Motion Type: {motion_type}"
            )

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        ttc_calculator = TTCCalculator()
        ttc_calculator.run()
    except rospy.ROSInterruptException:
        pass
