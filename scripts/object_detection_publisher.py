#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Bool, Float32
import torch

class ObjectDetector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Load YOLO model
        self.model = YOLO('yolov8n.pt')
        
        # Initialize depth image
        self.current_depth_image = None
        self.depth_time = None
        
        # Create subscribers
        self.image_sub = rospy.Subscriber("/processed_image", Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.depth_callback)
        
        # Publishers
        self.marker_pub = rospy.Publisher("/detected_object", Marker, queue_size=10)
        self.image_pub = rospy.Publisher("/detection_visualization", Image, queue_size=10)
        self.lane_status_pub = rospy.Publisher("/person_in_lane", Bool, queue_size=10)
        self.distance_pub = rospy.Publisher("/person_distance", Float32, queue_size=10)
        
        # Blue lane detection parameters
        self.blue_lower = np.array([100, 50, 50])
        self.blue_upper = np.array([130, 255, 255])
        
        # Detection parameters
        self.max_depth_age = rospy.Duration(0.5)  # Maximum age for depth data
        
        rospy.loginfo("Object detector initialized")

    def depth_callback(self, depth_msg):
        """
        Callback for depth image processing
        """
        try:
            self.current_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            self.depth_time = rospy.Time.now()
        except CvBridgeError as e:
            rospy.logerr(f"Depth image conversion error: {e}")

    def detect_blue_lane(self, image):
        """
        Detect the blue lane in the image
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        kernel = np.ones((5,5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        return blue_mask

    def calculate_distance(self, depth_image, box):
        """
        Calculate the average distance to the detected person using depth data
        Returns distance in meters
        """
        if depth_image is None:
            return None

        x1, y1, x2, y2 = map(int, box)
        
        # Get center region of the bounding box (20% of the box size)
        box_width = x2 - x1
        box_height = y2 - y1
        center_x1 = int(x1 + box_width * 0.4)
        center_x2 = int(x1 + box_width * 0.6)
        center_y1 = int(y1 + box_height * 0.4)
        center_y2 = int(y1 + box_height * 0.6)
        
        # Get the depth region of interest (ROI)
        depth_roi = depth_image[center_y1:center_y2, center_x1:center_x2]
        
        # Filter out invalid/infinite depth values and convert to meters
        valid_depths = depth_roi[np.isfinite(depth_roi)]
        valid_depths = valid_depths[valid_depths > 0] / 1000.0  # Convert mm to meters
        
        if len(valid_depths) > 0:
            # Use median to be more robust against outliers
            median_depth = np.median(valid_depths)
            return median_depth
        return None

    def check_box_lane_intersection(self, box_coords, blue_mask):
        """
        Check if the bounding box intersects with the blue lane
        """
        x1, y1, x2, y2 = map(int, box_coords)
        box_mask = np.zeros_like(blue_mask)
        cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, -1)
        intersection = cv2.bitwise_and(blue_mask, box_mask)
        return np.any(intersection > 0)

    def draw_detections(self, image, results, is_in_lane, distance=None):
        """
        Draw bounding boxes, labels, and distance information
        """
        annotated_image = image.copy()
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf)
                cls = int(box.cls)
                
                color = (0, 255, 0) if is_in_lane else (0, 0, 255)
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                status_text = "IN LANE" if is_in_lane else "OUT OF LANE"
                distance_text = f"Dist: {distance:.2f}m" if distance is not None else "Dist: No depth"
                label = f"{results[0].names[cls]}: {conf:.2f} - {status_text} - {distance_text}"
                
                cv2.putText(annotated_image, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated_image

    def rgb_callback(self, rgb_msg):
        """
        Callback for RGB image processing
        """
        try:
            # Check if we have recent depth data
            if (self.current_depth_image is None or 
                self.depth_time is None or 
                (rospy.Time.now() - self.depth_time) > self.max_depth_age):
                rospy.logwarn_throttle(1, "No recent depth data available")
            
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            
            # Detect blue lane
            blue_mask = self.detect_blue_lane(cv_image)
            
            # Run object detection
            results = self.model(cv_image)
            
            person_in_lane = False
            person_distance = None
            
            # Process results
            if results and len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    box_coords = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf)
                    cls = int(box.cls)
                    
                    if cls == 0:  # If person
                        person_in_lane = self.check_box_lane_intersection(box_coords, blue_mask)
                        
                        if self.current_depth_image is not None:
                            person_distance = self.calculate_distance(self.current_depth_image, box_coords)*100
                        
                        center_x = (box_coords[0] + box_coords[2]) / 2
                        center_y = (box_coords[1] + box_coords[3]) / 2
                        
                        # Publish marker
                        self.publish_marker(center_x, center_y, 
                                         person_distance if person_distance else 0,
                                         conf, person_in_lane)
                        
                        # Log detection
                        status = "IN" if person_in_lane else "OUT OF"
                        distance_text = f"at {person_distance:.2f}m" if person_distance else "distance unknown"
                        rospy.loginfo(f"Person detected {status} lane {distance_text}")
            
            # Publish results
            self.lane_status_pub.publish(Bool(person_in_lane))
            if person_distance is not None:
                self.distance_pub.publish(Float32(person_distance))
            
            # Draw and publish visualization
            annotated_image = self.draw_detections(cv_image, results, person_in_lane, person_distance)
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(annotated_image, "bgr8"))
                
        except CvBridgeError as e:
            rospy.logerr(f"RGB image conversion error: {e}")

    def publish_marker(self, x, y, z, confidence, in_lane):
        """
        Publish a visualization marker for the detected object
        """
        marker = Marker()
        marker.header.frame_id = "camera_frame"
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        
        marker.color.r = 0.0 if in_lane else 1.0
        marker.color.g = 1.0 if in_lane else 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        self.marker_pub.publish(marker)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = ObjectDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass