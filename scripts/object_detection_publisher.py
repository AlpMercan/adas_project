#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import torch
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray, Header

class YOLODepthDetector:
    def __init__(self):
        rospy.init_node('yolo_depth_detector', anonymous=True)
        self.bridge = CvBridge()
        
        # Initialize YOLO model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
        self.model.conf = 0.5
        
        # Target classes for detection
        self.target_classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.image_width = 640
        self.image_height = 480
        fov_horizontal = 1.047198  # radians
        self.focal_length = self.image_width / (2 * np.tan(fov_horizontal / 2))

        
        # Camera parameters
        self.depth_scale = rospy.get_param('~depth_scale', 0.001)  # Default for RealSense (1mm = 0.001m)
        self.min_depth = rospy.get_param('~min_depth', 0.5)  # Minimum valid depth in meters
        self.max_depth = rospy.get_param('~max_depth', 10.0)  # Maximum valid depth in meters
        
        # Initialize subscribers with time synchronization
        rgb_sub = Subscriber('/camera/rgb/image_raw', Image)
        depth_sub = Subscriber('/camera/depth/image_raw', Image)
        
        # Synchronize RGB and depth images
        self.ts = ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=5,
            slop=0.1
        )
        self.ts.registerCallback(self.image_callback)
        
        # Publishers
        self.detection_pub = rospy.Publisher('/yolo/detections', Float32MultiArray, queue_size=1)
        self.marker_pub = rospy.Publisher('/yolo/markers', MarkerArray, queue_size=1)
        self.debug_pub = rospy.Publisher('/yolo/debug_image', Image, queue_size=1)
        
        rospy.loginfo("YOLO detector initialized on device: %s", self.device)

    def get_depth_from_roi(self, depth_image, bbox, depth_scale=0.001):
        """Get reliable depth measurement from ROI for OpenNI camera"""
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        # Get center region (20% of bounding box)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        roi_size = min((x2 - x1), (y2 - y1)) // 5
        
        # Define ROI boundaries
        roi_x1 = max(0, center_x - roi_size)
        roi_x2 = min(self.image_width, center_x + roi_size)
        roi_y1 = max(0, center_y - roi_size)
        roi_y2 = min(self.image_height, center_y + roi_size)
        
        # Extract depth ROI
        depth_roi = depth_image[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # OpenNI depth format handling
        if depth_roi.dtype == np.uint16:
            # Convert from millimeters to meters
            depth_roi = depth_roi.astype(float) * depth_scale
        
        # Filter out invalid measurements and apply cutoffs
        valid_depths = depth_roi[
            (depth_roi >= self.min_depth) & 
            (depth_roi <= self.max_depth) &
            (depth_roi > 0)  # Filter out zero values
        ]
        
        if len(valid_depths) > 0:
            # Use median for robustness
            depth = np.median(valid_depths)
            # Calculate confidence based on number of valid points
            confidence = len(valid_depths) / (roi_size * roi_size * 4)
            if confidence > 0.5:  # At least 50% valid points
                rospy.logdebug(f"Depth measurement: {depth:.2f}m (confidence: {confidence:.2f})")
                return depth
            else:
                rospy.logdebug("Low confidence depth measurement")
                return None
        else:
            rospy.logdebug("No valid depth measurements in ROI")
            return None

    def process_detections(self, results, depth_image):
        """Process YOLO detections and calculate distances"""
        detections_msg = Float32MultiArray()
        marker_array = MarkerArray()
        detections_data = []
        
        try:
            # Determine depth image format and scaling
            if depth_image.dtype == np.uint16:
                rospy.logdebug("Depth image format: UINT16")
            else:
                rospy.logdebug(f"Depth image format: {depth_image.dtype}")
            
            # Process each detection
            for det in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                class_name = self.model.names[int(cls)]
                
                if class_name not in self.target_classes:
                    continue
                
                # Get depth measurement
                depth = self.get_depth_from_roi(depth_image, (x1, y1, x2, y2), self.depth_scale)
                
                if depth is not None:
                    # Store detection data
                    detection_data = [
                        float(cls),
                        x1, y1, x2, y2,
                        float(conf),
                        depth
                    ]
                    detections_data.extend(detection_data)
                    
                    # Create marker
                    marker = self.create_marker(
                        class_name,
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                        depth,
                        len(marker_array.markers)
                    )
                    marker_array.markers.append(marker)
                    
                    rospy.loginfo(f"Detected {class_name} at {depth:.2f}m")
        
        except Exception as e:
            rospy.logerr(f"Error processing detections: {str(e)}")
            return None, None
        
        detections_msg.data = detections_data
        return detections_msg, marker_array

    def create_marker(self, class_name, center_x, center_y, distance, marker_id):
        """Create visualization marker with correct camera parameters"""
        marker = Marker()
        marker.header = Header()
        marker.header.frame_id = "camera_link"  # Matches URDF frame
        marker.header.stamp = rospy.Time.now()
        
        marker.ns = "yolo_detections"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        # Calculate 3D position using camera parameters
        x = distance  # Forward direction
        y = (center_x - self.image_width/2) * distance / self.focal_length  # Left-right
        z = (center_y - self.image_height/2) * distance / self.focal_length  # Up-down
        
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        
        # Set appropriate sizes based on class
        if class_name == 'person':
            size = [0.5, 0.5, 1.7]  # Average human dimensions
        elif class_name == 'car':
            size = [4.5, 1.8, 1.5]  # Average car dimensions
        elif class_name == 'truck':
            size = [6.0, 2.0, 2.5]  # Average truck dimensions
        elif class_name == 'bus':
            size = [12.0, 2.5, 3.0]  # Average bus dimensions
        else:
            size = [1.0, 1.0, 1.0]  # Default size
        
        marker.scale.x = size[0]
        marker.scale.y = size[1]
        marker.scale.z = size[2]
        
        # Set orientation (aligned with camera frame)
        marker.pose.orientation.w = 1.0
        
        # Color based on distance (green to red gradient)
        marker.color.a = 0.7
        marker.color.r = min(1.0, distance / self.max_depth)
        marker.color.g = min(1.0, 1.0 - (distance / self.max_depth))
        marker.color.b = 0.2
        
        return marker

    def visualize_detections(self, image, results, depth_image):
        """Create debug visualization with improved distance display"""
        vis_image = image.copy()
        
        try:
            for det in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                class_name = self.model.names[int(cls)]
                
                if class_name not in self.target_classes:
                    continue
                
                # Get depth for visualization
                depth = self.get_depth_from_roi(depth_image, (x1, y1, x2, y2), self.depth_scale)
                
                # Choose color based on depth
                if depth is not None:
                    # Color gradient: red (close) to green (far)
                    color = (
                        int(255 * (1 - depth / 5.0)),  # B
                        int(255 * (depth / 5.0)),      # G
                        0                              # R
                    )
                else:
                    color = (0, 0, 255)  # Red for invalid depth
                
                # Draw bounding box
                cv2.rectangle(vis_image, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            color, 2)
                
                # Add label with class, confidence and distance
                label = f"{class_name} {conf:.2f}"
                if depth is not None:
                    label += f" {depth:.2f}m"
                
                # Draw label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(vis_image,
                            (int(x1), int(y1) - label_size[1] - 10),
                            (int(x1 + label_size[0]), int(y1)),
                            color, -1)
                
                # Draw label text
                cv2.putText(vis_image, label,
                           (int(x1), int(y1 - 5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (255, 255, 255), 2)
            
            return vis_image
            
        except Exception as e:
            rospy.logerr(f"Error in visualization: {str(e)}")
            return image

    def image_callback(self, rgb_msg, depth_msg):
        """Process synchronized RGB and depth images"""
        try:
            # Convert ROS messages to OpenCV images
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg)  # Let CvBridge auto-detect format
            
            rospy.logdebug(f"Depth image type: {depth_image.dtype}, shape: {depth_image.shape}")
            
            # Run YOLO detection
            results = self.model(rgb_image)
            
            # Process detections and get messages
            detections_msg, marker_array = self.process_detections(results, depth_image)
            
            if detections_msg is not None:
                # Publish detection results
                self.detection_pub.publish(detections_msg)
                self.marker_pub.publish(marker_array)
                
                # Create and publish debug visualization
                debug_image = self.visualize_detections(rgb_image, results, depth_image)
                debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
                self.debug_pub.publish(debug_msg)
                
        except Exception as e:
            rospy.logerr(f"Error in image callback: {str(e)}")

def main():
    try:
        detector = YOLODepthDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()