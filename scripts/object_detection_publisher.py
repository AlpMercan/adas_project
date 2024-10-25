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
from adas_project.msg import ObjectLocation  
import image_geometry

class YOLODepthDetector:
    def __init__(self):
        rospy.init_node('yolo_depth_detector', anonymous=True)
        self.bridge = CvBridge()
        
        # Debug flags
        self.debug = True
        
        rospy.loginfo("Initializing YOLO detector...")
        
        # Initialize camera model
        self.camera_model = image_geometry.PinholeCameraModel()
        self.camera_info_received = False
        
        try:
            # Initialize YOLO model with debug info
            rospy.loginfo("Loading YOLO model...")
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
            self.model.conf = 0.25  # Düşük confidence threshold
            rospy.loginfo("YOLO model loaded successfully")
            
            # Print available classes
            rospy.loginfo(f"Available YOLO classes: {self.model.names}")
        except Exception as e:
            rospy.logerr(f"Error loading YOLO model: {str(e)}")
            raise
        
        # Target classes for detection
        self.target_classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
        rospy.loginfo(f"Target classes: {self.target_classes}")
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        rospy.loginfo(f"Using device: {self.device}")
        
        # Image parameters
        self.image_width = rospy.get_param('~image_width', 640)
        self.image_height = rospy.get_param('~image_height', 480)
        fov_horizontal = 1.047198
        self.focal_length = self.image_width / (2 * np.tan(fov_horizontal / 2))
        
        # Camera parameters
        self.depth_scale = rospy.get_param('~depth_scale', 0.001)
        self.min_depth = rospy.get_param('~min_depth', 0.5)
        self.max_depth = rospy.get_param('~max_depth', 10.0)
        
        # Get topic names from ROS params
        rgb_topic = rospy.get_param('~rgb_topic', '/camera/rgb/image_raw')
        depth_topic = rospy.get_param('~depth_topic', '/camera/depth/image_raw')
        camera_info_topic = rospy.get_param('~camera_info_topic', '/camera/rgb/camera_info')
        
        rospy.loginfo(f"Subscribing to RGB topic: {rgb_topic}")
        rospy.loginfo(f"Subscribing to depth topic: {depth_topic}")
        rospy.loginfo(f"Subscribing to camera info topic: {camera_info_topic}")
        
        # Initialize subscribers
        rgb_sub = Subscriber(rgb_topic, Image)
        depth_sub = Subscriber(depth_topic, Image)
        camera_info_sub = rospy.Subscriber(camera_info_topic, CameraInfo, self.camera_info_callback)
        
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
        self.object_location_pub = rospy.Publisher('/object_location', ObjectLocation, queue_size=10)
        
        rospy.loginfo("YOLO detector initialization completed")


    def camera_info_callback(self, msg):
        if not self.camera_info_received:
            rospy.loginfo("Received camera info message")
            self.camera_model.fromCameraInfo(msg)
            self.camera_info_received = True
    def get_3d_point(self, x, y, depth):
        """Convert 2D point and depth to 3D point"""
        if not self.camera_info_received:
            return None

        # Project 2D point to 3D using camera model
        ray = self.camera_model.projectPixelTo3dRay((x, y))
        scale = depth / ray[2]
        x3d = ray[0] * scale
        y3d = ray[1] * scale
        z3d = depth

        return Point(x3d, y3d, z3d)

    def draw_cursor(self, image, x, y, color=(0, 255, 255)):
        """Draw a cursor (crosshair) at the specified position"""
        cursor_size = 10
        thickness = 2
        
        # Draw horizontal line
        cv2.line(image, 
                 (x - cursor_size, y),
                 (x + cursor_size, y),
                 color,
                 thickness)
        
        # Draw vertical line
        cv2.line(image,
                 (x, y - cursor_size),
                 (x, y + cursor_size),
                 color,
                 thickness)
        
        # Draw small circle at intersection
        cv2.circle(image, (x, y), 3, color, -1)

    def get_depth_from_roi(self, depth_image, bbox, depth_scale=0.001):
        """Get reliable depth measurement from ROI"""
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
        
        # Handle depth format
        if depth_roi.dtype == np.uint16:
            depth_roi = depth_roi.astype(float) * depth_scale
        
        # Filter out invalid measurements
        valid_depths = depth_roi[
            (depth_roi >= self.min_depth) & 
            (depth_roi <= self.max_depth) &
            (depth_roi > 0)
        ]
        
        if len(valid_depths) > 0:
            depth = np.median(valid_depths)
            confidence = len(valid_depths) / (roi_size * roi_size * 4)
            if confidence > 0.5:
                return depth
        return None

    def publish_object_location(self, class_name, depth, cursor_x, cursor_y):
        """Publish object location information"""
        if not self.camera_info_received:
            rospy.logwarn("Camera info not received yet, cannot publish 3D location")
            return

        # Create message
        msg = ObjectLocation()
        msg.object_name = class_name
        msg.distance = depth

        # Get 3D point
        point_3d = self.get_3d_point(cursor_x, cursor_y, depth)
        if point_3d is not None:
            msg.position = point_3d
            self.object_location_pub.publish(msg)

    def create_marker(self, class_name, center_x, center_y, distance, marker_id):
        """Create visualization marker"""
        marker = Marker()
        marker.header = Header()
        marker.header.frame_id = "camera_link"
        marker.header.stamp = rospy.Time.now()
        
        marker.ns = "yolo_detections"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        # Calculate 3D position
        x = distance
        y = (center_x - self.image_width/2) * distance / self.focal_length
        z = (center_y - self.image_height/2) * distance / self.focal_length
        
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        
        # Set size based on class
        if class_name == 'person':
            size = [0.5, 0.5, 1.7]
        elif class_name == 'car':
            size = [4.5, 1.8, 1.5]
        elif class_name == 'truck':
            size = [6.0, 2.0, 2.5]
        elif class_name == 'bus':
            size = [12.0, 2.5, 3.0]
        else:
            size = [1.0, 1.0, 1.0]
        
        marker.scale.x = size[0]
        marker.scale.y = size[1]
        marker.scale.z = size[2]
        
        marker.pose.orientation.w = 1.0
        
        marker.color.a = 0.7
        marker.color.r = min(1.0, distance / self.max_depth)
        marker.color.g = min(1.0, 1.0 - (distance / self.max_depth))
        marker.color.b = 0.2
        
        return marker

    def process_detections(self, results, depth_image):
        """Process YOLO detections and calculate distances"""
        detections_msg = Float32MultiArray()
        marker_array = MarkerArray()
        detections_data = []
        
        try:
            for det in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                class_name = self.model.names[int(cls)]
                
                if class_name not in self.target_classes:
                    continue
                
                depth = self.get_depth_from_roi(depth_image, (x1, y1, x2, y2), self.depth_scale)
                
                if depth is not None:
                    detection_data = [
                        float(cls),
                        x1, y1, x2, y2,
                        float(conf),
                        depth
                    ]
                    detections_data.extend(detection_data)
                    
                    marker = self.create_marker(
                        class_name,
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                        depth,
                        len(marker_array.markers)
                    )
                    marker_array.markers.append(marker)
        
        except Exception as e:
            rospy.logerr(f"Error processing detections: {str(e)}")
            return None, None
        
        detections_msg.data = detections_data
        return detections_msg, marker_array

    def visualize_detections(self, image, results, depth_image):
        """Create debug visualization"""
        vis_image = image.copy()
        
        try:
            for det in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                class_name = self.model.names[int(cls)]
                
                if class_name not in self.target_classes:
                    continue
                
                depth = self.get_depth_from_roi(depth_image, (x1, y1, x2, y2), self.depth_scale)
                
                if depth is not None:
                    # Calculate cursor position (bottom center)
                    cursor_x = int((x1 + x2) // 2)
                    cursor_y = int(y2)
                    
                    # Publish object location
                    self.publish_object_location(class_name, depth, cursor_x, cursor_y)
                    
                    color = (
                        int(255 * (1 - depth / 5.0)),
                        int(255 * (depth / 5.0)),
                        0
                    )
                else:
                    color = (0, 0, 255)
                
                # Draw bounding box
                cv2.rectangle(vis_image, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            color, 2)
                
                # Draw cursor
                cursor_x = int((x1 + x2) // 2)
                cursor_y = int(y2)
                self.draw_cursor(vis_image, cursor_x, cursor_y, color=(0, 255, 255))
                
                # Add label
                label = f"{class_name} {conf:.2f}"
                if depth is not None:
                    label += f" {depth:.2f}m"
                
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(vis_image,
                            (int(x1), int(y1) - label_size[1] - 10),
                            (int(x1 + label_size[0]), int(y1)),
                            color, -1)
                
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
            if self.debug:
                rospy.loginfo("Received synchronized RGB and depth images")
            
            # Convert ROS messages to OpenCV images
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg)
            
            if self.debug:
                rospy.loginfo(f"RGB image shape: {rgb_image.shape}")
                rospy.loginfo(f"Depth image shape: {depth_image.shape}, type: {depth_image.dtype}")
            
            # Run YOLO detection
            results = self.model(rgb_image)
            
            # Log detection results
            detections = results.xyxy[0]
            if len(detections) > 0:
                rospy.loginfo(f"Number of detections: {len(detections)}")
                for det in detections:
                    x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                    class_name = self.model.names[int(cls)]
                    rospy.loginfo(f"Detected {class_name} with confidence {conf:.2f}")
            else:
                if self.debug:
                    rospy.loginfo("No detections in this frame")
            
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
                
                if self.debug:
                    rospy.loginfo("Published detection results and debug image")
                
        except Exception as e:
            rospy.logerr(f"Error in image callback: {str(e)}")
            import traceback
            rospy.logerr(traceback.format_exc())


def main():
    try:
        detector = YOLODepthDetector()
        rospy.loginfo("Starting YOLO detector node...")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in main: {str(e)}")
        import traceback
        rospy.logerr(traceback.format_exc())
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()