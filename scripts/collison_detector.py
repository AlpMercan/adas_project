#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from adas_project.msg import ObjectLocation
from std_msgs.msg import Bool

class CollisionDetector:
    def __init__(self):
        rospy.init_node('collision_detector_node', anonymous=True)
        self.bridge = CvBridge()
        
        # Son alınan görüntü ve nesne verilerini saklamak için
        self.latest_processed_image = None
        self.latest_object_data = None
        self.image_timestamp = None
        self.object_timestamp = None
        
        # Mesafe eşiği
        self.distance_threshold = 1.0  # 1 metre
        
        # Publisher'lar
        self.collision_pub = rospy.Publisher('/lane_object_collision', Bool, queue_size=1)
        self.processed_image_pub = rospy.Publisher('/processed_image_safe', Image, queue_size=1)
        self.visualization_pub = rospy.Publisher('/collision_visualization', Image, queue_size=1)
        
        # Subscriber'lar
        rospy.Subscriber('/processed_image', Image, self.image_callback)
        rospy.Subscriber('/object_location', ObjectLocation, self.object_callback)
        
        rospy.loginfo("Collision detector node initialized")
        self.debug = True
        
    def image_callback(self, msg):
        try:
            self.latest_processed_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_timestamp = msg.header.stamp
            self.check_collision()
        except Exception as e:
            rospy.logerr(f"Error in image callback: {str(e)}")
            
    def object_callback(self, msg):
        self.latest_object_data = msg
        self.object_timestamp = rospy.Time.now()
        self.check_collision()
        
    def is_blue_region(self, image, x, y):
        """Verilen koordinattaki pikselin mavi bölgede olup olmadığını kontrol eder"""
        if x < 0 or y < 0 or y >= image.shape[0] or x >= image.shape[1]:
            return False
            
        pixel = image[y, x]
        # BGR formatında mavi renk kontrolü
        return pixel[0] > 150 and pixel[1] < 100 and pixel[2] < 100
        
    def modify_lane_color(self, image, color):
        """Şerit rengini değiştirir"""
        modified = image.copy()
        # Mavi bölgeleri bul (BGR format)
        blue_mask = cv2.inRange(modified, (150, 0, 0), (255, 100, 100))
        
        if color == "green":
            # Mavi bölgeleri yeşil yap
            modified[blue_mask > 0] = [0, 255, 0]
        
        return modified
        
    def check_collision(self):
        """Nesne ve şerit çakışmasını kontrol eder"""
        # Eğer processed image yoksa, işlem yapma
        if self.latest_processed_image is None:
            rospy.loginfo("No processed image available")
            return
            
        try:
            # Eğer object data yoksa, normal görüntüyü yayınla
            if self.latest_object_data is None:
                self.processed_image_pub.publish(self.bridge.cv2_to_imgmsg(self.latest_processed_image, "bgr8"))
                rospy.loginfo("No object detected, publishing normal image")
                
                # Debug görselleştirme - normal durum
                if self.debug:
                    viz_image = self.latest_processed_image.copy()
                    cv2.putText(viz_image, "NO OBJECT", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                               (255, 255, 0), 2)
                    self.visualization_pub.publish(self.bridge.cv2_to_imgmsg(viz_image, "bgr8"))
                return
            
            # Nesnenin piksel koordinatları
            x = self.latest_object_data.pixel_x
            y = self.latest_object_data.pixel_y
            distance = self.latest_object_data.distance
            
            # Çakışma kontrolü
            is_collision = self.is_blue_region(self.latest_processed_image, x, y)
            
            # Sonucu yayınla
            self.collision_pub.publish(Bool(data=is_collision))
            
            # Debug görselleştirme için görüntüyü hazırla
            viz_image = self.latest_processed_image.copy()
            
            if is_collision:
                if distance <= self.distance_threshold:
                    # Tehlikeli durum: Boş görüntü yolla
                    empty_image = np.zeros_like(self.latest_processed_image)
                    self.processed_image_pub.publish(self.bridge.cv2_to_imgmsg(empty_image, "bgr8"))
                    rospy.logwarn(f"Dangerous collision detected! Distance: {distance:.2f}m")
                    status = "STOP!"
                    color = (0, 0, 255)  # Kırmızı
                else:
                    # Uyarı durumu: Yeşil şeritli görüntü yolla
                    warning_image = self.modify_lane_color(self.latest_processed_image, "green")
                    self.processed_image_pub.publish(self.bridge.cv2_to_imgmsg(warning_image, "bgr8"))
                    rospy.loginfo(f"Warning! Object in lane but at safe distance: {distance:.2f}m")
                    status = "WARNING"
                    color = (0, 255, 0)  # Yeşil
            else:
                # Güvenli durum: Normal görüntü yolla
                self.processed_image_pub.publish(self.bridge.cv2_to_imgmsg(self.latest_processed_image, "bgr8"))
                rospy.loginfo("No collision detected")
                status = "SAFE"
                color = (255, 255, 0)  # Sarı
            
            # Debug görselleştirme
            if self.debug:
                # Nesne konumunu göster
                cv2.circle(viz_image, (x, y), 5, color, -1)
                cv2.circle(viz_image, (x, y), 7, color, 2)
                
                # Nesne bilgilerini ekle
                object_text = f"{self.latest_object_data.object_name} - {distance:.2f}m"
                cv2.putText(viz_image, object_text, 
                           (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Durum bilgisini ekranın üst kısmına ekle
                cv2.putText(viz_image, status, 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                self.visualization_pub.publish(self.bridge.cv2_to_imgmsg(viz_image, "bgr8"))
                
        except Exception as e:
            rospy.logerr(f"Error in check_collision: {str(e)}")
            # Hata durumunda normal görüntüyü yayınla
            if self.latest_processed_image is not None:
                self.processed_image_pub.publish(self.bridge.cv2_to_imgmsg(self.latest_processed_image, "bgr8"))

def main():
    try:
        detector = CollisionDetector()
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()