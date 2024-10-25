#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class MotionController:
    def __init__(self):
        rospy.init_node('motion_controller_node', anonymous=True)
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
        
        # Publishers and subscribers
        self.image_sub = rospy.Subscriber('/processed_image', Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # State variables
        self.no_detection_counter = 0
        self.max_no_detection = 5
        
        rospy.loginfo("Lane following motion controller initialized")

    def detect_lane_markers(self, image):
        """Mavi yol alanını tespit et"""
        try:
            # BGR'den HSV'ye dönüştür
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Mavi renk için HSV aralığı
            lower_blue = np.array([100, 50, 50])   # Mavi için alt sınır
            upper_blue = np.array([130, 255, 255]) # Mavi için üst sınır
            
            # Mavi maske oluştur
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # Gürültüyü azalt
            kernel = np.ones((5,5), np.uint8)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
            
            # Görüntüyü göster (debug için)
            cv2.imshow('Blue Path Mask', blue_mask)
            cv2.waitKey(1)
            
            return blue_mask
            
        except Exception as e:
            rospy.logerr(f"Error in detect_lane_markers: {str(e)}")
            return None

    def calculate_center_offset(self, image):
        """Mavi yolun merkezinden offset'i hesapla"""
        try:
            # Görüntü boyutlarını al
            height, width = image.shape[:2]
            
            # Alt yarıyı ROI olarak al (biraz daha yukarı bak)
            roi_height = int(height * 0.6)  # ROI'yi artırdık
            roi_top = height - roi_height
            roi = image[roi_top:height, :]
            
            # Mavi yolu tespit et
            path_mask = self.detect_lane_markers(roi)
            
            if path_mask is None:
                return None
            
            # Yol noktalarını bul
            nonzero = cv2.findNonZero(path_mask)
            
            if nonzero is None or len(nonzero) < 100:  # Minimum nokta sayısını artırdık
                rospy.logwarn("Not enough path points detected")
                return None
            
            # Yol merkezini hesapla
            points_mean = np.mean(nonzero, axis=0)
            center_x = points_mean[0][0]
            
            # Görüntü merkezinden offset'i hesapla
            image_center = width / 2
            offset = -(center_x - image_center)  # Negatif işaret dönüş yönünü düzeltir
            
            # Debug görüntüsü
            debug_image = cv2.cvtColor(path_mask, cv2.COLOR_GRAY2BGR)
            cv2.circle(debug_image, (int(center_x), int(roi_height/2)), 5, (0, 0, 255), -1)
            cv2.line(debug_image, (int(width/2), 0), (int(width/2), roi_height), (0, 255, 0), 1)
            cv2.imshow('Path Detection Debug', debug_image)
            cv2.waitKey(1)
            
            # Debug bilgisi
            rospy.loginfo(f"Path center: {center_x:.1f}, Offset: {offset:.1f}")
            
            return offset
            
        except Exception as e:
            rospy.logerr(f"Error in calculate_center_offset: {str(e)}")
            return None


    def pid_control(self, error):
        """PID kontrolü uygula"""
        if error is None:
            self.integral = 0
            self.last_error = 0
            return 0
        
        # PID bileşenlerini hesapla
        proportional = self.Kp * error
        
        self.integral = np.clip(self.integral + error, -self.max_integral, self.max_integral)
        integral = self.Ki * self.integral
        
        derivative = self.Kd * (error - self.last_error)
        
        # Toplam kontrol sinyali
        angular_velocity = proportional + integral + derivative
        
        # Açısal hızı limitle
        angular_velocity = np.clip(angular_velocity, -self.max_angular_speed, self.max_angular_speed)
        
        self.last_error = error
        return angular_velocity

    def publish_velocity(self, linear, angular):
        """Hız komutlarını yayınla"""
        try:
            twist = Twist()
            
            # Keskin dönüşlerde yavaşla
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
        """Sıfır hız yayınla"""
        try:
            twist = Twist()
            self.cmd_vel_pub.publish(twist)
        except Exception as e:
            rospy.logerr(f"Error publishing zero velocity: {str(e)}")

    def image_callback(self, msg):
        """Gelen görüntüyü işle ve robotu kontrol et"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Şerit offset'ini hesapla
            offset = self.calculate_center_offset(cv_image)
            
            if offset is not None:
                self.no_detection_counter = 0
                
                # PID ile açısal hızı hesapla
                angular_velocity = self.pid_control(offset)
                
                # Hızları yayınla
                self.publish_velocity(self.linear_speed, angular_velocity)
            else:
                self.no_detection_counter += 1
                if self.no_detection_counter >= self.max_no_detection:
                    rospy.logwarn("Lost lane markers")
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