#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from collections import deque

class LaneDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback, queue_size=1)
        self.image_pub = rospy.Publisher('/processed_image', Image, queue_size=1)
        
        # Transformation matrices
        self.M = None
        self.Minv = None
        
        # Lane detection history için deque
        self.left_fit_history = deque(maxlen=5)
        self.right_fit_history = deque(maxlen=5)
        
        # Confidence değerleri
        self.min_pixels = 50
        self.lane_detection_confidence = 0.0

    def region_of_interest(self, img):
        """Define a region of interest that focuses on the road with wider view"""
        height = img.shape[0]
        width = img.shape[1]
        
        # ROI'yi daha geniş ve daha yükseğe çıkan bir şekilde tanımlayalım
        roi_vertices = np.array([
            [(0, height),                    # Sol alt
            (0, height * 0.5),             # Sol orta - daha yukarı çektik
            (width * 0.25, height * 0.35),  # Sol üst - daha yukarı
            (width * 0.75, height * 0.35),  # Sağ üst - daha yukarı
            (width, height * 0.5),         # Sağ orta - daha yukarı çektik
            (width, height)]               # Sağ alt
        ], dtype=np.int32)
        
        # Görselleştirme için ROI'yi çiz
        roi_visualization = img.copy()
        cv2.polylines(roi_visualization, [roi_vertices], True, (0, 255, 0), 2)
        cv2.imshow('ROI Visualization', roi_visualization)
        
        # Maske oluştur
        mask = np.zeros_like(img)
        if len(img.shape) > 2:
            channel_count = img.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        
        cv2.fillPoly(mask, [roi_vertices], ignore_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        
        return masked_image
    


    def perspective_transform(self, img):
        """Apply perspective transform with corrected points"""
        height = img.shape[0]
        width = img.shape[1]
        
        
        src_points = np.float32([
            [width * 0.35, height * 0.6],    
            [width * 0.65, height * 0.6],    
            [width * 0.9, height * 0.95],    
            [width * 0.1, height * 0.95]     
        ])
        
        
        dst_points = np.float32([
            [width * 0.2, 0],              
            [width * 0.8, 0],              
            [width * 0.8, height],         
            [width * 0.2, height]          
        ])
        
        self.M = cv2.getPerspectiveTransform(src_points, dst_points)
        self.Minv = cv2.getPerspectiveTransform(dst_points, src_points)
        
        warped = cv2.warpPerspective(img, self.M, (width, height))
        
        transform_visualization = img.copy()
        cv2.polylines(transform_visualization, [src_points.astype(np.int32)], True, (255, 0, 0), 2)
        cv2.imshow('Perspective Transform Points', transform_visualization)
        
        return warped
    


    def color_threshold(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        combined_mask = cv2.bitwise_or(white_mask, red_mask)
        
        kernel = np.ones((3,3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        return combined_mask

    def find_lane_pixels(self, binary_warped):
        """Find lane pixels with improved detection for curves"""
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        
        midpoint = int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        nwindows = 15        
        margin = 150         
        minpix = 30         
        
        window_height = int(binary_warped.shape[0]//nwindows)
        
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        left_lane_inds = []
        right_lane_inds = []
        
        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            
            win_xleft_low = max(0, leftx_current - margin)
            win_xleft_high = min(binary_warped.shape[1], leftx_current + margin)
            win_xright_low = max(0, rightx_current - margin)
            win_xright_high = min(binary_warped.shape[1], rightx_current + margin)
            
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                            (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
        
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            return None, None, None, None
        
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        return leftx, lefty, rightx, righty

    def fit_polynomial(self, binary_warped):
        leftx, lefty, rightx, righty = self.find_lane_pixels(binary_warped)
        
        if leftx is None or rightx is None:
            return None, None, None
            
        try:
            # 3. dereceden polinom uydur (daha esnek eğriler için)
            left_fit = np.polyfit(lefty, leftx, 3)
            right_fit = np.polyfit(righty, rightx, 3)
            
            # Şerit tespiti güvenilirliğini hesapla
            left_points = len(leftx)
            right_points = len(rightx)
            self.lane_detection_confidence = min(1.0, (left_points + right_points) / (2 * self.min_pixels))
            
            # Geçmiş değerleri kaydet
            self.left_fit_history.append(left_fit)
            self.right_fit_history.append(right_fit)
            
            # Son 5 frame'in ortalamasını al
            left_fit = np.mean(self.left_fit_history, axis=0)
            right_fit = np.mean(self.right_fit_history, axis=0)
            
            # Y değerlerini oluştur
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
            
            # X değerlerini hesapla (3. dereceden polinom için)
            left_fitx = (left_fit[0] * ploty**3 + 
                        left_fit[1] * ploty**2 + 
                        left_fit[2] * ploty + 
                        left_fit[3])
            right_fitx = (right_fit[0] * ploty**3 + 
                         right_fit[1] * ploty**2 + 
                         right_fit[2] * ploty + 
                         right_fit[3])
            
            return left_fitx, right_fitx, ploty
            
        except (TypeError, np.linalg.LinAlgError):
            return None, None, None

    def draw_lanes(self, original_img, left_fitx, right_fitx, ploty):
        """Draw detected lanes with blue fill between lanes"""
        warp_zero = np.zeros_like(original_img[:,:,0]).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Şerit noktalarını yeniden düzenle
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Şeritler arası alanı mavi renk ile doldur (B, G, R)
        cv2.fillPoly(color_warp, np.int_([pts]), (255, 0, 0))  # Mavi dolgu
        
        # Şeritleri daha belirgin renklerle çiz
        cv2.polylines(color_warp, np.int_([pts_left]), False, (0, 0, 255), thickness=15)    # Sol şerit kırmızı
        cv2.polylines(color_warp, np.int_([pts_right]), False, (0, 0, 255), thickness=15)   # Sağ şerit kırmızı
        
        # Perspektif dönüşümünü geri al
        newwarp = cv2.warpPerspective(color_warp, self.Minv, 
                                    (original_img.shape[1], original_img.shape[0]))
        
        # Güven seviyesini görüntüye ekle
        confidence_text = f"Lane Detection Confidence: {self.lane_detection_confidence:.2f}"
        cv2.putText(newwarp, confidence_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Görüntüleri birleştir (opaklığı biraz artırdım)
        result = cv2.addWeighted(original_img, 1, newwarp, 0.4, 0)
        return result

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # ROI uygula
            roi_image = self.region_of_interest(cv_image)
            cv2.imshow('ROI', roi_image)
            
            # Perspektif dönüşümü uygula
            warped_image = self.perspective_transform(roi_image)
            cv2.imshow('Warped', warped_image)
            
            # Renk ve kenar tespiti
            binary = self.color_threshold(warped_image)
            cv2.imshow('Binary', binary)
            
            # Şeritleri tespit et ve çiz
            left_fitx, right_fitx, ploty = self.fit_polynomial(binary)
            
            if left_fitx is not None and right_fitx is not None:
                result = self.draw_lanes(cv_image, left_fitx, right_fitx, ploty)
                cv2.imshow('Result', result)
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(result, "bgr8"))
            else:
                cv2.imshow('Result', cv_image)
                self.image_pub.publish(msg)
            
            cv2.waitKey(1)
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")

def main():
    rospy.init_node('lane_detector')
    LaneDetector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()