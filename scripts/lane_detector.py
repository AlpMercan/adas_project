#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class LaneDetector:
    def __init__(self):
        """Initialize the lane detector with support for three lanes"""
        rospy.init_node('lane_detector_node', anonymous=True)
        self.bridge = CvBridge()
        
        # Create subscriber and publisher
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        self.image_pub = rospy.Publisher('/processed_image', Image, queue_size=1)
        
        # Define image dimensions
        self.width = 800
        self.height = 480
        
        # Define ROI
        self.roi = (
            0,                          # x start
            int(self.height * 0.35),    # y start
            int(self.width * 0.95),     # width
            int(self.height * 0.65)     # height
        )
        
        # Lane detection parameters
        self.nwindows = 9
        self.margin = 100
        self.minpix = 50
        self.smooth_factor = 0.8
        
        # Initialize polynomial fits for all three lanes
        self.left_fit = None
        self.middle_fit = None
        self.right_fit = None
        
        # Setup perspective transform
        self.setup_perspective_transform()
        
        rospy.loginfo("Three-lane detector initialized")

    def setup_perspective_transform(self):
        """Define the source and destination points for perspective transform"""
        x, y, w, h = self.roi
        
        self.src_points = np.float32([
            [w * 0.1, h * 0.95],    # Bottom left
            [w * 0.9, h * 0.95],    # Bottom right
            [w * 0.6, h * 0.3],     # Top right
            [w * 0.4, h * 0.3]      # Top left
        ])
        
        self.dst_points = np.float32([
            [w * 0.2, h],           # Bottom left
            [w * 0.8, h],           # Bottom right
            [w * 0.8, 0],           # Top right
            [w * 0.2, 0]            # Top left
        ])
        
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.Minv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

    def color_threshold(self, image):
        """Apply color thresholding to detect white and red colors"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # White color mask (adjusted parameters)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Red color mask (two ranges because red wraps around in HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(white_mask, red_mask)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((3,3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Create debug visualization
        color_debug = np.dstack((combined_mask, combined_mask, combined_mask))
        color_debug[white_mask > 0] = [255, 255, 255]  # White
        color_debug[red_mask > 0] = [0, 0, 255]        # Red
        
        cv2.imshow('Color Threshold Debug', color_debug)
        
        return combined_mask
    def find_lane_pixels(self, binary_warped):
        """Find lane pixels using sliding window method with three lane detection"""
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        
        # Create debug visualization
        debug_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        
        # Plot histogram
        hist_img = np.zeros((100, binary_warped.shape[1], 3), dtype=np.uint8)
        histogram_scaled = (histogram * 100 / np.max(histogram)).astype(np.int32)
        for x, h in enumerate(histogram_scaled):
            cv2.line(hist_img, (x, 99), (x, 99-h), (0, 255, 0), 1)
        
        # Find peaks for three lanes
        # Divide image into three sections
        third_point = np.int32(histogram.shape[0]//3)
        
        # Find the maximum for each third of the image
        leftx_base = np.argmax(histogram[:third_point])
        middlex_base = np.argmax(histogram[third_point:2*third_point]) + third_point
        rightx_base = np.argmax(histogram[2*third_point:]) + 2*third_point
        
        # Draw peak points on histogram
        cv2.circle(hist_img, (leftx_base, 99-histogram_scaled[leftx_base]), 5, (255, 0, 0), -1)  # Blue
        cv2.circle(hist_img, (middlex_base, 99-histogram_scaled[middlex_base]), 5, (0, 255, 0), -1)  # Green
        cv2.circle(hist_img, (rightx_base, 99-histogram_scaled[rightx_base]), 5, (0, 0, 255), -1)  # Red
        
        cv2.imshow('Lane Detection Histogram', hist_img)
        
        # Set height of windows
        window_height = np.int32(binary_warped.shape[0]//self.nwindows)
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions to be updated for each window
        leftx_current = leftx_base
        middlex_current = middlex_base
        rightx_current = rightx_base
        
        # Lists to store lane pixel indices
        left_lane_inds = []
        middle_lane_inds = []
        right_lane_inds = []
        
        # Step through the windows one by one
        for window in range(self.nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            
            # Define window boundaries for all three lanes
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            
            win_xmid_low = middlex_current - self.margin
            win_xmid_high = middlex_current + self.margin
            
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            
            # Draw the windows on the debug image
            cv2.rectangle(debug_img, (win_xleft_low, win_y_low), 
                         (win_xleft_high, win_y_high), (255, 0, 0), 2)  # Blue
            cv2.rectangle(debug_img, (win_xmid_low, win_y_low), 
                         (win_xmid_high, win_y_high), (0, 255, 0), 2)  # Green
            cv2.rectangle(debug_img, (win_xright_low, win_y_low), 
                         (win_xright_high, win_y_high), (0, 0, 255), 2)  # Red
            
            # Identify the nonzero pixels in x and y within each window
            good_left_inds = ((nonzeroy >= win_y_low) & 
                            (nonzeroy < win_y_high) & 
                            (nonzerox >= win_xleft_low) & 
                            (nonzerox < win_xleft_high)).nonzero()[0]
            
            good_mid_inds = ((nonzeroy >= win_y_low) & 
                            (nonzeroy < win_y_high) & 
                            (nonzerox >= win_xmid_low) & 
                            (nonzerox < win_xmid_high)).nonzero()[0]
            
            good_right_inds = ((nonzeroy >= win_y_low) & 
                             (nonzeroy < win_y_high) & 
                             (nonzerox >= win_xright_low) & 
                             (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            middle_lane_inds.append(good_mid_inds)
            right_lane_inds.append(good_right_inds)
            
            # If found enough pixels, recenter next window
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_mid_inds) > self.minpix:
                middlex_current = np.int32(np.mean(nonzerox[good_mid_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))
        
        # Concatenate the arrays of indices
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            middle_lane_inds = np.concatenate(middle_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            return [], [], [], [], [], [], debug_img

        # Extract line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        middlex = nonzerox[middle_lane_inds]
        middley = nonzeroy[middle_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Color the detected pixels for debugging
        debug_img[lefty, leftx] = [255, 0, 0]    # Blue for left lane
        debug_img[middley, middlex] = [0, 255, 0]  # Green for middle lane
        debug_img[righty, rightx] = [0, 0, 255]    # Red for right lane
        
        return leftx, lefty, middlex, middley, rightx, righty, debug_img

    def fit_polynomial(self, leftx, lefty, middlex, middley, rightx, righty):
        """Fit polynomials to all three lanes"""
        # Minimum points required for fitting
        min_points = 10
        
        # Initialize fits
        left_fit = self.left_fit
        middle_fit = self.middle_fit
        right_fit = self.right_fit
        
        # Fit new polynomials if enough points are found
        if len(leftx) > min_points:
            left_fit = np.polyfit(lefty, leftx, 2)
        if len(middlex) > min_points:
            middle_fit = np.polyfit(middley, middlex, 2)
        if len(rightx) > min_points:
            right_fit = np.polyfit(righty, rightx, 2)
        
        # Apply smoothing if previous fits exist
        if self.left_fit is not None and left_fit is not None:
            left_fit = self.smooth_factor * self.left_fit + (1 - self.smooth_factor) * left_fit
        if self.middle_fit is not None and middle_fit is not None:
            middle_fit = self.smooth_factor * self.middle_fit + (1 - self.smooth_factor) * middle_fit
        if self.right_fit is not None and right_fit is not None:
            right_fit = self.smooth_factor * self.right_fit + (1 - self.smooth_factor) * right_fit
        
        # Store the new fits
        self.left_fit = left_fit
        self.middle_fit = middle_fit
        self.right_fit = right_fit
        
        return left_fit, middle_fit, right_fit

    def draw_lanes(self, binary_warped, left_fit, middle_fit, right_fit):
        """Draw all three lanes with distinct colors"""
        # Generate y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        
        # Calculate x values for each lane
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            middle_fitx = middle_fit[0]*ploty**2 + middle_fit[1]*ploty + middle_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except (TypeError, AttributeError):
            return None
        
        # Create image to draw on
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        
        # Draw left lane area (blue with transparency)
        if left_fit is not None and middle_fit is not None:
            pts_left = np.hstack((
                np.array([np.transpose(np.vstack([left_fitx, ploty]))]),
                np.array([np.flipud(np.transpose(np.vstack([middle_fitx, ploty])))])
            ))
            overlay = out_img.copy()
            cv2.fillPoly(overlay, np.int_([pts_left]), (255, 0, 0))
            out_img = cv2.addWeighted(overlay, 0.3, out_img, 0.7, 0)
        
        # Draw middle lane area (green with transparency)
        if middle_fit is not None and right_fit is not None:
            pts_middle = np.hstack((
                np.array([np.transpose(np.vstack([middle_fitx, ploty]))]),
                np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            ))
            overlay = out_img.copy()
            cv2.fillPoly(overlay, np.int_([pts_middle]), (0, 255, 0))
            out_img = cv2.addWeighted(overlay, 0.3, out_img, 0.7, 0)
        
        # Draw the actual lane lines
        if left_fit is not None:
            for i in range(len(ploty)-1):
                cv2.line(out_img, 
                        (int(left_fitx[i]), int(ploty[i])), 
                        (int(left_fitx[i+1]), int(ploty[i+1])), 
                        (255, 200, 0), 4)  # Bright blue
        
        if middle_fit is not None:
            for i in range(len(ploty)-1):
                cv2.line(out_img, 
                        (int(middle_fitx[i]), int(ploty[i])), 
                        (int(middle_fitx[i+1]), int(ploty[i+1])), 
                        (0, 255, 0), 4)  # Green
        
        if right_fit is not None:
            for i in range(len(ploty)-1):
                cv2.line(out_img, 
                        (int(right_fitx[i]), int(ploty[i])), 
                        (int(right_fitx[i+1]), int(ploty[i+1])), 
                        (0, 200, 255), 4)  # Bright red
        
        # Add curve information
        font = cv2.FONT_HERSHEY_SIMPLEX
        if left_fit is not None:
            cv2.putText(out_img, f'Left curve: {left_fit[0]:.6f}', 
                       (30, 30), font, 0.7, (255, 255, 255), 2)
        if middle_fit is not None:
            cv2.putText(out_img, f'Middle curve: {middle_fit[0]:.6f}', 
                       (30, 60), font, 0.7, (255, 255, 255), 2)
        if right_fit is not None:
            cv2.putText(out_img, f'Right curve: {right_fit[0]:.6f}', 
                       (30, 90), font, 0.7, (255, 255, 255), 2)
        
        return out_img

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Extract ROI
            x, y, w, h = self.roi
            roi_image = cv_image[y:y+h, x:x+w]
            
            # Apply perspective transform
            warped = cv2.warpPerspective(roi_image, self.M, (w, h))
            
            # Apply color thresholding
            binary_warped = self.color_threshold(warped)
            
            # Find lane pixels for all three lanes
            leftx, lefty, middlex, middley, rightx, righty, window_img = self.find_lane_pixels(binary_warped)
            
            # Create debug image showing detected points
            debug_img = window_img.copy()
            
            # Color the detected points
            if len(leftx) > 0:
                debug_img[lefty, leftx] = [255, 0, 0]  # Blue for left lane
            if len(middlex) > 0:
                debug_img[middley, middlex] = [0, 255, 0]  # Green for middle lane
            if len(rightx) > 0:
                debug_img[righty, rightx] = [0, 0, 255]  # Red for right lane
            
            # Show debug visualization of detected points
            cv2.imshow('Detected Lane Points', debug_img)
            
            # Fit polynomials for all three lanes
            left_fit, middle_fit, right_fit = self.fit_polynomial(
                leftx, lefty, middlex, middley, rightx, righty)
            
            # Draw lanes with enhanced visualization
            if left_fit is not None or middle_fit is not None or right_fit is not None:
                lane_visualization = self.draw_lanes(binary_warped, left_fit, middle_fit, right_fit)
                
                if lane_visualization is not None:
                    # Unwarp the lane visualization
                    unwarped_lane = cv2.warpPerspective(lane_visualization, self.Minv, (w, h))
                    
                    # Combine with original image
                    result = cv_image.copy()
                    mask = (unwarped_lane != 0).any(axis=2)
                    result[y:y+h, x:x+w][mask] = cv2.addWeighted(
                        result[y:y+h, x:x+w][mask], 0.7,
                        unwarped_lane[mask], 0.3, 0)
                    
                    # Add debug information on the original image
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2
                    color = (0, 255, 0)  # Green text
                    
                    # Display number of points detected for each lane
                    debug_info = [
                        f"Left Lane Points: {len(leftx)}",
                        f"Middle Lane Points: {len(middlex)}",
                        f"Right Lane Points: {len(rightx)}"
                    ]
                    
                    y_offset = 30
                    for i, text in enumerate(debug_info):
                        cv2.putText(result, text, 
                                  (30, y_offset + i*30),
                                  font, font_scale, color, 
                                  thickness)
                    
                    # Show all visualization windows
                    cv2.imshow('Original with Lanes', result)
                    cv2.imshow('Bird\'s Eye View with Lanes', lane_visualization)
                    cv2.imshow('Window View', window_img)
                    
                    # Show binary warped image for debugging
                    binary_debug = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
                    cv2.imshow('Binary Warped', binary_debug)
                    
                    cv2.waitKey(1)
                    
                    # Publish processed image
                    processed_msg = self.bridge.cv2_to_imgmsg(result, "bgr8")
                    self.image_pub.publish(processed_msg)
                    
            else:
                rospy.logwarn("No lanes detected in this frame")
                
        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")
            rospy.logerr(f"Error occurred at line {e.__traceback__.tb_lineno}")

def main():
    try:
        detector = LaneDetector()
        
        # Print initialization message
        rospy.loginfo("Lane detector node is running. Press Ctrl+C to terminate.")
        
        # Keep the node running
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Lane detector node terminated.")
    except Exception as e:
        rospy.logerr(f"Unexpected error in lane detector node: {str(e)}")
    finally:
        # Cleanup
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()