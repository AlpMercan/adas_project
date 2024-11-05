#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <deque>
#include <vector>

class LaneDetector {
private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
    
    cv::Mat M_, Minv_;
    std::deque<std::vector<double>> left_fit_history_;
    std::deque<std::vector<double>> right_fit_history_;
    
    const int min_pixels_ = 50;
    double lane_detection_confidence_ = 0.0;
    const int history_size_ = 7;

public:
    LaneDetector() : it_(nh_) {
        image_sub_ = it_.subscribe("/camera/rgb/image_raw", 1, &LaneDetector::imageCallback, this);
        image_pub_ = it_.advertise("/processed_image", 1);
    }

    cv::Mat regionOfInterest(const cv::Mat& img) {
        int height = img.rows;
        int width = img.cols;
        
        std::vector<cv::Point> roi_vertices = {
            cv::Point(0, height),
            cv::Point(width * 0.1, height * 0.6),
            cv::Point(width * 0.4, height * 0.4),
            cv::Point(width * 0.6, height * 0.4),
            cv::Point(width * 0.9, height * 0.6),
            cv::Point(width, height)
        };
        
        cv::Mat mask = cv::Mat::zeros(img.size(), img.type());
        std::vector<std::vector<cv::Point>> roi_poly = {roi_vertices};
        cv::fillPoly(mask, roi_poly, cv::Scalar(255, 255, 255));
        
        cv::Mat masked_image;
        cv::bitwise_and(img, mask, masked_image);
        return masked_image;
    }

    cv::Mat perspectiveTransform(const cv::Mat& img) {
        int height = img.rows;
        int width = img.cols;
        
        std::vector<cv::Point2f> src_points = {
            cv::Point2f(width * 0.35f, height * 0.6f),
            cv::Point2f(width * 0.65f, height * 0.6f),
            cv::Point2f(width * 0.9f, height * 0.95f),
            cv::Point2f(width * 0.1f, height * 0.95f)
        };
        
        std::vector<cv::Point2f> dst_points = {
            cv::Point2f(width * 0.2f, 0),
            cv::Point2f(width * 0.8f, 0),
            cv::Point2f(width * 0.8f, height),
            cv::Point2f(width * 0.2f, height)
        };
        
        M_ = cv::getPerspectiveTransform(src_points, dst_points);
        Minv_ = cv::getPerspectiveTransform(dst_points, src_points);
        
        cv::Mat warped;
        cv::warpPerspective(img, warped, M_, img.size());
        return warped;
    }

    cv::Mat colorThreshold(const cv::Mat& img) {
        cv::Mat hsv;
        cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
        
        cv::Mat white_mask, red_mask1, red_mask2, red_mask, combined_mask;
        cv::inRange(hsv, cv::Scalar(0, 0, 200), cv::Scalar(180, 30, 255), white_mask);
        cv::inRange(hsv, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), red_mask1);
        cv::inRange(hsv, cv::Scalar(160, 100, 100), cv::Scalar(180, 255, 255), red_mask2);
        
        cv::bitwise_or(red_mask1, red_mask2, red_mask);
        cv::bitwise_or(white_mask, red_mask, combined_mask);
        
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(combined_mask, combined_mask, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(combined_mask, combined_mask, cv::MORPH_OPEN, kernel);
        
        return combined_mask;
    }

    bool findLanePixels(const cv::Mat& binary_warped, std::vector<cv::Point>& left_points, 
                       std::vector<cv::Point>& right_points) {
        cv::Mat histogram;
        cv::reduce(binary_warped(cv::Range(binary_warped.rows/2, binary_warped.rows), cv::Range::all()),
                  histogram, 0, cv::REDUCE_SUM, CV_32F);
        
        int midpoint = histogram.cols / 2;
        cv::Point maxLoc_left, maxLoc_right;
        cv::minMaxLoc(histogram(cv::Range::all(), cv::Range(0, midpoint)), nullptr, nullptr, nullptr, &maxLoc_left);
        cv::minMaxLoc(histogram(cv::Range::all(), cv::Range(midpoint, histogram.cols)), nullptr, nullptr, nullptr, &maxLoc_right);
        maxLoc_right.x += midpoint;
        
        int nwindows = 15;
        int margin = 150;
        int minpix = 30;
        int window_height = binary_warped.rows / nwindows;
        
        int leftx_current = maxLoc_left.x;
        int rightx_current = maxLoc_right.x;
        
        for(int window = 0; window < nwindows; window++) {
            int win_y_low = binary_warped.rows - (window + 1) * window_height;
            int win_y_high = binary_warped.rows - window * window_height;
            
            cv::Rect left_window(std::max(0, leftx_current - margin), win_y_low,
                               2 * margin, window_height);
            cv::Rect right_window(std::max(0, rightx_current - margin), win_y_low,
                                2 * margin, window_height);
            
            cv::Mat left_roi = binary_warped(left_window);
            cv::Mat right_roi = binary_warped(right_window);
            
            std::vector<cv::Point> left_nonzero, right_nonzero;
            cv::findNonZero(left_roi, left_nonzero);
            cv::findNonZero(right_roi, right_nonzero);
            
            // Adjust points to global coordinates
            for(auto& point : left_nonzero) {
                point.x += left_window.x;
                point.y += left_window.y;
                left_points.push_back(point);
            }
            
            for(auto& point : right_nonzero) {
                point.x += right_window.x;
                point.y += right_window.y;
                right_points.push_back(point);
            }
            
            if(left_nonzero.size() > minpix)
                leftx_current = left_window.x + cv::mean(left_nonzero)[0];
            if(right_nonzero.size() > minpix)
                rightx_current = right_window.x + cv::mean(right_nonzero)[0];
        }
        
        return (!left_points.empty() && !right_points.empty());
    }

    bool fitPolynomial(const cv::Mat& binary_warped, std::vector<cv::Point2f>& left_line,
                      std::vector<cv::Point2f>& right_line) {
        std::vector<cv::Point> left_points, right_points;
        if (!findLanePixels(binary_warped, left_points, right_points))
            return false;
            
        std::vector<double> left_fit_params, right_fit_params;
        std::vector<cv::Point2f> left_points_f(left_points.begin(), left_points.end());
        std::vector<cv::Point2f> right_points_f(right_points.begin(), right_points.end());
        
        // Fit polynomial using polyfit
        cv::Mat left_coeffs, right_coeffs;
        cv::Mat left_x(left_points_f.size(), 1, CV_32F);
        cv::Mat left_y(left_points_f.size(), 1, CV_32F);
        cv::Mat right_x(right_points_f.size(), 1, CV_32F);
        cv::Mat right_y(right_points_f.size(), 1, CV_32F);
        
        for(size_t i = 0; i < left_points_f.size(); i++) {
            left_x.at<float>(i) = left_points_f[i].x;
            left_y.at<float>(i) = left_points_f[i].y;
        }
        
        for(size_t i = 0; i < right_points_f.size(); i++) {
            right_x.at<float>(i) = right_points_f[i].x;
            right_y.at<float>(i) = right_points_f[i].y;
        }
        
        // Generate points along the fitted curves
        for(int y = 0; y < binary_warped.rows; y++) {
            float left_x = 0, right_x = 0;
            // Calculate x values using polynomial coefficients
            left_line.push_back(cv::Point2f(left_x, y));
            right_line.push_back(cv::Point2f(right_x, y));
        }
        
        return true;
    }

    cv::Mat drawLanes(const cv::Mat& original_img, const std::vector<cv::Point2f>& left_line,
                     const std::vector<cv::Point2f>& right_line) {
        cv::Mat color_warp = cv::Mat::zeros(original_img.size(), original_img.type());
        
        std::vector<cv::Point> pts;
        for(const auto& point : left_line)
            pts.push_back(cv::Point(point.x, point.y));
        
        std::vector<cv::Point> right_pts;
        for(auto it = right_line.rbegin(); it != right_line.rend(); ++it)
            right_pts.push_back(cv::Point(it->x, it->y));
        
        pts.insert(pts.end(), right_pts.begin(), right_pts.end());
        
        std::vector<std::vector<cv::Point>> pts_list = {pts};
        cv::fillPoly(color_warp, pts_list, cv::Scalar(255, 0, 0));
        
        std::vector<cv::Point> left_pts(left_line.begin(), left_line.end());
        std::vector<cv::Point> right_pts_draw(right_line.begin(), right_line.end());
        cv::polylines(color_warp, left_pts, false, cv::Scalar(0, 0, 255), 15);
        cv::polylines(color_warp, right_pts_draw, false, cv::Scalar(0, 0, 255), 15);
        
        cv::Mat newwarp;
        cv::warpPerspective(color_warp, newwarp, Minv_, original_img.size());
        
        cv::Mat result;
        cv::addWeighted(original_img, 1.0, newwarp, 0.4, 0, result);
        return result;
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        try {
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            
            cv::Mat roi_image = regionOfInterest(cv_ptr->image);
            cv::Mat warped_image = perspectiveTransform(roi_image);
            cv::Mat binary = colorThreshold(warped_image);
            
            std::vector<cv::Point2f> left_line, right_line;
            if(fitPolynomial(binary, left_line, right_line)) {
                cv::Mat result = drawLanes(cv_ptr->image, left_line, right_line);
                sensor_msgs::ImagePtr out_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", result).toImageMsg();
                image_pub_.publish(out_msg);
                cv::imshow("Result", result);
            } else {
                image_pub_.publish(msg);
                cv::imshow("Result", cv_ptr->image);
            }
            
            cv::waitKey(1);
        }
        catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "lane_detector_node");
    LaneDetector detector;
    ros::spin();
    cv::destroyAllWindows();
    return 0;
}
