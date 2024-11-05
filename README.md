# Advanced Driver Assistance System (ADAS)

## Overview
This repository contains a comprehensive implementation of an Advanced Driver Assistance System (ADAS) using ROS (Robot Operating System). The system integrates lane detection, motion control, and safety features to provide autonomous driving capabilities with a focus on safety and reliability.

## System Architecture
The system consists of three main components:
1. Lane Detection System
2. Motion Control System
3. Safety and Monitoring System

### Key Features
- Real-time lane detection and tracking
- Adaptive motion control with PID
- Obstacle detection and avoidance
- Automatic lane changing
- Time-to-Collision (TTC) calculation
- Emergency stop system
- Visualization and debugging tools

## Requirements

### Hardware
- Camera with ROS driver support
- Robot/Vehicle with ROS interface
- Computer with ROS installation

### Software
- ROS (Robot Operating System)
- Python 3.x
- OpenCV
- NumPy
- Additional ROS packages:
  - geometry_msgs
  - sensor_msgs
  - visualization_msgs
  - cv_bridge
  - std_msgs

## Installation

```bash
# Create a catkin workspace
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src

# Clone the repository
git clone https://github.com/AlpMercan/adas_system.git
git clone https://github.com/aws-robotics/aws-robomaker-racetrack-world

# Install dependencies
sudo apt install ros-noetic-husky-desktop

sudo apt-get update
sudo apt-get install python3-opencv
sudo apt-get install ros-noetic-cv-bridge
sudo apt-get install ros-noetic-vision-opencv
rosdep install --from-paths src --ignore-src -r -y


# Build the workspace
cd ~/catkin_ws
catkin_make

# Source the workspace
source devel/setup.bash
```

## System Components

### 1. Lane Detection Node

#### Features
- Real-time lane boundary detection
- Perspective transformation for bird's-eye view
- Color thresholding for lane marker detection
- Rolling average for stable lane tracking
- ROI optimization

#### Parameters
```yaml
# Lane Detection Parameters
min_pixels: 50
nwindows: 15
margin: 150
minpix: 30
```

#### Subscribed Topics
- `/camera/rgb/image_raw` (sensor_msgs/Image)

#### Published Topics
- `/processed_image` (sensor_msgs/Image)

### 2. Motion Control Node

#### Features
- PID-based steering control
- Adaptive speed control
- Automatic lane changing
- Safety-first approach
- Real-time visualization

#### Parameters
```yaml
# Motion Control Parameters
linear_speed: 1.0
max_angular_speed: 0.5
Kp: 0.005
Ki: 0.0001
Kd: 0.001
max_integral: 50
safety_distance_threshold: 1.0
```

#### Subscribed Topics
- `/processed_image` (sensor_msgs/Image)
- `/person_in_lane` (std_msgs/Bool)
- `/person_distance` (std_msgs/Float32)
- `/ttc` (std_msgs/Float32)

#### Published Topics
- `/cmd_vel` (geometry_msgs/Twist)
- `/detected_object` (visualization_msgs/Marker)
- `/detection_visualization` (sensor_msgs/Image)

## Usage

### Starting the System
1. Launch the ADAS system:
```bash
roslaunch adas_project adas_visualization.launch
```

2. Launch the motion controler:
```bash
rosrun adas_project motion_controller.py
```

### Basic Operations
1. **Lane Following Mode**
   - System automatically follows detected lanes
   - Maintains center position between lane markers
   - Adjusts speed based on curve intensity

2. **Obstacle Avoidance**
   - Automatically detects obstacles
   - Initiates lane change when safe
   - Maintains safe following distance

3. **Emergency Procedures**
   - Automatic emergency stop when critical obstacles detected
   - Graceful degradation when lane markers lost
   - Safety distance maintenance

## Configuration

### Lane Detection Tuning
```yaml
# config/lane_detection.yaml
color_threshold:
  white_lower: [0, 0, 200]
  white_upper: [180, 30, 255]
  yellow_lower: [20, 100, 100]
  yellow_upper: [30, 255, 255]

transform_points:
  src: [[0.35, 0.6], [0.65, 0.6], [0.9, 0.95], [0.1, 0.95]]
  dst: [[0.2, 0], [0.8, 0], [0.8, 1], [0.2, 1]]
```

### Motion Control Tuning
```yaml
# config/motion_control.yaml
pid_params:
  Kp: 0.005
  Ki: 0.0001
  Kd: 0.001

safety_params:
  min_distance: 1.0
  emergency_stop_distance: 0.5
  max_speed: 1.0
```

## Safety Features

### Emergency Stop Conditions
- Obstacle within critical distance
- Lane detection failure
- System malfunction detection
- TTC below threshold

### Failsafe Mechanisms
1. Automatic speed reduction in curves
2. Gradual deceleration on detection loss
3. Redundant obstacle detection
4. System health monitoring

## Debugging and Visualization

### Available Debug Topics
- `/detection_visualization`: Lane detection visualization
- `/path_planning_debug`: Motion planning visualization
- `/safety_zone_visualization`: Safety zone indicators

### Debug Commands
```bash
# View lane detection
rosrun image_view image_view image:=/detection_visualization

# Monitor vehicle status
rostopic echo /vehicle_status

# View TTC values
rostopic echo /ttc
```

## Troubleshooting

### Common Issues and Solutions

1. **Lane Detection Issues**
   - Problem: Poor lane detection
   - Solution: Adjust color thresholds and ROI parameters

2. **Motion Control Issues**
   - Problem: Unstable movement
   - Solution: Fine-tune PID parameters

3. **Safety System Issues**
   - Problem: False emergency stops
   - Solution: Adjust safety thresholds

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
MIT License

## Authors
- Alp Mercan
- [Additional contributors]
