cmake_minimum_required(VERSION 3.0.2)
project(adas_project)

# Find required packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  cv_bridge
  image_transport
  tf2
  tf2_ros
  gazebo_ros
  visualization_msgs
  message_generation
  message_filters
)

# Find OpenCV
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(cv_bridge REQUIRED)


add_message_files(
  FILES
  ObjectLocation.msg
)

# Generate messages
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

# Specify catkin package settings
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
    roscpp
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    cv_bridge
    tf2
    tf2_ros
    gazebo_ros
    visualization_msgs
    message_runtime
  DEPENDS OpenCV EIGEN3
)

# Specify include directories
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# Create the package directory structure
execute_process(
  COMMAND ${CMAKE_COMMAND} -E make_directory
  ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/scripts
  ${CMAKE_CURRENT_SOURCE_DIR}/launch
  ${CMAKE_CURRENT_SOURCE_DIR}/config
  ${CMAKE_CURRENT_SOURCE_DIR}/worlds
  ${CMAKE_CURRENT_SOURCE_DIR}/msg
  ${CMAKE_CURRENT_SOURCE_DIR}/urdf
)

# Add your nodes as executables
add_executable(lane_detector_node src/Lane_Detector.cpp)
add_executable(motion_controller_node 
    src/MotionController.cpp
    src/Simple_LaneChanger.cpp
)
add_executable(ttc_calculator_node 
    src/TTCCalculator.cpp
)
# Add dependencies for message generation
add_dependencies(lane_detector_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(motion_controller_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(ttc_calculator_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# Link libraries for each executable
target_link_libraries(lane_detector_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

target_link_libraries(motion_controller_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

target_link_libraries(ttc_calculator_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${EIGEN3_LIBRARIES}
)
