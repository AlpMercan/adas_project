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
  tf2
  tf2_ros
  gazebo_ros
  visualization_msgs
  image_transport
  message_generation
)

# Find OpenCV
find_package(OpenCV REQUIRED)

add_message_files(
  FILES
  ObjectLocation.msg
)

# Declare custom services if needed
# add_service_files(
#   FILES
#   ObjectDetection.srv
# )

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
  DEPENDS OpenCV
)

# Specify include directories
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
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

# Add executables
# add_executable(${PROJECT_NAME}_node src/main.cpp)
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
#   ${OpenCV_LIBRARIES}
# )
