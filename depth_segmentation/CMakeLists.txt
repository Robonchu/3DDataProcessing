cmake_minimum_required(VERSION 2.8.3)
project(depth_segmentation)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  message_filters
  rospy
  sensor_msgs
  std_msgs
  tf2_ros
)

catkin_python_setup()

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES depth_segmentation
#  CATKIN_DEPENDS cv_bridge geometry_msgs rospy
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

