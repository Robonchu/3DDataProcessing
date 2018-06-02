# Overview

## Provided Function

- model free segmentaion using only depth image

## Reference

- Real-Time 3D Segmentation of Cluttered Scenes for Robot Grasping


## Subscribed Topics

- **color_image** (`sensor_msgs.msg/Image`)

- **depth_image** (`sensor_msgs.msg/Image`)

- **depth_camera_info** (`sensor_msgs.msg/Image`)

## Published Topics

- **segmentation_image** (`sensor_msgs.msg/Image`)

## How to use

1. roslaunch depth_segmentation segmentation_object.launch
