#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import geometry_msgs.msg
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

def segmentation(image):
    # TBD
    # extract 300 ~ 600 [mm] points
    seg_region = np.where((image > 600) | (image < 300), 0.0, image)
    return seg_region.astype(np.float32)

class SegmentationObject:
    u"""segmentation object"""

    def __init__(self):
        u"""initialize"""
        self._cvbridge = CvBridge()
        self._color_image_sub = message_filters.Subscriber(
            "color_image", Image)
        self._depth_image_sub = message_filters.Subscriber(
            "depth_image", Image)
        self._segmentation_image_pub = rospy.Publisher(
            "segmentation_image", Image, queue_size=10)
        self._segmentation_pcd_pub = rospy.Publisher(
            "segmentation_pcd", PointCloud2)
        depth_camera_info = rospy.wait_for_message(
            "depth_camera_info", CameraInfo)
        self._inv_intrinsic_camera_matrix = np.linalg.inv(np.array(depth_camera_info.K).reshape(3, 3))
        self._ts = message_filters.ApproximateTimeSynchronizer(
            [self._color_image_sub, self._depth_image_sub], 30, 0.5)
        self._ts.registerCallback(self.rgbd_image_callback)
        self._tf_br = tf2_ros.TransformBroadcaster()             

    def rgbd_image_callback(self, color, depth):
        try:
            color_image = self._cvbridge.imgmsg_to_cv2(color, 'bgr8')
            depth_image = self._cvbridge.imgmsg_to_cv2(depth, 'passthrough')
        except CvBridgeError as e:
            # TODO: raise exception
            rospy.logerr(e)
            return None

        depth_array = np.array(depth_image, dtype=np.float32)
        seg_region = segmentation(depth_array)
        self._segmentation_image_pub.publish(self._cvbridge.cv2_to_imgmsg(
            seg_region))

        object_points = []
        for x in range(seg_region.shape[1]):
            for y in range(seg_region.shape[0]):
                if seg_region[y,x] :
                    seg_image_point = np.array([x, y, 1])
                    seg_depth_point = seg_region[y][x] * 1e-3 # depth_array[y][x] * 1e-3
                    if seg_depth_point == 0:
                        rospy.logerr("invalid depth data")
                        return None
                    object_point = np.dot(
                        self._inv_intrinsic_camera_matrix, seg_image_point) * seg_depth_point
                    object_points.append(object_point)

        pc_header = color.header
        pc_header.stamp = rospy.Time.now()
        object_pcd = pc2.create_cloud_xyz32(pc_header, object_points)
        self._segmentation_pcd_pub.publish(object_pcd)        


if __name__ == "__main__":
    rospy.init_node('segmentation_object')
    segmentation_object = SegmentationObject()
    rospy.spin()
