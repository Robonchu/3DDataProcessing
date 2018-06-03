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

_TIME_WINDOW_SIZE = 2
_DIFF_LENGTH = 30
_MINIMUM_RANGE = 300
_MAXIMUM_RANGE = 600

def time_averaging_filter(images):

    mod_images = []
    for frame in range(len(images)-1):
        # diff_image = abs(images[frame+1]-images[frame])
        # mod_image0 = np.where(diff_image <= _DIFF_LENGTH, 0.0, images[frame])
        diff_image = abs(images[frame+1]-images[0])
        mod_image0 = np.where(diff_image <= _DIFF_LENGTH, 0.0, images[0])
        mod_image1 = np.where(diff_image > _DIFF_LENGTH, 0.0, images[frame+1])
        mod_image = mod_image0 + mod_image1
        mod_images.append(mod_image)

    mod_images.append(images[0])
    ave_image = 1.0 * sum(mod_images) / len(mod_images)
    return ave_image.astype(np.float32)

def normal_calculation(image):
    normal_image = np.zeros(image.shape+(3,), dtype=np.float32)
    for x in range(1,image.shape[1]):
        for y in range(1,image.shape[0]):
            t = np.array([x,y-1,image[y-1][x]])
            l = np.array([x-1,y,image[y][x-1]])
            c = np.array([x,y,image[y][x]])
            d = np.cross((l-c), (t-c))
            normal_d = 1.0 * d / np.linalg.norm(d)
            normal_image[y][x] = normal_d
    return normal_image

def edge_search(image):
    edge_image = np.zeros(image.shape[:2], dtype=np.float32)
    for x in range(1,image.shape[1]-1):
        for y in range(1,image.shape[0]-1):
            nw = np.dot(image[y][x], image[y-1][x-1])
            n = np.dot(image[y][x], image[y][x-1])
            ne = np.dot(image[y][x], image[y+1][x-1])
            e = np.dot(image[y][x], image[y+1][x])
            se = np.dot(image[y][x], image[y+1][x+1])
            s = np.dot(image[y][x], image[y][x+1])
            sw = np.dot(image[y][x], image[y-1][x+1])
            w = np.dot(image[y][x], image[y-1][x])
            dot8 = np.array([nw, n, ne, e, se, s, sw, w])
            edge_flag = dot8 < 0.85
            if True in edge_flag:
                edge_image[y][x] = 0
            else:
                edge_image[y][x] = 255
    return edge_image

def closing(image):
    # 8 neighborhood
    neiborhood8 = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]],
                           np.uint8)

    img_dst = cv2.morphologyEx(image, cv2.MORPH_CLOSE, neiborhood8)
    return img_dst


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
        self._edge_image_pub = rospy.Publisher(
            "edge_image", Image, queue_size=10)
        self._segmentation_pcd_pub = rospy.Publisher(
            "segmentation_pcd", PointCloud2)
        depth_camera_info = rospy.wait_for_message(
            "depth_camera_info", CameraInfo)
        self._inv_intrinsic_camera_matrix = np.linalg.inv(np.array(depth_camera_info.K).reshape(3, 3))
        self._ts = message_filters.ApproximateTimeSynchronizer(
            [self._color_image_sub, self._depth_image_sub], 30, 0.5)
        self._ts.registerCallback(self.rgbd_image_callback)
        self._tf_br = tf2_ros.TransformBroadcaster()             
        self._store_image = []
        
    def rgbd_image_callback(self, color, depth):
        try:
            color_image = self._cvbridge.imgmsg_to_cv2(color, 'bgr8')
            depth_image = self._cvbridge.imgmsg_to_cv2(depth, 'passthrough')
        except CvBridgeError as e:
            # TODO: raise exception
            rospy.logerr(e)
            return None

        depth_array = np.array(depth_image, dtype=np.float32)

        # extract 300 ~ 600 [mm] points
        seg_region = np.where((depth_array > _MAXIMUM_RANGE) | (depth_array < _MINIMUM_RANGE), 0.0, depth_array)
        # median filter 3 x 3
        seg_region = cv2.medianBlur(seg_region, ksize=3)
        seg_region = closing(seg_region)
        #seg_region = cv2.medianBlur(seg_region, ksize=5)
        #seg_region = cv2.GaussianBlur(seg_region,(5,5),0)
        
        seg_region = seg_region.astype(np.float32)
    
        self._store_image.append(seg_region)
        if len(self._store_image) < _TIME_WINDOW_SIZE:
            return None
        else:
            self._store_image.pop(0)
            
        # do time averaging 
        seg_region = time_averaging_filter(self._store_image)

        normal_image = normal_calculation(seg_region)
        normal_blur_image = cv2.GaussianBlur(normal_image,(5,5),0)

        edge_region = edge_search(normal_blur_image)
        edge_region = closing(edge_region)
        
        # publish sgementation image for debug
        print "publish edge_region"
        self._edge_image_pub.publish(self._cvbridge.cv2_to_imgmsg(
            edge_region))

        print "publish seg_region"
        self._segmentation_image_pub.publish(self._cvbridge.cv2_to_imgmsg(
            seg_region))
        
        # TODO: speeding up for loop
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
