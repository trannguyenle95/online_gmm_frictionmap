#!/usr/bin/env python3

# documentation: http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29
# tutorial followed: https://youtu.be/MkiWj4VwZjc
import time
import rospy
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import PointCloud2
from jsk_recognition_msgs.msg import ClusterPointIndices
import numpy as np
from scipy.spatial.distance import cdist
from subscriber_node import scene_segmentation
class visualization:
    def __init__(self):
        rospy.init_node('visualization_node')
        self.downsampled_pc_msg = PointCloud2()
        self.segmented_pc_msg = PointCloud2()
        self.ee_position = PointStamped()
        self.haptic_flag = False
        global haptic_sub
        haptic_sub = rospy.Subscriber("/hybrid_controller/position_force", Float64MultiArray, self.haptic_callback)

        global downsampled_sub
        downsampled_sub = rospy.Subscriber("/downsampled_pc_from_main", PointCloud2, self.downsampled_cb_once)
        global downsample_pub
        downsample_pub = rospy.Publisher('/downsampled_pc', PointCloud2, queue_size=1000)
        global segmented_sub
        segmented_sub = rospy.Subscriber("/segmented_pc_from_main", PointCloud2, self.segmented_cb_once)
        global segmented_pub
        segmented_pub = rospy.Publisher('/segmented_pc', PointCloud2, queue_size=1000)
        global haptic_pub
        haptic_pub = rospy.Publisher('/ee_position', PointStamped, queue_size=10)

    def downsampled_cb_once(self, msg):
        self.downsampled_pc_msg = msg
        downsampled_sub.unregister()

    def segmented_cb_once(self, msg):
        self.segmented_pc_msg = msg
        segmented_sub.unregister()

    def publish_clouds(self):
        while not rospy.is_shutdown():
            if not self.haptic_flag and self.downsampled_pc_msg.width and self.segmented_pc_msg.width:
                rate = rospy.Rate(10)
                downsample_pub.publish(self.downsampled_pc_msg) #publish the message
                segmented_pub.publish(self.segmented_pc_msg)
                rate.sleep() #sleep for the amount setted with Rate
            elif self.haptic_flag:
                rate = rospy.Rate(10)
                downsample_pub.publish(self.downsampled_pc_msg) #publish the message
                segmented_pub.publish(self.segmented_pc_msg)
                haptic_pub.publish(self.ee_position)
                rate.sleep() #sleep for the amount setted with Rate

    def haptic_callback(self, msg):
        self.ee_position.header.stamp = rospy.Time.now()
        self.ee_position.header.frame_id = "panda_link0"
        self.ee_position.point.x = msg.data[0]       
        self.ee_position.point.y = msg.data[1]  
        self.ee_position.point.z = msg.data[2]  
        self.haptic_flag = True
        if not len(msg.data):
            self.haptic_flag = False
            haptic_sub.unregister()


if __name__ == '__main__':
    try:
        rospy.loginfo("Started subscriber_node, now listening to messages...")
        vis = visualization() 
        vis.publish_clouds()
        rospy.spin()

        # supervoxels, downsampled_pc_points, downsampled_pc_rgb = segment_scene_once()
        # supervoxels_points, supervoxels_rgb = reorganize_points_supervoxel(supervoxels, downsampled_pc_points, downsampled_pc_rgb)
        # supervoxels_mean_rgb = []
        # test_point = np.array([-0.0668195 ,  0.0413865 ,  0.011787])
        # start_time = time.time()
        # closest_supervoxel, idx_closest_supervoxel = closest_array_to_target_cdist(test_point,supervoxels_points)
        # print("Array containing the closest point to", test_point, "is", closest_supervoxel, "idx: ",idx_closest_supervoxel)
        # print("--- %s miliseconds ---" % ((time.time() - start_time)*1000))

        # for i in supervoxels_rgb:
        #     supervoxels_mean_rgb.append(np.mean(i,axis=0))
    except rospy.ROSInterruptException:
        pass