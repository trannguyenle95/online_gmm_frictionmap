#!/usr/bin/env python3

# documentation: http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29
# tutorial followed: https://youtu.be/MkiWj4VwZjc

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
import ros_numpy
import numpy as np

def callback(data):
    # callback function need as a parameter, the data object
    #rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

    # data = ros_numpy.numpify(data)
    data = ros_numpy.point_cloud2.split_rgb_field(data)
    points=np.zeros((data.shape[0],3))
    points[:,0]=data['x']
    points[:,1]=data['y']
    points[:,2]=data['z']
    print(points.shape)
    rgb=np.zeros((data.shape[0],3))
    rgb[:,0]=data['r']
    rgb[:,1]=data['g']
    rgb[:,2]=data['b']
    print(rgb.shape)

    #data.data access the String msg received!

def test(rgb):
    print("this is: ", rgb)
# listen function is executed in a permanent way till the node is stopped
def listen():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener_node', anonymous=True)

    # subscribe to the topic, specifing the data (String) and a callback function for when is received
    rospy.Subscriber("/points", PointCloud2, callback)
       
    # spin() simply keeps python from exiting until this node is stopped
    # rospy.spin()

if __name__ == '__main__':
    try:
        rospy.loginfo("Started subscriber_node, now listening to messages...")
        points, rgb = listen()
        test(rgb)
    except rospy.ROSInterruptException:
        pass