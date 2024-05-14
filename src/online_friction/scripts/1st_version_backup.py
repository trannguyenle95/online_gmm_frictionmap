
# listen function is executed in a permanent way till the node is stopped
def segment_scene_once():
    rospy.init_node('segment_scene_node', anonymous=True)
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    launch = roslaunch.parent.ROSLaunchParent(uuid, ["/online_friction/src/jsk_pcl_ros_samples/launch/supervoxel_segmentation.launch"])
    launch.start()
    rospy.loginfo("started")
    # subscribe to the topic, specifing the data (String) and a callback function for when is received
    # rospy.Subscriber("/points", PointCloud2, callback)
    enter = input("Press enter to segment scene")
    if enter=='':
        data = rospy.wait_for_message("/supervoxel_segmentation/output/cloud", PointCloud2, timeout=10)
        indices_data = rospy.wait_for_message("/supervoxel_segmentation/output/indices", ClusterPointIndices, timeout=10)
        # Getting supervoxels and indices of point in that supervoxel
        supervoxels = extract_indices_from_cluster(indices_data) 
        # Getting xyz and rgb of points
        data = ros_numpy.numpify(data)
        data = ros_numpy.point_cloud2.split_rgb_field(data)
        downsampled_pc_points=np.zeros((data.shape[0],3))
        downsampled_pc_points[:,0]=data['x']
        downsampled_pc_points[:,1]=data['y']
        downsampled_pc_points[:,2]=data['z']
        print(downsampled_pc_points.shape)
        downsampled_pc_rgb=np.zeros((data.shape[0],3))
        downsampled_pc_rgb[:,0]=data['r']
        downsampled_pc_rgb[:,1]=data['g']
        downsampled_pc_rgb[:,2]=data['b']
        print(downsampled_pc_rgb.shape)
        print(supervoxels)
        # time.sleep(2)
        # rosnode.kill_nodes(['rviz'])
        # launch.shutdown()

    else:
        exit()
    # launch.shutdown()
    return supervoxels, downsampled_pc_points, downsampled_pc_rgb