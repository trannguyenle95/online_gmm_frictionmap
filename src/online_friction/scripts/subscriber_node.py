#!/usr/bin/env python3

# documentation: http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29
# tutorial followed: https://youtu.be/MkiWj4VwZjc
import time
import rospy
import pickle
from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import PointCloud2
from jsk_recognition_msgs.msg import ClusterPointIndices
import numpy as np
np.float = np.float64 
import ros_numpy
from scipy.spatial.distance import cdist
# from streaming_gmm import plotting_gmms
from streaming_gmm.streaming_variational_gmm import StreamingVariationalGMM, VariationalGMM
from streaming_gmm.gmr import *
import logging
import matplotlib.pyplot as plt

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
    
class scene_segmentation:
    def __init__(self):
        rospy.init_node('segment_scene_node')
        K = 2 # Number of components per class.
        D = 4 # Dimension of data.
        self.new_data = None
        self.result_list_gmm = []
        self.streaming_vb_gmm = StreamingVariationalGMM(K, D, max_iter=50, alpha_0=.01)
        self.measured_friction_mean = [] # Mean friction coeff of supervoxels when measured
        self.measured_friction_std = [] # Std friction coeff of supervoxels when measured
        self.estimated_friction_mean = [] #
        self.estimated_friction_std = [] #
        self.temp_friction_mean = None
        self.gmm_weights = []
        self.sigma = []


        self.supervoxels = []
        self.downsampled_pc_points = []
        self.downsampled_pc_rgb = []
        self.supervoxels_points = []
        self.supervoxels_rgb_mean = []
        self.downsampled_pc_msg = []
        self.segmented_pc_msg = []
        self.segmented_debug_msg = []
        global downsample_sub
        downsample_sub = rospy.Subscriber('/supervoxel_segmentation/output/cloud', PointCloud2, self.downsample_cb_once)
        global segmented_sub
        segmented_sub = rospy.Subscriber('/supervoxel_segmentation/output/indices', ClusterPointIndices, self.segmented_cb_once)
        global segmented_debug_output_sub
        segmented_debug_output_sub = rospy.Subscriber('/cluster_point_indices_decomposer/debug_output', PointCloud2, self.segmented_debug_cb_once)
        global downsample_pub
        downsample_pub = rospy.Publisher("/downsampled_pc_from_main", PointCloud2, queue_size=1000)
        global segmented_debug_pub
        segmented_debug_pub = rospy.Publisher("/segmented_pc_from_main", PointCloud2, queue_size=1000)
        global position_force_sub
        position_force_sub = rospy.Subscriber("/hybrid_controller/position_force", Float64MultiArray, self.haptic_matching_callback)
        self.count = 0
        self.all_done = False
    def get_downsampled_data(self):
        time.sleep(1) # Give some time for processing
        return self.downsampled_pc_points, self.downsampled_pc_rgb

    def get_segmented_data(self):
        # time.sleep(1)
        supervoxels_points, supervoxels_rgb = self.reorganize_points_supervoxel(self.supervoxels, self.downsampled_pc_points, self.downsampled_pc_rgb)
        self.supervoxel_points = supervoxels_points
        self.supervoxels_rgb_mean = np.asarray([np.mean(sublist,axis=0) for sublist in supervoxels_rgb])
        return self.supervoxels, self.supervoxel_points, self.supervoxels_rgb_mean

    def extract_indices_from_cluster(self, indices_data):
        '''
        Extracts supervoxels (indices of points in supervoxels) from ClusterPointIndices msg.

        Args:
            indices_data (ClusterPointIndices): The ClusterPointIndices message containing cluster indices.

        Returns:
            list: A list of supervoxels, where each supervoxel contains indices of points.
        '''
        cluster_indices = indices_data.cluster_indices
        supervoxels = []
        for element in cluster_indices:
            supervoxels.append(np.asarray(element.indices))      
        print("We have ",len(supervoxels), " supervoxels.")
        self.temp_friction_mean  = [[] for _ in range(len(supervoxels))]
        self.measured_friction_mean = np.zeros(len(supervoxels)) 
        self.measured_friction_std = np.zeros(len(supervoxels)) 
        self.supervoxels = supervoxels
        # Init friction vectors
        return self.supervoxels

    def reorganize_points_supervoxel(self, supervoxels, downsampled_pc_points, downsampled_pc_rgb):
        '''
        Organize original xyz and rgb points into the segmented supervoxels.
        
        Args:
            supervoxels (list): List of supervoxels.
            downsampled_pc_points (list): List of downsampled point cloud xyz points.
            downsampled_pc_rgb (list): List of downsampled point cloud rgb values.
        
        Returns:
            tuple: A tuple containing two lists - supervoxels_points and supervoxels_rgb.
                - supervoxels_points (list): List of xyz points for each supervoxel.
                - supervoxels_rgb (list): List of rgb values for each supervoxel.
        '''
        supervoxels_points = []
        supervoxels_rgb = []
        for supervoxel in supervoxels:
            supervoxel_points = [downsampled_pc_points[i] for i in supervoxel]
            supervoxel_rgb = [downsampled_pc_rgb[i] for i in supervoxel]

            supervoxel_points = np.stack(supervoxel_points, axis=0)
            supervoxel_rgb = np.stack(supervoxel_rgb, axis=0)

            supervoxels_points.append(supervoxel_points)
            supervoxels_rgb.append(supervoxel_rgb)
        return supervoxels_points, supervoxels_rgb

    def check_closest_supervoxel(self, target, arrays):
        """
        Find the closest supervoxel arrays to a given target.

        Parameters:
        - target: The target point array to compare with.
        - arrays: A list of supervoxel arrays to search through.

        Returns:
        - closest_array: The closest supervoxel array to the target.
        - closest_array_idx: The index of the closest array in the list.
        """
        target_arr = np.array(target)
        # Initialize variables to track the minimum distance and closest array
        min_distance = float('inf')
        closest_array = None
        for idx, arr in enumerate(arrays):
            if len(arr) == 0:
                continue
            # Calculate pairwise distances between target and all points in the current array
            distances = cdist(np.atleast_2d(target_arr), arr)
            # Find the minimum distance in the current array
            min_dist_in_array = np.min(distances)
            # Update closest array if the minimum distance is smaller
            if min_dist_in_array < min_distance:
                min_distance = min_dist_in_array
                closest_array = arr
                closest_array_idx = idx
        return closest_array, closest_array_idx
    
    
    
    def process_measured_friction(self,ee_force,closest_supervoxel_idx):
        muy = abs(ee_force[0]/ee_force[1])
        self.temp_friction_mean[closest_supervoxel_idx].append(muy)
        self.measured_friction_mean = np.asarray([np.mean(sublist) for sublist in self.temp_friction_mean])
        self.measured_friction_std = np.asarray([np.std(sublist) for sublist in self.temp_friction_mean])
        # print("------")
        # print("idx: ", closest_supervoxel_idx, "- Measured friction mean: ", self.measured_friction_mean)
        # print("idx: ", closest_supervoxel_idx, "- Measured friction std: ", self.measured_friction_std)
        # print("------")
        return self.measured_friction_mean, self.measured_friction_std

    def downsample_cb_once(self, msg):
        ori_msg = msg
        msg = ros_numpy.numpify(msg)
        msg = ros_numpy.point_cloud2.split_rgb_field(msg)
        downsampled_pc_points=np.zeros((msg.shape[0],3))
        downsampled_pc_points[:,0]=msg['x']
        downsampled_pc_points[:,1]=msg['y']
        downsampled_pc_points[:,2]=msg['z']
        print(downsampled_pc_points.shape)
        downsampled_pc_rgb=np.zeros((msg.shape[0],3))
        downsampled_pc_rgb[:,0]=msg['r']
        downsampled_pc_rgb[:,1]=msg['g']
        downsampled_pc_rgb[:,2]=msg['b']
        print(downsampled_pc_rgb.shape)
        self.downsampled_pc_points = downsampled_pc_points
        self.downsampled_pc_rgb = downsampled_pc_rgb
        self.downsampled_pc_msg = ori_msg
        downsample_sub.unregister()

    def segmented_cb_once(self, msg):
        self.segmented_pc_msg = msg
        self.supervoxels = self.extract_indices_from_cluster(msg) 
        segmented_sub.unregister()

    def segmented_debug_cb_once(self, msg):   
        self.segmented_debug_msg = msg
        segmented_debug_output_sub.unregister()

    def publish_clouds(self):
        '''
        This method continuously publishes point clouds to the visualization.py node until it receives connections
        on both the downsampled and segmented topics. It logs the number of connections and stops publishing once
        there are connections on both topics.

        Returns:
            None
        '''
        while not rospy.is_shutdown(): 
            rate = rospy.Rate(1)
            downsample_pub.publish(self.downsampled_pc_msg)
            segmented_debug_pub.publish(self.segmented_debug_msg)
            connections_downsampled = downsample_pub.get_num_connections()
            connections_segmented = segmented_debug_pub.get_num_connections()
            rospy.loginfo('Connections to downsampled pc topic: %d', connections_downsampled)
            rospy.loginfo('Connections to segmented pc topic: %d', connections_segmented)
            if connections_downsampled > 0 and connections_segmented > 0:
                rospy.loginfo('Published clouds')
                downsample_pub.unregister()
                segmented_debug_pub.unregister()
                break
            rate.sleep() 

    def haptic_matching_callback(self, msg):
        self.count += 1
        if self.count >= 3000: #TODO: Change this to a better way to unsubscribe the subscriber otherwise it will keep running. Maybe when force_z is above 0?
            position_force_sub.unregister()
            self.show_result()
        else:
            ee_position = np.array([msg.data[0],msg.data[1],msg.data[2]- 0.02])
            ee_force = np.array([msg.data[3],msg.data[4]])
            closest_supervoxel, closest_supervoxel_idx = self.check_closest_supervoxel(ee_position, self.supervoxel_points)
            measured_friction_mean, measured_friction_std = self.process_measured_friction(ee_force,closest_supervoxel_idx)

            new_data = np.append(self.supervoxels_rgb_mean[closest_supervoxel_idx],measured_friction_mean[closest_supervoxel_idx])
            new_data = np.reshape(new_data, (1, 4))
            if self.count % 10 == 0:
                self.streaming_vb_gmm.update_with_new_data(new_data)
                self.result_list_gmm = self.streaming_vb_gmm.checkpoints
                # print("Iter: ", len(self.result_list_gmm))
                variational_params = self.result_list_gmm[-1]['streaming_variational_parameters']
                self.sigma = np.linalg.inv(variational_params['nu_k'][:, np.newaxis, np.newaxis] * variational_params['W_k'])
                gmm_weights = np.sum(variational_params['alpha_k'])
                self.gmm_weights = variational_params['alpha_k'] / gmm_weights
                # print("sigma: ", sigma.shape, "mean: ", variational_params['m_k'].shape, "weight: ", gmm_weights.shape ) # Only for debugging gmr
                print("Count: ", self.count, "mean_now: ", variational_params['m_k'], "weight: ", self.gmm_weights)
                res_mean = predict(indices=np.array([0, 1, 2]), X = self.supervoxels_rgb_mean , means = variational_params['m_k'] , covariances = self.sigma, priors = self.gmm_weights, random_state=check_random_state(0))
                print("res_mean: ", res_mean.shape, res_mean)       
    def get_done_status(self):
        return self.all_done
    
    def show_result(self):
        self.count = 0
        print("Done")
        unexplored_supervoxels = []
        for idx, value in enumerate(np.isnan(self.measured_friction_mean)):
            if value:
                unexplored_supervoxels.append(self.supervoxels_rgb_mean[idx])
            else:
                pass
        unexplored_supervoxels = np.asarray(unexplored_supervoxels)   
        print("unexplored_supervoxels: ", unexplored_supervoxels.shape)             
        variational_params = self.result_list_gmm[-1]['streaming_variational_parameters']
        m_k = variational_params['m_k']
        nu_k = variational_params['nu_k']
        W_k = variational_params['W_k']
        data = [self.supervoxels_rgb_mean,m_k,self.gmm_weights,self.sigma]
        with open('res.data', 'wb') as f:
            pickle.dump(data, f)
        # plotting_functions.plot_gmm(unexplored_supervoxels, 
        #             m_k, 
        #             np.linalg.inv(nu_k[:, np.newaxis, np.newaxis] * W_k))
        # return fig

if __name__ == '__main__':
    try:
        # create logger with 'spam_application'
        logger = logging.getLogger("Online_friction")
        logger.setLevel(logging.DEBUG)

        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(CustomFormatter())
        logger.addHandler(ch)
        
        rospy.loginfo("Started subscriber_node, now listening to messages...")
        enter = input("Press enter to segment scene")
        scene_segmentation = scene_segmentation()
        if enter=='':
            start_time = time.time()
            downsampled_points,downsampled_rgb = scene_segmentation.get_downsampled_data()
            print("downsampled_xyz: ",len(downsampled_points),"downsampled_rgb: ", len(downsampled_rgb))
            supervoxels,supervoxels_points,supervoxels_rgb= scene_segmentation.get_segmented_data()
            print("supervoxels: ",len(supervoxels),"supervoxels_xyz: ", len(supervoxels_points),"supervoxels_rgb: ",len(supervoxels_rgb))
            scene_segmentation.publish_clouds()
            logger.info("Scene is segmented in %s seconds ---", ((time.time() - start_time)))
        else:
            exit()
        print("Please run the exploration now for online friction estimation.")
        rospy.spin()

        # supervoxels, downsampled_pc_points, downsampled_pc_rgb = segment_scene_once()
        # supervoxels_points, supervoxels_rgb = reorganize_points_supervoxel(supervoxels, downsampled_pc_points, downsampled_pc_rgb)
        # supervoxels_mean_rgb = []
        # test_point = np.array([-0.0668195 ,  0.0413865 ,  0.011787])
        # 
        # closest_supervoxel, idx_closest_supervoxel = closest_array_to_target_cdist(test_point,supervoxels_points)
        # print("Array containing the closest point to", test_point, "is", closest_supervoxel, "idx: ",idx_closest_supervoxel)
        # print("--- %s miliseconds ---" % ((time.time() - start_time)*1000))

        # for i in supervoxels_rgb:
        #     supervoxels_mean_rgb.append(np.mean(i,axis=0))
    except rospy.ROSInterruptException:
        pass