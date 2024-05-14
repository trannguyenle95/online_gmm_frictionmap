## Run the following commands in correct order for online estimation with recorded bag files:
- roscore
  # For point cloud transform, filter, and supervoxel segmentation.
- roslaunch online_friction supervoxel_segmentation.launch
- python3 visualization.py
- run online_friction subscriber_node.py #The online gmm model is save to res.data
- After this, you can kill off 1 - supervoxel_segmentation.launch.
  # For visualizing segmented point cloud + ee_pose
- rviz -d src/online_friction/rviz/publish.rviz 

## Plots GMM
- python3 /src/online_friction/scripts/streaming_gmm/plotting_gmms.py # To plot the gmm model in saved res.data
