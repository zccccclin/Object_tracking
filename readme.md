# Object tracking
This repo explores object pose tracking through 3D-3D correspondence/2D-3D correspondence between image frames.

# Set up
Clone the repo into ros package and build.
```
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin_make
cd /src
git clone ssh://git@stash.dyson.global.corp:7999/~zhiclin/object_tracking.git
catkin_make
source /devel/setup.bash
```

# Config.yaml
config files can be edited n /config folder

# Running on live RGBD video feed
Launch message publisher: `roslaunch object_tracking live_track.launch`

To run matching based pose estimator: `rosrun object_tracking matcher_node.py`

To run optical flow tracker based pose estimator: `rosrun object_tracking tracker_node.py`

# Running on recorded ROSBAG
First launch message publisher: `roslaunch object_tracking live_track.launch`
Record the ROSBAG:
`rosbag record /aruco_single/result /aruco_single/pose /camera/color/camera_info /camera/aligned_depth_to_color/camera_info /camera/color/image_raw /camera/aligned_depth_to_color/image_raw`

#### After ROSBAG is saved:
1. Run Rviz viewer: `roslaunch object_tracking rosbag_track.launch`
2. Run ROSBAG recording: `rosbag play rosbag_recording_file_name.bag`