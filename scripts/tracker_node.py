#!/usr/bin/env python
import rospy
import csv
import cv2
import os
import numpy as np
import time
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from sensor_msgs.msg import PointCloud2, CameraInfo, Image
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_msgs.msg import Float32
import message_filters
import tf
from func import *
from math import pi

class KPTracker:
    def __init__(self):
        # get params
        correction = rospy.get_param("/correction",False)
        logging = rospy.get_param("/log_data",False)
        tracker_use_orb = rospy.get_param('/tracker_use_orb',True)
        correction_interval = rospy.get_param('/correction_interval',10)
        update_interval = rospy.get_param('/update_interval',5)
        print(f"Correction: {correction}, Correction interval: {correction_interval}, Loggging: {logging}, Use ORB: {tracker_use_orb}, Update interval: {update_interval}.")

        # subscribers
        self.colorInfo = rospy.wait_for_message('camera/color/camera_info', CameraInfo)
        self.depthInfo = rospy.wait_for_message('/camera/aligned_depth_to_color/camera_info', CameraInfo)
        self.arucosub = rospy.Subscriber('/aruco_single/pose', PoseStamped,self.aruco_cb)

        self.colorsub = message_filters.Subscriber('/camera/color/image_raw',Image)
        self.depthsub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        self.ts = message_filters.TimeSynchronizer([self.colorsub,self.depthsub],10,1)
        self.ts.registerCallback(self.callback)
        # publishers
        self.svdposepub = rospy.Publisher('/SVDPose',PoseStamped,queue_size=10)
        self.pnpposepub = rospy.Publisher('/PNPPose',PoseStamped,queue_size=10)
        self.pointpub = rospy.Publisher('points2', PointCloud2, queue_size=100)
        self.imagepub = rospy.Publisher('/node_img', Image, queue_size=10)
        self.svdxyzerrorpub = rospy.Publisher('svd_dist_error', Float32, queue_size=10)
        self.pnpxyzerrorpub = rospy.Publisher('pnp_dist_error', Float32, queue_size=10)
        self.svdrpyerrorpub = rospy.Publisher('svd_ang_error', Float32, queue_size=10)
        self.pnprpyerrorpub = rospy.Publisher('pnp_ang_error', Float32, queue_size=10)

        # Camera K matrices:
        self.C_K = np.asarray(self.colorInfo.K).reshape(3,3)
        self.D_K = np.asarray(self.depthInfo.K).reshape(3,3)
        self.D_width = self.depthInfo.width
        self.D_height = self.depthInfo.height
        self.ref_frame = self.colorInfo.header.frame_id
        # CV initialization
        self.bridge = CvBridge()

        # Img variable 
        self.curr_frame = None
        self.prev_frame = None
        self.curr_depth = None
        self.prev_depth = None
        self.fps = 0
        self.initialized = False

        # Improvements
        self.correction = correction

        # Logging
        self.logging = logging
        path = os.path.dirname(os.path.abspath(__file__))
        path = path + '/../log/log.csv'
        header = ['SVD_X_e','SVD_Y_e',"SVD_Z_e","SVD_dist_e",'PNP_X_e','PNP_Y_e',"PNP_Z_e","PNP_dist_e","SVD_Roll_e","SVD_Pitch_e","SVD_Yaw_e","SVD_Mean_e","PNP_Roll_e","PNP_Pitch_e","PNP_Yaw_e","PNP_Mean_e"]
        self.f = open(path, "w")
        self.c = csv.writer(self.f)
        if self.logging:
            self.c.writerow(header)

        # Detector and tracker param
        self.use_orb = tracker_use_orb
        self.orb = cv2.ORB_create()
        self.frame_idx = 0
        self.trajectory_len = 10
        self.update_interval = update_interval
        self.correction_interval = correction_interval
        self.trajectories = []
        self.lk_params = dict(winSize  = (15, 15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict(maxCorners = 20,
                    qualityLevel = 0.5,
                    minDistance = 5,
                    blockSize = 7 )
    


    def aruco_cb(self,msg):
        self.gt_homo = pose_to_homo(msg)
        if not self.initialized:
            print("Initializing Ground Truth Pose")
            self.svdhomo = self.gt_homo
            self.pnphomo = self.gt_homo
            self.initialized = True
            print('changed')
        self.gt_position = np.array([msg.pose.position.x,msg.pose.position.y,msg.pose.position.z])
        self.gt_euler = np.array(tf.transformations.euler_from_quaternion(np.array([msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z,msg.pose.orientation.w])))

    def callback(self,color,depth):
        start = time.time()

        #------------------------------------ START of 2D Tracking-----------------------------------------
        #color msg -> image processing
        frame = self.bridge.imgmsg_to_cv2(color)
        img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        
        # Tracking if feature points detected
        if len(self.trajectories) > 0:
            # Cross check to get good tracks
            img0, img1 = self.prev_frame, self.curr_frame
            p0 = np.float32([trajectory[-1] for trajectory in self.trajectories]).reshape(-1, 1, 2)
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
            p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1

            # Update track
            new_trajectories = []
            self.prev_idx = []
            self.curr_idx = []
            for trajectory, (x, y), good_flag in zip(self.trajectories, p1.reshape(-1, 2), zip(good,_st.flatten())):
                if not (good_flag[0] and bool(good_flag[1])):
                    continue
                if x > self.D_width or y > self.D_height:
                    rospy.logwarn('Estimated track out of frame')
                    continue
                self.prev_idx.append(trajectory[-1]) # Get last tracked point pixel
                self.curr_idx.append((x,y))
                trajectory.append((x, y))
                if len(trajectory) > self.trajectory_len:
                    del trajectory[0]
                new_trajectories.append(trajectory)
                cv2.circle(img, (int(x), int(y)), 3, (255, 255, 0), 1)
            self.trajectories = new_trajectories
            self.prev_idx = np.array(self.prev_idx,dtype=np.int16)
            self.curr_idx = np.array(self.curr_idx,dtype=np.int16)

        # Detect new feature points to track for every interval
        if self.frame_idx % self.update_interval == 0:
            # Mask to mask out detected points
            self.mask = np.zeros_like(self.curr_frame)
            self.mask[:] = 255
            for x, y in [np.int32(trajectory[-1]) for trajectory in self.trajectories]:
                cv2.circle(self.mask, (x, y), 3, 0, -1)

            # Detect the good features to track
            p = cv2.goodFeaturesToTrack(self.curr_frame, mask = self.mask, **self.feature_params)
            if self.use_orb:
                p = self.orb.detect(self.curr_frame,mask=self.mask) 
                p = [[(kp.pt[0], kp.pt[1])] for kp in p]
                p = np.array(p)
            if p is not None:
                # If good features can be tracked - add that to the trajectories
                for x, y in np.float32(p).reshape(-1, 2):
                    self.trajectories.append([(x, y)])

                    
        #------------------------------------ END of 2D processing-----------------------------------------

        #------------------------------------ START of 3D processing-----------------------------------------
        # depth msg -> image processing
        depth_img = self.bridge.imgmsg_to_cv2(depth,'passthrough')
        self.curr_depth = depth_img
        svdRT = tf.transformations.identity_matrix()
        pnpRT = tf.transformations.identity_matrix()

        if self.prev_depth is not None and self.initialized:
            # Get 3D points and remove invalid 2D KP idx (no depth):
            self.prev_3d, self.curr_3d, self.prev_idx, self.curr_idx = pixelto3d(self.D_K, 
                                                                self.prev_idx, self.curr_idx, 
                                                                self.prev_depth, self.curr_depth)
            # Generate 3D KP pointcloud
            pc = xyz2pc(self.curr_3d)

            # Get R, T through PNP and SVD, if insufficient points, pose @ identity matrix (no transform)
            if len(self.curr_3d) > 6:
                pnpRT = solve_by_pnp(self.prev_3d,self.curr_idx,self.D_K)
                svdRT = solve_by_svd(self.prev_3d, self.curr_3d)
                #print(f"PNP solution: {pnpRT}")
                #print(f"SVD solution: {svdRT}")
            else: rospy.logwarn("Insufficient points for RT estimation, null transformation")

            if self.correction and self.frame_idx % self.correction_interval == 0:
                self.svdhomo = self.gt_homo
                self.pnphomo = self.gt_homo

            # Estimate new pose
            self.svdhomo = svdRT @ self.svdhomo  
            self.pnphomo = pnpRT @ self.pnphomo
            self.svdpose = homo_to_pose(self.svdhomo,self.ref_frame)
            self.pnppose = homo_to_pose(self.pnphomo,self.ref_frame)

        #------------------------------------ END of 3D processing-----------------------------------------
            # Euclidean dist error and rpy ang error
            svderror = dist3d_diff(self.svdhomo[:3,3],self.gt_position) * 100
            pnperror = dist3d_diff(self.pnphomo[:3,3],self.gt_position) * 100
            svd_xyz_error = (self.svdhomo[:3,3]-self.gt_position)*100
            pnp_xyz_error = (self.pnphomo[:3,3]-self.gt_position)*100
            svd_rpy = np.array(tf.transformations.euler_from_quaternion(tf.transformations.quaternion_from_matrix(self.svdhomo)))
            pnp_rpy = np.array(tf.transformations.euler_from_quaternion(tf.transformations.quaternion_from_matrix(self.pnphomo)))
            svd_rpy_error = ang_limit((abs(svd_rpy-self.gt_euler))*180/pi)
            pnp_rpy_error = ang_limit((abs(pnp_rpy-self.gt_euler))*180/pi)

            print(f"svd xyz: {self.svdhomo[:3,3].round(3)},  xyz_error: {svd_xyz_error.round(3)},  dist_error = {round(svderror,3)} cm")
            print(f"pnp xyz: {self.pnphomo[:3,3].round(3)},  xyz_error: {pnp_xyz_error.round(3)},  dist_error = {round(pnperror,3)} cm")
            print(f"gt xyz:  {self.gt_position.round(4)}")
            print(f"svd ang: {(svd_rpy*180/pi).round(1)},  ang_error: {svd_rpy_error.round(1)},  mean_error = {round(svd_rpy_error.mean(),1)} deg")
            print(f"pnp ang: {(pnp_rpy*180/pi).round(1)},  ang_error: {pnp_rpy_error.round(1)},  mean_error = {round(pnp_rpy_error.mean(),1)} deg")
            print(f"gt ang:  {(self.gt_euler*180/pi).round(1)}")
            
            svd_disterror = Float32()
            pnp_disterror = Float32()
            svd_angerror = Float32()
            pnp_angerror = Float32()
            svd_disterror.data = round(svderror*100,3)
            pnp_disterror.data = round(pnperror*100,3)
            svd_angerror.data = round((svd_rpy-self.gt_euler).mean(),3)
            pnp_angerror.data = round((pnp_rpy-self.gt_euler).mean(),3)

            # Draw all the trajectories
            cv2.polylines(img, [np.int32(trajectory) for trajectory in self.trajectories], False, (0, 255, 0))
            cv2.putText(img, 'track count: %d' % len(self.trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)
            cv2.putText(img, f"{self.fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f'No. 3DKP: {len(self.curr_3d)}', (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            


            # Publishers
            self.imagepub.publish(self.bridge.cv2_to_imgmsg(img))
            self.pointpub.publish(pc)
            self.svdposepub.publish(self.svdpose)
            self.pnpposepub.publish(self.pnppose)
            self.svdxyzerrorpub.publish(svd_disterror)
            self.pnpxyzerrorpub.publish(pnp_disterror)
            self.svdrpyerrorpub.publish(svd_angerror)
            self.pnprpyerrorpub.publish(pnp_angerror)


            # Logging
            if self.logging:
                data = [svd_xyz_error[0],svd_xyz_error[1],svd_xyz_error[2],svderror,pnp_xyz_error[0],pnp_xyz_error[1],pnp_xyz_error[2],pnperror,svd_rpy_error[0],svd_rpy_error[1],svd_rpy_error[2],svd_rpy_error.mean(),pnp_rpy_error[0],pnp_rpy_error[1],pnp_rpy_error[2],pnp_rpy_error.mean()]
                self.c.writerow(data)
            print("frame id: ",self.frame_idx)

        # update frame
        self.frame_idx += 1
        self.prev_depth = self.curr_depth 
        self.prev_frame = self.curr_frame

        # End time
        end = time.time()
        # calculate the FPS for current frame detection
        self.fps = 1 / (end-start)



'''
    def imagePointCallback(self,points):
        # Back project 3d points to pixel mask.
        self.projected_mask = np.zeros((self.D_height,self.D_width,3))
        xyz_array = self.pointcloud2_to_xyz_array(points,True)
        if xyz_array.shape[0]>0:
            x = xyz_array[:,0]
            y = xyz_array[:,1]
            z = xyz_array[:,2]
            tmp_w = (x*self.D_fx/z)+self.D_cx
            tmp_h = (y*self.D_fy/z)+self.D_cy
            tmp_w = np.clip(np.round_(tmp_w), a_min=0, a_max=self.D_width-1).astype(np.int32)
            tmp_h = np.clip(np.round_(tmp_h), a_min=0, a_max=self.D_height-1).astype(np.int32)


            self.projected_mask[tmp_h,tmp_w] = xyz_array

            
            
        self.points =  []
        for coord in self.coord_list:
            if sum(self.projected_mask[coord[1],coord[0]]) != 0:
                self.points.append(self.projected_mask[coord[1],coord[0]])
        self.points = np.array(self.points)
        if self.points.shape[0]>0:
            self.plotter(self.points)
'''



if __name__ == '__main__':
    rospy.init_node("tracker_node")
    kp_tracker = KPTracker()
    # spin
    rospy.spin()
    kp_tracker.f.close()