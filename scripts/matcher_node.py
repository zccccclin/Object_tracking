#!/usr/bin/env python
from logging import raiseExceptions
from math import pi
import csv, os, rospy, cv2, time, message_filters, tf
import numpy as np
from sensor_msgs.msg import PointCloud2, CameraInfo, Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from func import *


class KPMatcher:
    def __init__(self):
        # get params
        correction = rospy.get_param("/correction",False)
        logging = rospy.get_param("/log_data", False)
        matcher_func = rospy.get_param('/matcher_func','flann')
        correction_interval = rospy.get_param('/correction_interval',5)
        print(f"Correction: {correction}, Loggging: {logging}, Matcher func: {matcher_func}, Correction interval: {correction_interval}.")

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
        self.ref_frame = self.colorInfo.header.frame_id

        # CV initialization
        self.bridge = CvBridge()

        # Img variable 
        self.curr_frame = None
        self.prev_frame = None
        self.curr_depth = None
        self.prev_depth = None
        self.fps = 0
        self.frame_idx = 0
        self.correction_interval = correction_interval
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

        # Detector and matcher initialization
        self.orb = cv2.ORB_create()
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6, # 12
                    key_size = 10,     # 20
                    multi_probe_level = 1   ) #2
        search_params = dict(checks=100)
        if matcher_func == 'flann':
            self.matcher = cv2.FlannBasedMatcher(index_params,search_params)
        elif matcher_func == 'bf':
            self.matcher = cv2.BFMatcher()
        else:
            raise Exception("Invalid Matcher input, 'bf' and 'flann' accepted only.")



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
        #------------------------------------ START of 2D processing-----------------------------------------
        #color msg -> image processing
        frame = self.bridge.imgmsg_to_cv2(color)
        self.curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # kp matching process 
        if self.prev_frame is not None and self.initialized:
            kp1, des1 = self.orb.detectAndCompute(self.prev_frame, None)
            kp2, des2 = self.orb.detectAndCompute(self.curr_frame, None)
            #Get good matches (nearby, lower's thres)
            good_matches = []
            inlier_matches = []
            matches = self.matcher.knnMatch(des1,des2, k=2)
            try:
                for m,n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            except:
                pass
            
            # RANSAC (need at least 4 points)
            if len(good_matches) > 4:
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,2)
                H, mask = cv2.findHomography(src_pts,dst_pts,cv2.USAC_MAGSAC,5.0,0.99,2000)
                matchmask = mask.ravel().tolist()
                for m, n in zip(good_matches,matchmask):
                    if n == 1:
                        inlier_matches.append(m)

            # Get 2D KP idx
            self.prev_idx = np.int16([kp1[m.queryIdx].pt for m in inlier_matches])
            self.curr_idx = np.int16([kp2[m.trainIdx].pt for m in inlier_matches])

            # Draw matches
            img_matches = np.empty((max(self.prev_frame.shape[0], self.curr_frame.shape[0]), self.prev_frame.shape[1] + self.curr_frame.shape[1], 3), dtype=np.uint8)
            cv2.drawMatches(self.prev_frame, kp1, self.curr_frame, kp2, inlier_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.putText(img_matches, f'FPS: {int(self.fps)}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(img_matches, f'No. 2DKP: {len(inlier_matches)}', (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        #------------------------------------ END of 2D processing-----------------------------------------

        #------------------------------------ START of 3D processing-----------------------------------------
        # depth msg -> image processing
        depth_img = self.bridge.imgmsg_to_cv2(depth,'passthrough')
        self.curr_depth = depth_img
        svdRT = tf.transformations.identity_matrix()
        pnpRT = tf.transformations.identity_matrix()
        # 3d matching process
        if self.prev_depth is not None and self.initialized:
            # Get 3D points and remove invalid 2D KP idx (no depth):
            self.prev_3d, self.curr_3d, self.prev_idx, self.curr_idx = pixelto3d(self.D_K, 
                                                                        self.prev_idx, self.curr_idx, 
                                                                        self.prev_depth, self.curr_depth)
            cv2.putText(img_matches, f'No. 3DKP: {len(self.curr_3d)}', (480, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
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
    
            # Publishers
            self.imagepub.publish(self.bridge.cv2_to_imgmsg(img_matches))
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
        self.prev_frame = self.curr_frame
        self.prev_depth = self.curr_depth 
        self.frame_idx += 1

        # fps calculation
        end = time.time()
        total_time = end - start
        self.fps = 1 / total_time



if __name__ == '__main__':
    rospy.init_node("matcher_node")
    kp_tracker = KPMatcher()
    # spin
    rospy.spin()
    kp_tracker.f.close()
