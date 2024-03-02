from itertools import count

from sqlalchemy import true
import rospy
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Int16MultiArray

from cv_bridge import CvBridge, CvBridgeError
import sys
import os
import cv2
import numpy as np
from PIL import Image

class ImageListener:
    def __init__(self,depth,point,coord):
        self.bridge = CvBridge()
        self.depthsub = rospy.Subscriber(depth, msg_Image, self.imageDepthCallback)
        self.pointsub = rospy.Subscriber(point, PointCloud2, self.imagePointCallback)
        self.coordsub = rospy.Subscriber(coord, Int16MultiArray, self.coordCallback)
        self.uvs = []
        rospy.spin()

    def imageDepthCallback(self,data):
        cv_image = self.bridge.imgmsg_to_cv2(data,'passthrough')
        #if self.coord is not None:
            #for kp in self.coord:
                #cv2.circle(cv_image, (int(kp[0]), int(kp[1])), 3, (255, 255, 255), 1)

        #self.showImage('depth',cv_image)
        #cv2.imshow("depth",cv_image)
        #cv2.waitKey(1)
        
    def imagePointCallback(self,points):
        xyz_list = []
        for data in pc2.read_points_list(points,uvs=[[15,15]]):
            xyz_list.append([data[0], data[1], data[2], data[3]])
        print(xyz_list)

    def pointcloud_to_xyz(self,points):
        pass

    def showImage(self, imageName, inputImage, delay=1):
        cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
        cv2.imshow(imageName, inputImage)
        cv2.waitKey(delay)

    def coordCallback(self,msg):
        data = np.array(msg.data)
        data = np.reshape(data,(-1,2))
        self.uvs = []
        for coord in data:
            self.uvs.append((coord[0],coord[1]))
        

if __name__ == '__main__':
    rospy.init_node("pointcloud_node")
    depth = '/camera/depth/image_rect_raw'  # check the depth image topic in your Gazebo environmemt and replace this with your
    point = '/camera/depth/color/points'
    coord = '/coord'
    listener = ImageListener(depth,point,coord)
