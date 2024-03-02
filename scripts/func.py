from math import pi
import numpy as np
import cv2
import tf
from geometry_msgs.msg import Point, Quaternion, PoseStamped
from sensor_msgs.msg import PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2


def pixelto3d(K,prev_pt,curr_pt,prev_depth,curr_depth):
    cx = K[0][2]
    cy = K[1][2]
    fx = K[0][0]
    fy = K[1][1] 
    prev_3d = []
    curr_3d = []
    prev_pt_tmp = []
    curr_pt_tmp = []
    for prev, curr in zip(prev_pt,curr_pt):
        prev_z = prev_depth[prev[1],prev[0]]
        curr_z = curr_depth[curr[1],curr[0]]
        if prev_z != 0 and curr_z != 0:
            prevz = prev_z*0.001
            prevx = ((prev[0] - cx)*(1/fx))*prevz
            prevy = ((prev[1] - cy)*(1/fy))*prevz
            currz = curr_z*0.001
            currx = ((curr[0] - cx)*(1/fx))*currz
            curry = ((curr[1] - cy)*(1/fy))*currz
            prev_3d.append([prevx,prevy,prevz])
            curr_3d.append([currx,curry,currz])
            prev_pt_tmp.append(prev)
            curr_pt_tmp.append(curr)
    prev_3d = np.array(prev_3d,dtype=np.float64)
    curr_3d = np.array(curr_3d,dtype=np.float64)
    prev_pt_tmp = np.array(prev_pt_tmp,dtype=np.float64)
    curr_pt_tmp = np.array(curr_pt_tmp,dtype=np.float64)
    if prev_3d.shape == (0,):
        prev_3d = prev_3d.reshape((0,3))
        curr_3d = curr_3d.reshape((0,3))
    return prev_3d,curr_3d,prev_pt_tmp,curr_pt_tmp

def solve_by_pnp(arr_3d,arr_2d,k):
    succ, rvec, tvec, _i = cv2.solvePnPRansac(arr_3d,arr_2d,k,None,flags=cv2.SOLVEPNP_SQPNP,reprojectionError=2,iterationsCount=500) #SOLVEPNP_EPNP, SOLVEPNP_SQPNP, SOLVEPNP_AP3P
    rmat = cv2.Rodrigues(rvec)[0]
    RT = tf.transformations.identity_matrix()
    RT[:3,:3] = rmat
    RT[:3,3] = tvec.flatten()
    return RT

def solve_by_svd(prev_3d, curr_3d):
    prev_3d = prev_3d.T
    curr_3d = curr_3d.T
    cent1 = np.mean(prev_3d,axis=1)
    cent2 = np.mean(curr_3d,axis=1)
    cent1 = cent1.reshape(-1,1)
    cent2 = cent2.reshape(-1,1)
    pt1 = prev_3d - cent1
    pt2 = curr_3d - cent2
    #print(pt1.shape, pt2.shape) 
    W = pt1 @ pt2.T
    #print(W.shape)
    u,s,vh = np.linalg.svd(W)
    #print(u.shape, vh.shape)
    R = vh.T @ u.T

    if np.linalg.det(R) < 0:
        vh[2,:] *= -1
        R = vh.T @ u.T

    T = cent2 - (R @ cent1)

    RT = tf.transformations.identity_matrix()
    RT[:3,:3] = R
    RT[:3,3] = T.flatten()
    return RT

def pose_to_homo(pose):
    xyz = np.array([pose.pose.position.x,pose.pose.position.y,pose.pose.position.z])
    quat = np.array([pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w])
    H = tf.transformations.quaternion_matrix(quat)
    H[:3,3] = xyz
    return H

def homo_to_pose(homo,ref_frame):
    pose = PoseStamped()
    pose.header.frame_id = ref_frame
    pos = Point(*homo[:3,3])
    quat = Quaternion(*tf.transformations.quaternion_from_matrix(homo))
    pose.pose.position = pos
    pose.pose.orientation = quat
    return pose

def xyz2pc(xyz):
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1)]

    header = Header()
    header.frame_id = "camera_color_optical_frame"

    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    points = np.array([x,y,z,z]).reshape(4,-1).T

    pointcloud = pc2.create_cloud(header, fields, points)
    return pointcloud

def dist3d_diff(arr1,arr2):
    x1,y1,z1 = arr1[0], arr1[1], arr1[2]
    x2,y2,z2 = arr2[0], arr2[1], arr2[2]
    dist = ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**0.5
    return dist


def ang_limit(rpy):
    a = []
    for i in rpy:
        ang = i
        if i >= 180:
            ang = 360 - i
        a.append(ang)
    return np.array(a)
'''
# Additional function param (pcd msg to xyzarray)
DUMMY_FIELD_PREFIX = '__'
type_mappings = [(PointField.INT8, np.dtype('int8')), (PointField.UINT8, np.dtype('uint8')), (PointField.INT16, np.dtype('int16')),
                (PointField.UINT16, np.dtype('uint16')), (PointField.INT32, np.dtype('int32')), (PointField.UINT32, np.dtype('uint32')),
                (PointField.FLOAT32, np.dtype('float32')), (PointField.FLOAT64, np.dtype('float64'))]
pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)
pftype_sizes = {PointField.INT8: 1, PointField.UINT8: 1, PointField.INT16: 2, PointField.UINT16: 2,
                PointField.INT32: 4, PointField.UINT32: 4, PointField.FLOAT32: 4, PointField.FLOAT64: 8}

def fields_to_dtype(fields, point_step):
    offset = 0
    np_dtype_list = []
    for f in fields:
        while offset < f.offset:
            # might be extra padding between fields
            np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        dtype = pftype_to_nptype[f.datatype]
        if f.count != 1:
            dtype = np.dtype((dtype, f.count))

        np_dtype_list.append((f.name, dtype))
        offset += pftype_sizes[f.datatype] * f.count

    # might be extra padding between points
    while offset < point_step:
        np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
        offset += 1
        
    return np_dtype_list

def pointcloud2_to_array(cloud_msg, squeeze=True):

    # construct a numpy record type equivalent to the point type of this cloud
    dtype_list = fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)

    # parse the cloud into an array
    cloud_arr = np.frombuffer(cloud_msg.data, dtype_list)

    # remove the dummy fields that were added
    cloud_arr = cloud_arr[
        [fname for fname, _type in dtype_list if not (fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]
    
    if squeeze and cloud_msg.height == 1:
        return np.reshape(cloud_arr, (cloud_msg.width,))
    else:
        return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))


def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):

    # remove crap points
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]
    
    # pull out x, y, and z values
    points = np.zeros(cloud_array.shape + (3,), dtype=dtype)
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    points[...,2] = cloud_array['z']
    return points

def pointcloud2_to_xyz_array(cloud_msg, remove_nans=True):
    return get_xyz_points(pointcloud2_to_array(cloud_msg), remove_nans=remove_nans)
'''