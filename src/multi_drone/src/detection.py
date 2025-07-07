#!/usr/bin/env python

## Implementation
import numpy as np
np.float = float
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from enum import Enum
from scipy.spatial.transform import Rotation as R

## ROS related
import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2
import ros_numpy
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension

## Performance
import time
import pyransac3d as pyrsc

class Detector():
    class Criteria(Enum):
        area = 1
        closeness = 2
        variance = 3
    class BoxFilter_Person():
        def __init__(self):
            self.Th_height = np.array([1.2, 2]) 
            self.Th_width = np.array([0.15,1])
            self.Th_length = np.array([0.3,1])
            self.Th_area = np.array([0, 2])
            self.Th_min_pointdensity = 8
    class BoxFilter_Vehicle():
        def __init__(self):
            self.Th_height = np.array([1.2, 2.1])  
            self.Th_width = np.array([1.4,3.5])
            self.Th_length = np.array([2,6])
            self.Th_area = np.array([0, 13])
            self.Th_min_pointdensity = 2.5
            self.Th_ratio = 1.5

    def __init__(self):
        self.fitting_criteria = self.Criteria.variance
        self.d_theta_deg_for_search = 1.0
        self.box_filter_person = self.BoxFilter_Person()
        self.box_filter_vehicle = self.BoxFilter_Vehicle()

    def Clustering(self,data):
        return(DBSCAN(eps=2, min_samples=5).fit(data))

    def LShapeFitting(self,cluster,z_th):
        xy = np.array([cluster[:,0], cluster[:,1]]).T

        d_theta = np.deg2rad(self.d_theta_deg_for_search)
        min_cost = (-float('inf'), None)
        for theta in np.arange(0.0, np.pi / 2.0 - d_theta, d_theta):
            rot_mat = R.from_euler('z', theta).as_matrix()[0:2, 0:2]
            c = xy @ rot_mat

            c1 = c[:, 0]
            c2 = c[:, 1]

            # Select criteria
            cost = 0.0
            if self.fitting_criteria == self.Criteria.area:
                cost = self.calc_area_criterion(c1, c2)
            elif self.fitting_criteria == self.Criteria.closeness:
                cost = self.calc_closeness_criterion(c1, c2)
            elif self.fitting_criteria == self.Criteria.variance:
                cost = self.calc_variance_criterion(c1, c2)

            if min_cost[0] < cost:
                min_cost = (cost, theta)

        # calc best rectangle
        sin_s = np.sin(min_cost[1])
        cos_s = np.cos(min_cost[1])

        c1_s = xy @ np.array([cos_s, sin_s]).T
        c2_s = xy @ np.array([-sin_s, cos_s]).T

        rect = RectangleData()
        rect.a[0] = cos_s
        rect.b[0] = sin_s
        rect.c[0] = min(c1_s)
        rect.a[1] = -sin_s
        rect.b[1] = cos_s
        rect.c[1] = min(c2_s)
        rect.a[2] = cos_s
        rect.b[2] = sin_s
        rect.c[2] = max(c1_s)
        rect.a[3] = -sin_s
        rect.b[3] = cos_s
        rect.c[3] = max(c2_s)
        rect.orientation = min_cost[1]
        rect.calc_rect_contour()
        rect.center_z = z_th
        rect.dz = np.abs(rect.center_z-np.max(cluster[:,2]))
        rect.points = len(cluster)
        return rect

    def BoxFiltering(self,box,n_points):
        h = box.dz
        w = box.dx if box.dx < box.dy else box.dy
        l = box.dy if box.dx < box.dy else box.dx

        if self.box_filter_person.Th_height[0] <= h <= self.box_filter_person.Th_height[1] and \
            self.box_filter_person.Th_width[0] <= w <= self.box_filter_person.Th_width[1] and \
            self.box_filter_person.Th_length[0] <= l <= self.box_filter_person.Th_length[1] and \
            self.box_filter_person.Th_area[0] <= (w*l) <= self.box_filter_person.Th_area[1] and \
            self.box_filter_person.Th_min_pointdensity <= n_points/(w*l*h):
                print("Target Person")
                return True
        if self.box_filter_vehicle.Th_height[0] <= h <= self.box_filter_vehicle.Th_height[1] and \
            self.box_filter_vehicle.Th_width[0] <= w <= self.box_filter_vehicle.Th_width[1] and \
            self.box_filter_vehicle.Th_length[0] <= l <= self.box_filter_vehicle.Th_length[1] and \
            self.box_filter_vehicle.Th_area[0] <= (w*l) <= self.box_filter_vehicle.Th_area[1] and \
            self.box_filter_vehicle.Th_min_pointdensity <= 10 and\
            self.box_filter_vehicle.Th_ratio < (l/w):
                print("Target Vehicle")
                return True
        return False

    def run(self,points,z_th):
        clustering = self.Clustering(points)
        unique_labels = np.unique(clustering.labels_)
        clusters = []
        detection = []
        for label in unique_labels:
            if label == -1:  # Noise points, not assigned to any cluster
                continue
            cluster_points = points[clustering.labels_ == label]
            if (np.min(cluster_points[:,2])<z_th+1):
                clusters.append(cluster_points)
                rect = self.LShapeFitting(cluster_points,z_th)
                if(self.BoxFiltering(rect,len(cluster_points))):
                    detection.append(rect)

        return clusters, detection        
        
    @staticmethod
    def calc_area_criterion(c1, c2):
        c1_max, c1_min, c2_max, c2_min = Detector.find_min_max(c1, c2)
        alpha = -(c1_max - c1_min) * (c2_max - c2_min)
        return alpha

    def calc_closeness_criterion(self, c1, c2):
        c1_max, c1_min, c2_max, c2_min = Detector.find_min_max(c1, c2)

        # Vectorization
        d1 = np.minimum(c1_max - c1, c1 - c1_min)
        d2 = np.minimum(c2_max - c2, c2 - c2_min)
        d = np.maximum(np.minimum(d1, d2), self.min_dist_of_closeness_criteria)
        beta = (1.0 / d).sum()

        return beta

    @staticmethod
    def calc_variance_criterion(c1, c2):
        c1_max, c1_min, c2_max, c2_min = Detector.find_min_max(c1, c2)

        # Vectorization
        d1 = np.minimum(c1_max - c1, c1 - c1_min)
        d2 = np.minimum(c2_max - c2, c2 - c2_min)
        e1 = d1[d1 < d2]
        e2 = d2[d1 >= d2]
        v1 = - np.var(e1) if len(e1) > 0 else 0.
        v2 = - np.var(e2) if len(e2) > 0 else 0.
        gamma = v1 + v2

        return gamma

    @staticmethod
    def find_min_max(c1, c2):
        c1_max = max(c1)
        c2_max = max(c2)
        c1_min = min(c1)
        c2_min = min(c2)
        return c1_max, c1_min, c2_max, c2_min

class RectangleData:
    class detection_type(Enum):
        car = 0
        person = 1
    
    def __init__(self):
        self.a = [None] * 4
        self.b = [None] * 4
        self.c = [None] * 4

        self.rect_c_x = [None] * 5
        self.rect_c_y = [None] * 5

        self.center_x = None
        self.center_y = None
        self.center_z = None
        self.dx = None
        self.dy = None
        self.dz = None
        self.orientation = None
        self.points = None

        self.id = None


    def plot(self):
        self.calc_rect_contour()
        plt.plot(self.rect_c_x, self.rect_c_y, "-k")
        plt.scatter(self.center_x,self.center_y,marker = "x",s=50)

    def calc_rect_contour(self):
        self.rect_c_x[0], self.rect_c_y[0] = self.calc_cross_point(
            self.a[0:2], self.b[0:2], self.c[0:2])
        self.rect_c_x[1], self.rect_c_y[1] = self.calc_cross_point(
            self.a[1:3], self.b[1:3], self.c[1:3])
        self.rect_c_x[2], self.rect_c_y[2] = self.calc_cross_point(
            self.a[2:4], self.b[2:4], self.c[2:4])
        self.rect_c_x[3], self.rect_c_y[3] = self.calc_cross_point(
            [self.a[3], self.a[0]], [self.b[3], self.b[0]], [self.c[3], self.c[0]])
        self.rect_c_x[4], self.rect_c_y[4] = self.rect_c_x[0], self.rect_c_y[0]
        self.center_x = self.rect_c_x[0] + (self.rect_c_x[2] - self.rect_c_x[0])/2
        self.center_y = self.rect_c_y[0] + (self.rect_c_y[2] - self.rect_c_y[0])/2
        self.dx = np.sqrt((self.rect_c_x[1] - self.rect_c_x[0])**2 + (self.rect_c_y[1] - self.rect_c_y[0])**2)
        self.dy = np.sqrt((self.rect_c_x[2] - self.rect_c_x[1])**2 + (self.rect_c_y[2] - self.rect_c_y[1])**2)
        self.dz = None

    @staticmethod
    def calc_cross_point(a, b, c):
        x = (b[0] * -c[1] - b[1] * -c[0]) / (a[0] * b[1] - a[1] * b[0])
        y = (a[1] * -c[0] - a[0] * -c[1]) / (a[0] * b[1] - a[1] * b[0])
        return x, y

def get_xyz_points(cloud_array,simulation=False, remove_nans=True, dtype=np.float32):
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]

    points = np.zeros(cloud_array.shape + (4,), dtype=dtype)
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    if simulation:
        points[...,2] = cloud_array['z']
    else:
        points[...,2] = -cloud_array['z']

    return points

def filter_points(points, xyThreshold=40, groundPointDistance=0.3):
    points = points[(np.abs(points[:, 0]) > 0.1) & (np.abs(points[:, 1]) > 0.1) & ((points[:, 2]) < 1)]

    # Fit a plane to the points to identify the ground
    planeModel = pyrsc.Plane()
    bestEquation, bestInliers = planeModel.fit(points[:, :3], groundPointDistance, 100)

    # Remove ground points based on the fitted plane
    ground_mask = np.zeros(len(points), dtype=bool)
    ground_mask[bestInliers] = True
    points = points[~ground_mask]

    ground_th = -abs(bestEquation[3])+0.1

    points = points[(points[:, 2] > ground_th) & (np.abs(points[:, 0]) < xyThreshold) &   (np.abs(points[:, 1]) < xyThreshold)]
    return points,ground_th

def lidar_callback(msg, record=True, simulation=True):
    rospy.loginfo("------------------------------")
    rospy.loginfo(f"Timestamp:{msg.header.stamp.to_sec()}, UAV1 PC2 received")
    if simulation:
        try:
            transform = tf_buffer.lookup_transform(source_frame=msg.header.frame_id, target_frame='uav1/gps_baro_origin', time=rospy.Time(0), timeout=rospy.Duration(1))
            msg = do_transform_cloud(msg, transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("Failed to transform point cloud: %s", str(e))
            return
    
    msgCloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    ts = time.time()

    pointsInitial = get_xyz_points(msgCloud,simulation, True)

    if len(pointsInitial) == 0:
        rospy.logwarn("!! Empty ")
        detectionArray = Float64MultiArray()
        detectionArray.data = [msg.header.stamp.to_sec()]
        detectionArray.layout = MultiArrayLayout(
            dim=[
                MultiArrayDimension(label="detections", size=0, stride=0),
                MultiArrayDimension(label="components", size=1, stride=1)
            ],
            data_offset=0
        )
        detectionPublisher.publish(detectionArray)
        return 
    

    pointsFiltered,groundThreshold = filter_points(points=pointsInitial,xyThreshold=40,groundPointDistance=0.3)

    clusters, rect = detector.run(pointsFiltered,groundThreshold)

    rospy.loginfo(f"Inference time: {time.time()-ts}") 

    detectionArray = Float64MultiArray()
    detectionData = []
    for det in rect:
        detectionData.extend([msg.header.stamp.to_sec(), det.center_x, det.center_y, det.center_z, det.dx, det.dy, det.dz, det.orientation])
    detectionArray.data = detectionData
    num_detections = len(rect)
    detectionArray.layout = MultiArrayLayout(
        dim=[
            MultiArrayDimension(label="detections", size=num_detections, stride=num_detections * 8),
            MultiArrayDimension(label="components", size=8, stride=8)
        ],
        data_offset=0
    )

    if num_detections == 0:
        # Handle the case with no detections
        detectionArray.data = [msg.header.stamp.to_sec()]
        detectionArray.layout.dim[0].size = 0
        detectionArray.layout.dim[0].stride = 0
        detectionArray.layout.dim[1].size = 1
        detectionArray.layout.dim[1].stride = 1

    detectionPublisher.publish(detectionArray)

    return



if __name__ == '__main__':
   
    detector = Detector()

    rospy.init_node("detection_stage")

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    topic_name = '/uav1/os_cloud_nodelet/points'
    #topic_name = '/livox/lidar'
    subs = rospy.Subscriber(topic_name, PointCloud2, lidar_callback,queue_size=1,buff_size=2**24)
    detectionPublisher = rospy.Publisher('/uav1/detection_lidar', Float64MultiArray, queue_size=10)

    rospy.loginfo("[+] Detection ros node has started")
    rospy.spin()