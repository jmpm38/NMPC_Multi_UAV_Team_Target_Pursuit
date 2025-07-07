#!/usr/bin/env python
import rospy
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from enum import Enum
import logging
from ukf import UKF
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension

logging.basicConfig(
    level=logging.WARNING, 
    format='%(asctime)s - %(levelname)s: %(message)s', 
    datefmt='%H:%M:%S'  # Custom date format for shorter timestamp
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

generalLayout = MultiArrayLayout(
    dim=[
        MultiArrayDimension(label="components", size=4, stride=4)
    ],
    data_offset=0
)

num_states = 4


def matrix_norm(M):
    # Compute the norm of matrix M defined as sqrt(trace(M^T M))
    return np.sqrt(np.trace(np.dot(M.T, M)))

def load_and_sort_files(data_folder, uav_number):
    all_files = os.listdir(data_folder)
    txt_files = [f for f in all_files if f.endswith(".txt") and f != "vehicle_position_baro.txt"]
    sorted_txt_files = sorted(txt_files, key=lambda x: float(x.split('_')[0]))
    #sorted_txt_files = [f for f in sorted_txt_files if float(f.split('_')[0]) >= 936]
    return [(float(f.split('_')[0]), uav_number, f) for f in sorted_txt_files]

class StateSpaceModel():
    def __init__(self,A,B,C,D,Q,R):
        # x_k+1 = Ax_k + Bw_k
        # z_k = Cx_k + v_k
        # Q = E[w_kw_k]
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Q = Q
        self.R = R
        self.state = np.zeros((4, 1), dtype=float) #[x,y,vx,vy]
        self.P = np.diag([1,1,1,1]).astype(float)
        self.P = self.P.astype(float)
        
    

class estimator_node():

    # Track class to store track information data
    # state, prune count for estimation start flag and
    # status (Active, temporary or inactive)
    class TrackState():
        class Status(Enum):
            inactive = 0
            temporary = 1
            active = 2

        def __init__(self):
            self.state = np.zeros((num_states, 1), dtype=float) #[x,y,vx,vy]
            self.state[-1] += 1e-4
            self.prune = 1
            self.status = self.Status.inactive
          
    def __init__(self,name,model):
        self.name = name                     # Estimation node name for id
        self.measurement = None              # Measurment data
        self.neighbours = []                 # Neighbours data simulator
        self.model = model                   # Estimation model
        self.u = None                        # Information vector u, respect to measurement
        self.U = None                        # Information vector U, respect to observation covariance
        self.eps = 0.8                       # Epsilon, consensus tuning parameter
        self.ready = False                   # Boolean for estimation state, True = Estimation
        self.estimated_states = []           # Log estimation data
        self.measurement_data = []           # Log meeasurement data
        self.trackList = []                  # Track list variable for initializaiton 
        self.pruneCount = 0                  # Observation counter to detect loss of information
        
    # Run the estimation node, either for initialization or for 
    # estimation purposes.    
    def Run(self,measurement,timestamp):
        if self.trackList is None or len(self.trackList) == 0:
            # If trackList is empty means no target is being tracked
            rospy.loginfo("No track exists in the track list")
            # Create new tracks
            for meas in measurement:
                new_track_state = self.TrackState()
                new_track_state.status = 1
                new_track_state.state[0:2] = meas.reshape(-1, 1) 
                self.trackList.append(new_track_state)
        else:
            # If trackList is not empty, drone is either estimating or waiting
            # measurements for track confirmation 
            if self.ready:
                # Estimation state
                self.Association(measurement)
                self.Estimate(measurement, timestamp)
            else:
                # Initialization state
                self.StateInit(measurement)
  
    # State initialization function, which takes care of track tree when not esitmating
    def StateInit(self,measurement):
        associationThreshold = 1

        if measurement.size == 0:
            pass
        else:
            # If measurements are available, for each track calculate
            # closest distance measurement
            for track in self.trackList:
                distances = []
                current_state = track.state[0:2].T
                if measurement.size == 0:
                    continue
                else:
                    for meas in measurement:
                        distances.append(np.linalg.norm(meas - current_state, axis=1))
                    min_distance_index = np.argmin(distances)
                    if distances[min_distance_index]<associationThreshold:
                        # If the measurement is within a threshold of the track state, associate
                        track.prune += 1
                        track.state[0:2] = meas.reshape(-1, 1)
                        measurement = np.delete(measurement, min_distance_index, axis=0)    

            # After associaiton if a track has reached 10 consecutive measurments, begin tracking
            # pop up should appear
            pruneThreshold = 5
            for track in self.trackList:
                if track.prune > pruneThreshold:
                    print(track.state)
                    #user_input = input(f"{self.name}, Target detected. Follow: ")
                    #if condition user_input.upper() == 'Y'
                    if True:
                        self.model.set_state(track.state.reshape(-1))
                        self.ready = True
                        rospy.loginfo("[+] TARGET LOCKED")
                        return 
            
            # If any measurement has not been associated to a track generate new track possibilities  
            for meas in measurement:
                new_track_state = self.TrackState()
                new_track_state.status = 1
                new_track_state.state[0:2] = meas.reshape(-1, 1)  
                self.trackList.append(new_track_state)
        
    def Estimate(self,measurement,timestamp):
        
        
        neighbourInfo = self.Consensus()
        self.model.predict(0.1)
        
        if self.measurement.size == 0:
            rospy.loginfo(f"{self.name}, No measurement available")
            #If neighbourInfo is True means no neighbours had measurement information
            if(neighbourInfo == True):
                self.pruneCount += 1
            if (self.pruneCount == 50):
                self.ready = False
                self.trackList = []

        else:
            self.pruneCount = 0
            if self.measurement.shape[1]!=1:
                rospy.logdebug(f"{self.name}, Multiple measurements")
            
            R = np.diag([0.001,0.001]).astype(float)
            self.model.update([0,1],self.measurement,R)
            self.measurement_data.append((timestamp,self.measurement.flatten().tolist()))


        self.estimated_states.append((timestamp,self.model.x.flatten().tolist()))

    def Consensus(self):
        x_consensus = np.zeros(self.model.n_dim)
        p_consensus = np.zeros((self.model.n_dim, self.model.n_dim))
        
        ret = True

        ready_neighbours = [neighbour for neighbour in self.neighbours if neighbour.ready]
        n_ready_neighbours = len(ready_neighbours)


        if n_ready_neighbours == 0:
            # If no neighbour is ready, keep self.model.x as it is
            pass

        else:

            trust_scores = [1/(np.trace(self.model.p) + 1e-6)]
            for neighbour in ready_neighbours:
                trust_score = 1/(np.trace(neighbour.model.p) + 1e-6)
                trust_scores.append(trust_score)
            total_trust = sum(trust_scores)
            weights = [score / total_trust for score in trust_scores]

            w = 1 / (1 + n_ready_neighbours)
            x_consensus = weights[0] * self.model.x
            p_consensus = weights[0] * self.model.p

            for weight, neighbour in zip(weights[1:], ready_neighbours):
                x_consensus += weight * neighbour.model.x
                p_consensus += weight * neighbour.model.p

            self.model.x = x_consensus
            self.model.p = p_consensus
            ret = n_ready_neighbours == 0 or all(neighbour.pruneCount > 0 for neighbour in ready_neighbours)
        
        return ret 
    
    
    def Association(self,measurement):
        if measurement.size == 0:
            logger.debug(f"{self.name}, measurment vector empty for association")
            self.measurement = measurement
        else:
            current_state = self.model.get_state()[0:2]
            distances = np.linalg.norm(measurement - current_state, axis=1)
            min_distance_index = np.argmin(distances)

            dist = measurement-current_state
            dist_stat = [np.dot(np.dot(arr,np.linalg.inv(self.model.p[:2,:2])),np.transpose(arr)) for arr in dist]
            min_sdistance_index = np.argmin(dist_stat)
            #(dist_stat[min_sdistance_index])
            if (dist_stat[min_sdistance_index]<5000):
                self.measurement = np.array([measurement[min_distance_index]]).T
            else:
                self.measurement = np.array([])
            #print("Closest measurement:", np.array([measurement[min_distance_index]]).T)
            #print("Minimum distance: ", distances[min_distance_index])
            #print("Statistical distance distance: ", dist_stat)

    # Add neighbour information
    def AddNeighbours(self,neighbour):
        self.neighbours.append(neighbour)
            
def CalculateEstimationError(estimator,timestampsRef,xValuesRef,yValuesRef):
    errorEstimation = []
    for estimated_timestamp, estimated_position in estimator.estimated_states:
        # Find the closest reference position timestamp
        closest_ref_timestamp = min(timestampsRef, key=lambda x: abs(x - float(estimated_timestamp)))
        
        # Find the index of the closest reference position timestamp
        closest_ref_index = timestampsRef.index(closest_ref_timestamp)
        
        # Get the corresponding reference position
        ref_position = (xValuesRef[closest_ref_index], yValuesRef[closest_ref_index])
        
        # Calculate the error between estimated position and reference position
        error = (estimated_position[0] - ref_position[0], estimated_position[1] - ref_position[1])
        error = np.linalg.norm(error)
        # Append the error to the list of errors
        errorEstimation.append(error)

    errorMeas = []
    for estimated_timestamp, estimated_position in estimator.measurement_data:
        # Find the closest reference position timestamp
        closest_ref_timestamp = min(timestampsRef, key=lambda x: abs(x - float(estimated_timestamp)))
        
        # Find the index of the closest reference position timestamp
        closest_ref_index = timestampsRef.index(closest_ref_timestamp)
        
        # Get the corresponding reference position
        ref_position = (xValuesRef[closest_ref_index], yValuesRef[closest_ref_index])
        
        # Calculate the error between estimated position and reference position
        error = (estimated_position[0] - ref_position[0], estimated_position[1] - ref_position[1])
        error = np.linalg.norm(error)
        # Append the error to the list of errors
        errorMeas.append(error)

    return errorEstimation,errorMeas

def CalculateStateDifference(estimator1,estimator2):
    errorEstimation = []
    timestamps2 = [float(entry[0]) for entry in estimator2.estimated_states]
    xValues2 = [entry[1][0] for entry in estimator2.estimated_states]
    yValues2 = [entry[1][1] for entry in estimator2.estimated_states]
    vxValues2 = [entry[1][2] for entry in estimator2.estimated_states]
    vyValues2 = [entry[1][3] for entry in estimator2.estimated_states]

    
    for estimated_timestamp, estimated_position in estimator1.estimated_states:
        # Find the closest reference position timestamp
        closest_ref_timestamp = min(timestamps2, key=lambda x: abs(x - float(estimated_timestamp)))
        
        # Find the index of the closest reference position timestamp
        closest_ref_index = timestamps2.index(closest_ref_timestamp)
        
        # Get the corresponding reference position
        ref_position = (xValues2[closest_ref_index], yValues2[closest_ref_index],vxValues2[closest_ref_index],vyValues2[closest_ref_index])
        
        # Calculate the error between estimated position and reference position
        error = (estimated_position[0] - ref_position[0], estimated_position[1] - ref_position[1],estimated_position[2] - ref_position[2],estimated_position[3] - ref_position[3])
        error = np.linalg.norm(error)
        
        errorEstimation.append(error)

    return errorEstimation

def dynamicsFunction(x_in,timestep,u):
    '''this function is based on the x_dot and can be nonlinear as needed'''
    # Constant turn rate model
    ret = np.zeros(len(x_in))
    """ TC = (1-np.cos(x_in[4]*timestep))/x_in[4]
    TS = np.sin(x_in[4]*timestep)/x_in[4]
    ret[0] = x_in[0] + TS * x_in[2] - TC * x_in[3]
    ret[1] = x_in[1] + TC * x_in[2] + TS * x_in[3]
    ret[2] = np.cos(x_in[4]*timestep) * x_in[2] - np.sin(x_in[4]*timestep) * x_in[3]
    ret[3] = np.sin(x_in[4]*timestep) * x_in[2] + np.cos(x_in[4]*timestep) * x_in[3]
    ret[4] = x_in[4] """

    # Constant velocity model
    
    ret[0] = x_in[0] + timestep * x_in[2]
    ret[1] = x_in[1] + timestep * x_in[3]
    ret[2] = x_in[2]
    ret[3] = x_in[3]
    
    return ret

class ROSEstimationNode:
    def __init__(self):
        self.uav1Estimator = None
        self.uav2Estimator = None
        self.init_estimators()
        self.init_ros()

    def init_estimators(self):
        dt = 0.1

        Q = 10 * np.eye(num_states, dtype=float)  # model covariance

        R = 0.001 * np.eye(2, dtype=float)  # noise covariance   

        init_state = np.zeros(num_states, dtype=float)
        init_state[-1] += 1e-4

        init_covar = 10 * np.eye(num_states, dtype=float)

        self.uav1Estimator = estimator_node("uav1StateEstimator", UKF(num_states, Q, init_state, init_covar, 1e-3, 0, 2, dynamicsFunction))
        self.uav2Estimator = estimator_node("uav2StateEstimator", UKF(num_states, Q, init_state, init_covar, 1e-3, 0, 2, dynamicsFunction))

        #self.uav1Estimator.add_neighbours(self.uav2_estimator)
        #self.uav2Estimator.add_neighbours(self.uav1_estimator)
        
    def init_ros(self):
        rospy.init_node('estimation_node')
        self.uav1Subscriber = rospy.Subscriber('/uav1/detection_lidar', Float64MultiArray, self.uav1_callback)
        self.uav2Subscriber = rospy.Subscriber('/uav2/detection_lidar', Float64MultiArray, self.uav2_callback)

        self.uav1EstimatedStatePublisher = rospy.Publisher('/uav1/target_estimate', Float64MultiArray, queue_size=10)
        #self.uav2EstimatedStatePublisher = rospy.Publisher('/uav2/target_estimate', Float64MultiArray, queue_size=10)

    def process_message(self, msg):
        num_detections = msg.layout.dim[0].size
        num_components = msg.layout.dim[1].size
        
        if num_detections == 0:
            return msg.data[0], np.array([])
        
        timestamp = msg.data[0]
        measurements = np.array(msg.data).reshape(num_detections, num_components)

        return timestamp, measurements[:,1:]

    def uav1_callback(self, msg):
        
        timestamp, measurements = self.process_message(msg)
        rospy.loginfo("------------------------------")
        rospy.loginfo(f"Timestamp: {timestamp}, UAV1")
        ts = time.time()
        self.uav1Estimator.Run(measurements[:, :2] if len(measurements) > 0 else measurements, timestamp)
        rospy.loginfo(f"Estimation time: {time.time()-ts}")
        if self.uav1Estimator.ready:
            estimatedStateMessage = Float64MultiArray()
            estimatedStateMessage.data = self.uav1Estimator.model.x.flatten().tolist()
            estimatedStateMessage.layout = generalLayout
            self.uav1EstimatedStatePublisher.publish(estimatedStateMessage)

        

    def uav2_callback(self, msg):
        timestamp, measurements = self.process_message(msg)
        rospy.loginfo("------------------------------")
        rospy.loginfo(f"Timestamp: {timestamp}, UAV2")
        ts = time.time()
        self.uav2Estimator.run(measurements[:, :2] if len(measurements) > 0 else measurements, timestamp)
        rospy.loginfo(f"Estimation time: {time.time()-ts}")
        if self.uav2Estimator.ready:
            estimatedStateMessage = Float64MultiArray()
            estimatedStateMessage.data = self.uav2Estimator.model.x.flatten().tolist()
            self.uav2EstimatedStatePublisher.publish(estimatedStateMessage)

if __name__ == "__main__":
    node = ROSEstimationNode()
    rospy.loginfo("[+] Estimation ros node has started")
    rospy.spin()