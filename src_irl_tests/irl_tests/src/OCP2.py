import casadi as ca
import numpy as np
from matplotlib import pyplot as plt
from casadi import sin, cos, pi, tan, atan, atan2
import math

# Cost Function Terms

def dist_term_ocp(uav_pos, target_pos, Q1, standoff_dist):
    relative_pos = ca.minus(uav_pos, target_pos)
    distance = (relative_pos[0] ** 2) + (relative_pos[1] ** 2) + (relative_pos[2] ** 2)
    dist_term = Q1 * (distance ** 2)
    return dist_term

def control_term_ocp(current_control_term, u_ref, Q2, yaw_ref):
    control_deviation = ca.minus(current_control_term, u_ref)
    squared_control = ca.vertcat(control_deviation[0] * control_deviation[0], control_deviation[1] * control_deviation[1],
                                 control_deviation[2] * control_deviation[2], control_deviation[3] * control_deviation[3])
    angle = current_control_term[3] - yaw_ref
    yaw = ca.atan2(ca.sin(angle), ca.cos(angle)) # talvez funcione melhor com esta !!!!
    yaw_sq = yaw * yaw
    control_term = Q2[0] * squared_control[0] + Q2[1] * squared_control[1] + Q2[2] * squared_control[2] + Q2[3] * yaw_sq
    return control_term

def control_smooth_ocp(current_control_term, u_past, Q3):
    control_deviation = atan(current_control_term - u_past)
    squared_control_smth = control_deviation * control_deviation
    weighted_control_smooth = Q3 * squared_control_smth
    return weighted_control_smooth

def calculate_beta_ocp(target_pos, current_uav_pos, roll, pitch, yaw, y_rotation_ang):
    R_b2w_opt = ca.MX.zeros(3,3)
    R_b2w_opt[0,0] = cos(yaw) * cos(pitch) 
    R_b2w_opt[0,1] = cos(yaw) * sin(pitch) * sin(roll) - sin(yaw) * cos(roll)
    R_b2w_opt[0,2] = cos(yaw) * sin(pitch) * cos(roll) + sin(yaw) * sin(roll)

    R_b2w_opt[1,0] = sin(yaw) * cos(pitch)
    R_b2w_opt[1,1] = sin(yaw) * sin(pitch) * sin(roll) + cos(yaw) * cos(roll)
    R_b2w_opt[1,2] = sin(yaw) * sin(pitch) * cos(roll) - cos(yaw) * sin(roll)

    R_b2w_opt[2,0] = -sin(pitch)
    R_b2w_opt[2,1] = cos(pitch) * sin(roll)
    R_b2w_opt[2,2] = cos(pitch) * cos(roll)

    R_w2b_opt = R_b2w_opt.T
    cam_pos = [0.04, 0, -0.11]
    target_position_worldF = ca.minus(target_pos, current_uav_pos)
    target_position_worldF = ca.minus(target_position_worldF, cam_pos)
    target_position_bodyF = ca.mtimes(R_w2b_opt, target_position_worldF) #contains target coordinates on the body frame

    camera_rotation_matrix_b2c = ca.MX.zeros(3,3)
    camera_rotation_matrix_b2c[0,0] = cos(y_rotation_ang)
    camera_rotation_matrix_b2c[0,1] = 0
    camera_rotation_matrix_b2c[0,2] = sin(y_rotation_ang)

    camera_rotation_matrix_b2c[1,0] = 0
    camera_rotation_matrix_b2c[1,1] = 1
    camera_rotation_matrix_b2c[1,2] = 0

    camera_rotation_matrix_b2c[2,0] = -sin(y_rotation_ang)
    camera_rotation_matrix_b2c[2,1] = 0
    camera_rotation_matrix_b2c[2,2] = cos(y_rotation_ang)

    target_position_CF = ca.mtimes(camera_rotation_matrix_b2c, target_position_bodyF)
    return target_position_CF #beta

def centered_FOV_ocp(beta_angle, Q5):
    fov_term = Q5 * ((1-beta_angle) ** 2)
    return fov_term


