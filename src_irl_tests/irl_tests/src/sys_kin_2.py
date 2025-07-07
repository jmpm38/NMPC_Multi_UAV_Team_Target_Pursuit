import casadi as ca
import numpy as np
from matplotlib import pyplot as plt
from casadi import sin, cos, pi, tan
import math

#%% UAV Kinematics + Discretization
class UAV_kin_class:
    def __init__(self, dt):
        self.dt = dt

    # Integration method for discretization of the system kinematics differential equations
    def eulerforward(self, x_dot, x):
        return x + x_dot * self.dt

    def uav_kinematics(self):
        # State Variables
        p_x = ca.MX.sym('p_x')
        p_y = ca.MX.sym('p_y')
        p_z = ca.MX.sym('p_z')
        v_x = ca.MX.sym('v_x')
        v_y = ca.MX.sym('v_y')
        v_z = ca.MX.sym('v_z')
        phi = ca.MX.sym('roll') #roll
        theta = ca.MX.sym('pitch') #pitch
        psi = ca.MX.sym('yaw') #yaw

        states = ca.vertcat(
            p_x,
            p_y,
            p_z,
            v_x,
            v_y,
            v_z,
            phi,
            theta,
            psi
        )
        self.n_states = states.numel() # get the number of elements


        # Control Variables
        T = ca.MX.sym('Thrust_acc')
        phi_ref = ca.MX.sym('roll_ref')
        theta_ref = ca.MX.sym('pitch_ref')
        psi_ref = ca.MX.sym('yaw_ref')


        controls = ca.vertcat(
            T,
            phi_ref,
            theta_ref,
            psi_ref
        )
        self.n_controls = controls.numel()


        # Define the differential equation for each state
        # ODE Parameters
        # Phi, Theta, and Psi ode Parameters - values based on Kagan's PhD

        # YAW:
        # tau_psi = 0.9057692510389791 # 0.57
        # K_psi = 1.0030810632020697 # 1.09
        K_psi = 0.57
        tau_psi = 1.09

        # ROLL:
        # K_phi = 1.0997575861228897 # 0.95
        # tau_phi = 0.35303945522709046 # 0.4
        K_phi = 0.95
        tau_phi =0.4

        # PITCH:
        # K_theta = 1.1016541694252724 # 0.95
        # tau_theta = 0.3539721251121422 # 0.33
        K_theta = 0.95
        tau_theta = 0.33

        # Linear damping terms - from Kinematics Paper
        Ax = 0.1#23
        Ay = 0.1#23
        Az = 0.2
        # Gravity acceleration
        g = 9.81 # m/s^2
        m = 2.4 # kg # 2.4
        # Rotation Matrices
        # body to world frame
        R_b2w = ca.MX.zeros(3,3)
        R_b2w[0,0] = cos(psi) * cos(theta) 
        R_b2w[0,1] = (-sin(psi)) * cos(phi) + cos(psi) * sin(theta) * sin(phi)
        R_b2w[0,2] = sin(psi) * sin(phi) + cos(psi) * sin(theta) * cos(phi)

        R_b2w[1,0] = sin(psi) * cos(theta)
        R_b2w[1,1] = cos(psi) * cos(phi) + sin(psi) * sin(theta) * sin(phi)
        R_b2w[1,2] = (-cos(psi)) * sin(phi) + sin(psi) * sin(theta) * cos(phi)

        R_b2w[2,0] = -sin(theta)
        R_b2w[2,1] = cos(theta) * sin(phi)
        R_b2w[2,2] = cos(theta) * cos(phi)
        # world to body frame
        R_w2b = R_b2w.T

        # Rotation Matrix times Thrust acceleration vector in the body frame
        T_aux = ca.vertcat(0, 0, T)
        R_x_T = ca.MX.sym('R_x_T', 3)
        R_x_T = ca.mtimes(R_b2w,T_aux)

        rhs = ca.vertcat(
            v_x,
            v_y,
            v_z,
            R_x_T[0] - (Ax * v_x / m),
            R_x_T[1] - (Ay * v_y / m),
            R_x_T[2] - g - (Az * v_z / m), 
            (1 / tau_phi) * (K_phi * phi_ref - phi),
            (1 / tau_theta) *  (K_theta * theta_ref - theta),
            (1 / tau_psi) *  (K_psi * psi_ref - psi)
        )


        # Define the differential equation function (mapping)
        f = ca.Function('f', [states, controls], [rhs], ['x', 'u'], ['dx/dt'])

        x_next = self.eulerforward(f(states, controls), states)

        FF = ca.Function('FF', [states, controls], [x_next], ['x_k', 'u_k'], ['x_k+1'])

        return FF
    
    def number_states(self):
        return self.n_states
    
    def number_controls(self):
        return self.n_controls