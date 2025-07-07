from math import cos, sin
import numpy as np
import matplotlib.pyplot as plt

class Quadrotor():
    def __init__(self, uav_num, x=0, y=0, z=0, roll=0, pitch=0, yaw=0, size=1, show_animation=True): #0.25
        self.p1 = np.array([size / 2, 0, 0, 1]).T
        self.p2 = np.array([-size / 2, 0, 0, 1]).T
        self.p3 = np.array([0, size / 2, 0, 1]).T
        self.p4 = np.array([0, -size / 2, 0, 1]).T

        self.uav_num = uav_num + 1 #add target

        self.uavs = []
        for i in range(self.uav_num):
            self.uavs.append(UAV_Data())

        self.show_animation = show_animation

        if self.show_animation:
            plt.ion()
            fig = plt.figure()
            # for stopping simulation with the esc key.
            fig.canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

            self.ax = fig.add_subplot(111, projection='3d')

        # for uav_i in range(self.uav_num):
        #     self.update_pose(x, y, z, roll, pitch, yaw, uav_i, False, False)

    def simulation_info(self, iterations, positions, euler_angles, target_positions):
        self.iterations = iterations
        self.positions = positions
        self.euler_angles = euler_angles
        self.target_positions = target_positions
        self.plotter()
        # self.init_plot()
        # self.process_info()
        # self.run_simulation()

    def update_pose(self, x, y, z, roll, pitch, yaw, uav_i, clean_up, pause, color):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.uavs[uav_i].add_data(x, y, z)

        if self.show_animation:
            self.plot(clean_up, pause, uav_i, color)

    def transformation_matrix(self):
        x = self.x
        y = self.y
        z = self.z
        roll = self.roll
        pitch = self.pitch
        yaw = self.yaw
        return np.array(
            [[cos(yaw) * cos(pitch), -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll), sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll), x],
             [sin(yaw) * cos(pitch), cos(yaw) * cos(roll) + sin(yaw) * sin(pitch)
              * sin(roll), -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll), y],
             [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(roll), z]
             ])

    def plot(self, clean_up, pause, uav_i, color):  # pragma: no cover
        T = self.transformation_matrix()

        p1_t = np.matmul(T, self.p1)
        p2_t = np.matmul(T, self.p2)
        p3_t = np.matmul(T, self.p3)
        p4_t = np.matmul(T, self.p4)

        if(clean_up):
            plt.cla()

        # self.ax.plot([p1_t[0], p2_t[0], p3_t[0], p4_t[0]],
        #              [p1_t[1], p2_t[1], p3_t[1], p4_t[1]],
        #              [p1_t[2], p2_t[2], p3_t[2], p4_t[2]], 'k.')
        self.colored_axes_1(p1_t, p2_t, p3_t, p4_t, color, '.')

        # self.ax.plot([p1_t[0], p2_t[0]], [p1_t[1], p2_t[1]],
        #              [p1_t[2], p2_t[2]], 'r-')
        self.colored_axes_2(p1_t, p2_t, p3_t, p4_t, color, '-')

        # self.ax.plot([p3_t[0], p4_t[0]], [p3_t[1], p4_t[1]],
        #              [p3_t[2], p4_t[2]], 'r-')
        self.colored_axes_3(p1_t, p2_t, p3_t, p4_t, color, '-')

        # self.ax.plot(self.uavs[uav_i].x_data, self.uavs[uav_i].y_data, self.uavs[uav_i].z_data, 'b:')
        self.colored_axes_4(p1_t, p2_t, p3_t, p4_t, color, ':', uav_i)

        plt.xlim(-2, 40)
        plt.ylim(-2, 40)
        self.ax.set_zlim(0, 5)

        if(pause):
            plt.pause(0.1)

    def colored_axes_1(self, p1_t, p2_t, p3_t, p4_t, color, line_type):
        color = color + line_type
        self.ax.plot([p1_t[0], p2_t[0], p3_t[0], p4_t[0]],
                     [p1_t[1], p2_t[1], p3_t[1], p4_t[1]],
                     [p1_t[2], p2_t[2], p3_t[2], p4_t[2]], color)
        
    def colored_axes_2(self, p1_t, p2_t, p3_t, p4_t, color, line_type):
        color = color + line_type
        self.ax.plot([p1_t[0], p2_t[0]], [p1_t[1], p2_t[1]],
                     [p1_t[2], p2_t[2]], color)

    def colored_axes_3(self, p1_t, p2_t, p3_t, p4_t, color, line_type):
        color = color + line_type
        self.ax.plot([p3_t[0], p4_t[0]], [p3_t[1], p4_t[1]],
                     [p3_t[2], p4_t[2]], color)
        
    def colored_axes_4(self, p1_t, p2_t, p3_t, p4_t, color, line_type, uav_i):
        color = color + line_type
        self.ax.plot(self.uavs[uav_i].x_data, self.uavs[uav_i].y_data, self.uavs[uav_i].z_data, color)


    def plotter(self):
        input()
        color_vec = ['r', 'b', 'g', 'y']
        for i in range(self.iterations):
            self.update_pose(self.target_positions[0, i], self.target_positions[1, i], self.target_positions[2, i], 0, 0, 0, 0,True, False, color_vec[0])
            aux = 1
            for uav_i in range(0, 3*(self.uav_num-1), 3):
                if(uav_i == (3*(self.uav_num-1)-3)):
                    self.update_pose(self.positions[uav_i+0, i], self.positions[uav_i+1, i], self.positions[uav_i+2, i],
                         self.euler_angles[uav_i+0, i], self.euler_angles[uav_i+1, i], self.euler_angles[uav_i+2, i], aux, False, True, color_vec[aux])
                else:
                    self.update_pose(self.positions[uav_i+0, i], self.positions[uav_i+1, i], self.positions[uav_i+2, i],
                         self.euler_angles[uav_i+0, i], self.euler_angles[uav_i+1, i], self.euler_angles[uav_i+2, i], aux, False, False, color_vec[aux])
                aux = aux + 1
        input()


class UAV_Data():

    def __init__(self):
        self.x_data = []
        self.y_data = []
        self.z_data = []

    def add_data(self, val1, val2, val3):
        self.x_data.append(val1)
        self.y_data.append(val2)
        self.z_data.append(val3)

