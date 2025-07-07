import casadi as ca
import numpy as np
from math import pi, cos, sin
from matplotlib import pyplot as plt


class angular_optimization:

    # Initialize angular optimization class, setting the horizon and uav number, plus calling functions forn creating the ocp and solver
    def __init__(self, N_horizon, uav_num):
        self.N_horizon = N_horizon
        self.uav_num = uav_num

        self.initialize_optimization()
        self.define_ocp()
        self.solver_parameters()
        self.counter = 0

    # Initialize Optimization Problem
    def initialize_optimization(self):
        self.opti = ca.Opti()

        self.alpha = self.opti.variable(self.uav_num)#, self.N_horizon + 1)

        self.theta = self.opti.parameter(self.uav_num, self.N_horizon + 1)

        self.previous_alpha = self.opti.parameter(1)

    # Receive UAV positions over time horizon
    def set_parameters(self, uav_positions, target_positions):

        yaw_array = np.zeros((self.uav_num, self.N_horizon + 1))

        self.uav_positions = uav_positions
        self.target_positions = target_positions

        for k in range(self.N_horizon + 1):
            for uav_i in range(self.uav_num):
                relative_position = np.array([(target_positions[0, k] - uav_positions[0, k, uav_i]), (target_positions[1, k] - uav_positions[1, k, uav_i])])
                angle = np.arctan2(relative_position[1], relative_position[0])
                yaw_array[uav_i, k] = angle
                
        yaw_array = np.unwrap(yaw_array)
        # print(yaw_array)
        # input()
        self.opti.set_value(self.theta, yaw_array)
        self.opti.set_initial(self.alpha, yaw_array[:, 0])
        self.array_do_yaw = yaw_array
        if(self.counter == 0):
            self.opti.set_value(self.previous_alpha, yaw_array[0, 0])
            self.counter = self.counter + 1

    # Define OCP cost function and constraints
    def define_ocp(self):

        # Cost Function
        obj = 0
        epsilon = 1e-6
        for k in range(self.N_horizon + 1):
            for uav_i in range(0, self.uav_num, 1):
                # ang_dist = self.theta[uav_i, k] - self.alpha[uav_i]#ca.fmod(self.theta[uav_i, k] - self.alpha[uav_i] + ca.pi, 2 * ca.pi) - ca.pi
                ang_dist = ca.fmod(self.theta[uav_i, k] - self.alpha[uav_i] + ca.pi, 2 * ca.pi) - ca.pi
                obj = obj + (ang_dist ** 2)
                # for uav_j in range(uav_i + 1, self.uav_num, 1):
                #     value = ca.sin(self.theta[k, uav_i] - self.theta[k, uav_j])
                #     obj = obj + ((1 / value) ** 2)

        for uav_i in range(0, self.uav_num, 1):
                for uav_j in range(uav_i + 1, self.uav_num, 1):
                    value = ca.sin(self.wrap_angle(self.alpha[uav_i] - self.alpha[uav_j])) ** 2 + epsilon
                    obj = obj + 1000 * (1/value)
                    # self.opti.subject_to((self.wrap_angle(self.alpha[uav_i] - self.alpha[uav_j]) ** 2) > 1)
                    # self.opti.subject_to((ca.sin(self.alpha[uav_i] - self.alpha[uav_j])) == (ca.sin(ca.pi/3)))


        # Constraints
        for uav_i in range(self.uav_num):
            self.opti.subject_to(self.alpha[uav_i] < (50*ca.pi))
            self.opti.subject_to(self.alpha[uav_i] > (-50*ca.pi))
        # self.opti.subject_to((self.wrap_angle(self.alpha[0] - self.previous_alpha) ** 2) < 0.5)

        self.opti.minimize(obj)


    # Helper function for angle wrapping between [-pi, pi]
    def wrap_angle(self, angle):
        return ca.fmod(angle + ca.pi, 2 * ca.pi) - ca.pi

    # Set solver parameters  
    def solver_parameters(self):
        # Set the solver options
        options = {
            'print_time': False,
            'expand': True,  # Expand makes function evaluations faster but requires more memory
            'ipopt': {
                'print_level': 0,
                # 'tol': 5e-1,
                # 'dual_inf_tol': 5.0,
                # 'constr_viol_tol': 1e-1,
                # 'compl_inf_tol': 1e-1,
                # 'acceptable_tol': 1e-2,
                # 'acceptable_constr_viol_tol': 0.01,
                # 'acceptable_dual_inf_tol': 1e10,
                # 'acceptable_compl_inf_tol': 0.01,
                # 'acceptable_obj_change_tol': 1e20,
                # 'diverging_iterates_tol': 1e20,
                # 'warm_start_bound_push': 1e-4,
                # 'warm_start_bound_frac': 1e-4,
                # 'warm_start_slack_bound_frac': 1e-4,
                # 'warm_start_slack_bound_push': 1e-4,
                # 'warm_start_mult_bound_push': 1e-4,
            },
            'verbose': False,
        }
        
        # Set the solver with the options
        self.opti.solver('ipopt', options)

    # Solve optimization problem
    def solve(self, uav_i = 0):
        # print("SOLVING ANGULAR OPTIMIZATION")
        try:
            self.sol = self.opti.solve()
        except RuntimeError as e:
            # Handle solver failure
            print("Solver failed with error:", e)
            print(self.array_do_yaw)
        self.warm_start()
        # Assume received uav i starts at 0 index
        value = ((self.sol.value(self.alpha[uav_i]) + pi) % (2 * pi)) - pi

        self.opti.set_value(self.previous_alpha, self.sol.value(self.alpha[uav_i]))
        # print("VALOR: " + str(value))
        return value
        
    def warm_start(self,):
        self.opti.set_initial(self.alpha,self.sol.value(self.alpha))
        # print("------------------------------")
        # self.opti.solve()

    def plotter(self):
        # Assumes 3 UAVs
        time_stamp = 0 #self.N_horizon
        # print("angles init: ")
        # print(self.sol.value(self.alpha[0, time_stamp]))
        # print(self.sol.value(self.alpha[1, time_stamp]))
        # print(self.sol.value(self.alpha[2, time_stamp]))
        ang1 = ((self.sol.value(self.alpha[0, time_stamp]) + np.pi) % (2*np.pi))-np.pi
        ang2 = ((self.sol.value(self.alpha[1, time_stamp]) + np.pi) % (2*np.pi))-np.pi
        ang3 = ((self.sol.value(self.alpha[2, time_stamp]) + np.pi) % (2*np.pi))-np.pi
        print("angles 1-2-3: ")
        print(ang1)
        print(ang2)
        print(ang3)
        plt.figure()
        # plt.scatter(point1[0], point1[1], color='r', marker="*", label = "optimal yaw")
        # plt.scatter(point2[0], point2[1], color='g', marker="*")
        # plt.scatter(point3[0], point3[1], color='b', marker="*")
        plt.scatter(self.target_positions[0, time_stamp], self.target_positions[1, time_stamp], color='r', marker=".")
        # for uav_i in range(self.uav_num):
        plt.scatter(self.uav_positions[0, :, 0], self.uav_positions[1, :, 0], color='b', marker="*")
        plt.scatter(self.uav_positions[0, :, 1], self.uav_positions[1, :, 1], color='g', marker="*")
        plt.scatter(self.uav_positions[0, :, 2], self.uav_positions[1, :, 2], color='r', marker="*")

        target = [self.target_positions[0, time_stamp], self.target_positions[1, time_stamp]]
        sup_ang1 = -((((np.pi - ang1) + np.pi) % (2*np.pi))-np.pi)
        sup_ang2 = -((((np.pi - ang2) + np.pi) % (2*np.pi))-np.pi)
        sup_ang3 = -((((np.pi - ang3) + np.pi) % (2*np.pi))-np.pi)
        # print("SUPP ANG: ")
        # print(sup_ang1)
        # print(sup_ang2)
        # print(sup_ang3)
        pt1 = [self.target_positions[0, time_stamp] + 10 * cos(sup_ang1), self.target_positions[1, time_stamp] + 10 * sin(sup_ang1)]
        pt2 = [self.target_positions[0, time_stamp] + 10 * cos(sup_ang2), self.target_positions[1, time_stamp] + 10 * sin(sup_ang2)]
        pt3 = [self.target_positions[0, time_stamp] + 10 * cos(sup_ang3), self.target_positions[1, time_stamp] + 10 * sin(sup_ang3)]
        
        x1 = [target[0], pt1[0]]
        x2 = [target[0], pt2[0]]
        x3 = [target[0], pt3[0]]

        y1 = [target[1], pt1[1]]
        y2 = [target[1], pt2[1]]
        y3 = [target[1], pt3[1]]

        plt.plot(x1, y1, 'bo', linestyle= "--")
        plt.plot(x2, y2, 'go', linestyle= "--")
        plt.plot(x3, y3, 'ro', linestyle= "--")

        plt.show(block = False)
        print("ANGLES: ")
        print(np.degrees(ang1))
        print(np.degrees(ang2))
        print(np.degrees(ang3))
        
        print("1 e 2: " + str(np.degrees((((ang1-ang2) + np.pi) % (2*np.pi))-np.pi)))
        print("1 e 3: " + str(np.degrees((((ang1-ang3) + np.pi) % (2*np.pi))-np.pi)))
        print("2 e 3: " + str(np.degrees((((ang2-ang3) + np.pi) % (2*np.pi))-np.pi)))
        input()


def main():
    uav_1 = np.array([[-29.3015, -29.30014, -29.2978739, -29.31117543, -29.3405131,
                       -29.37827052, -29.41498858, -29.44137162, -29.44847727, -29.42739853,
                       -29.36889251, -29.26307813, -29.09923779, -28.86572638, -28.54993913,
                       -28.1382466,  -27.61645169],
            [-15.034,      -15.0313,     -15.02937157, -15.02889373, -15.02991781,
            -15.03215297, -15.03522462, -15.03877116, -15.04246783, -15.04602983,
            -15.0492121,  -15.05180824, -15.05364675, -15.0545838,  -15.05449275,
            -15.05325185, -15.05074011],
            [4.5386,     4.5328,    4.47469883, 4.45783143, 4.49383446, 4.56407295,
            4.64799206, 4.73129628, 4.80619366, 4.86904043, 4.91774662, 4.94988195,
            4.96178839, 4.9487832,  4.90647149, 4.83307801, 4.73233872]])
    uav_2 = np.array([[-18.0179,     -18.01688,    -18.0163152,  -18.01596876, -18.01597952,
            -18.01674602, -18.01889598, -18.02325514, -18.03080071, -18.04261783,
            -18.05985667, -18.08369229, -18.11528752, -18.15575795, -18.20613721,
            -18.26733993, -18.34011932],
            [-11.9684,     -11.96958,    -11.96897793, -11.96751468, -11.96584998,
            -11.96453644, -11.9640999,  -11.96506262, -11.96794111, -11.97324384,
            -11.98146719, -11.99309051, -12.00857057, -12.02833514, -12.05277509,
            -12.08223421, -12.11699541],
            [4.5386,     4.52968,    4.61789266, 4.73813487, 4.85638326, 4.95989272,
            5.04755489, 5.12145289, 5.18357673, 5.23580518, 5.27989057, 5.31744612,
            5.34993471, 5.37865663, 5.40473382, 5.42908652, 5.45239705]])
    uav_3 = np.array([[-17.9829,     -17.9844,     -17.98604905, -17.98782622, -17.98998859,
            -17.99300362, -17.99753762, -18.00443181, -18.01466052, -18.02929281,
            -18.04945403, -18.07628935, -18.11092973, -18.15445935, -18.20788289,
            -18.27209022, -18.34781528],
            [-17.9704,     -17.97066,    -17.97053849, -17.96992553, -17.96865158,
            -17.96648319, -17.96311974, -17.95820446, -17.95133966, -17.94209687,
            -17.9300283,  -17.9146786,  -17.8955967,  -17.87234795, -17.84452711,
            -17.81177301, -17.7737862 ],
            [4.5579,     4.54828,    4.63249117, 4.74770057, 4.86097124, 4.95986986,
            5.04323091, 5.11308746, 5.17138541, 5.21996552, 5.260548,   5.29471968,
            5.32392229, 5.3494398,  5.37238216, 5.39366148, 5.41395526]])
    target_positions = np.array([[-14.83422593, -14.83422593, -14.83422593, -14.83422593, -14.83422593,
            -14.83422593, -14.83422593, -14.83422593, -14.83422593, -14.83422593,
            -14.83422593, -14.83422593, -14.83422593, -14.83422593, -14.83422593,
            -14.83422593, -14.83422593, -14.83422593, -14.83422593, -14.83422593,
            -14.83422593, -14.83422593, -14.83422593, -14.83422593, -14.83422593,
            -14.83422593],
            [  7.71411369,   7.71413369,   7.71415369,   7.71417369,   7.71419369,
            7.71421369,   7.71423369,   7.71425369,   7.71427369,   7.71429369,
            7.71431369,   7.71433369,   7.71435369,   7.71437369,   7.71439369,
            7.71441369,   7.71443369,   7.71445369,   7.71447369,   7.71449369,
            7.71451369,   7.71453369,   7.71455369,   7.71457369,   7.71459369,
            7.71461369]])
    
    target_positions = np.array([[-25,-25,-25,-25,-25,
                                  -25,-25,-25,-25,-25,
                                  -25,-25,-25,-25,-25,
                                  -25,-25],
                                  [-15,-15,-15,-15,-15,
                                   -15,-15,-15,-15,-15,
                                   -15,-15,-15,-15,-15,
                                   -15,-15]])
    uav_positions = np.stack((uav_1, uav_2, uav_3), axis=-1)

    ocp = angular_optimization(16, 3)
    ocp.set_parameters(uav_positions, target_positions)
    ocp.solve(0)
    ocp.plotter()

if __name__ == "__main__":
    main() 

