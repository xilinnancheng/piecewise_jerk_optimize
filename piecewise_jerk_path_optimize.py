import math
import numpy
import osqp
from scipy import sparse
from matplotlib import pyplot as plt
from piecewise_jerk_optimize import PieceJerkOptimize

class PieceJerkPathOptimize(PieceJerkOptimize):
    def __init__(self, num_of_point):
        super().__init__(num_of_point)
        self.ref_path_s = []
        self.solution_theta = []
        self.solution_kappa = []
        self.solution_dkappa = []
        
    def SetReferencePathS(self, ref_path_s):
        self.ref_path_s = ref_path_s

    def CalculateQ(self):
        row = []
        col = []
        data = []

        # l cost
        for i in range(self.num_of_point):
            row.append(i)
            col.append(i)
            data.append(2.0 * self.w_x)

        # dl cost
        for i in range(self.num_of_point):
            row.append(self.num_of_point + i)
            col.append(self.num_of_point + i)
            data.append(2.0 * self.w_dx)

        # ddl cost
        for i in range(self.num_of_point):
            row.append(2 * self.num_of_point + i)
            col.append(2 * self.num_of_point + i)
            data.append(2.0 * self.w_ddx)
        
        # dddl cost
        for i in range(self.num_of_point - 1):
            row.append(2 * self.num_of_point + i)
            col.append(2 * self.num_of_point + i)
            data.append(data[2 * self.num_of_point + i] + self.w_dddx /self.step_square[i])
            row.append(2 * self.num_of_point + i + 1)
            col.append(2 * self.num_of_point + i + 1)
            data.append(data[2 * self.num_of_point + i + 1] + self.w_dddx /self.step_square[i])
            row.append(2 * self.num_of_point + i + 1)
            col.append(2 * self.num_of_point + i)
            data.append(-2.0 * self.w_dddx /self.step_square[i])
        
        row.append(2 * self.num_of_point + self.num_of_point - 1)
        col.append(2 * self.num_of_point + self.num_of_point - 1)
        data.append(data[2 * self.num_of_point + self.num_of_point - 1] /
            + self.w_dddx /self.step_square[self.num_of_point - 2])

        # l_ref cost
        for i in range(self.num_of_point):
            data[i] += 2.0 * self.w_ref_x
        
        Q = sparse.csc_matrix((data, (row, col)), shape=(
            self.num_of_variable, self.num_of_variable))
        
        return Q

    def VizResult(self):
        plt.figure(1)
        plt.subplot(4, 1, 1)
        plt.plot(self.ref_path_s, self.x_upper_bound,'r',marker="x")
        plt.plot(self.ref_path_s, self.x_lower_bound,'r',marker="x")
        plt.plot(self.ref_path_s, self.ref_x,'g',marker="x")
        plt.plot(self.ref_path_s, self.solution_x,'b')
        plt.grid()
        plt.legend(["upper_bound","lower_bound","ref_path_l","solution_l"])
        plt.title("PieceWise Jerk Path Optimization Solution")

        plt.subplot(4, 1, 2)
        plt.plot(self.ref_path_s, self.solution_dx,'b')
        plt.plot(self.ref_path_s, self.dx_upper_bound,'r')
        plt.plot(self.ref_path_s, self.dx_lower_bound,'r')
        plt.grid()
        plt.legend(["solution_dl","dl_bound"])

        plt.subplot(4, 1, 3)
        plt.plot(self.ref_path_s, self.solution_ddx,'b')
        plt.plot(self.ref_path_s, self.ddx_upper_bound,'r')
        plt.plot(self.ref_path_s, self.ddx_lower_bound,'r')
        plt.legend(["solution_ddl","ddl_bound"])
        plt.grid()

        plt.subplot(4, 1, 4)
        plt.plot(self.ref_path_s, self.solution_dddx,'b')
        plt.plot(self.ref_path_s, self.dddx_upper_bound,'r')
        plt.plot(self.ref_path_s, self.dddx_lower_bound,'r')
        plt.legend(["solution_dddl","dddl_bound"])
        plt.grid()
 
        plt.figure(2)
        for index in range(self.num_of_point):
            self.solution_theta.append(math.atan(self.solution_dx[index]/(1 - self.solution_x[index] * 0.0)) + 0.0)

            delta_theta = self.solution_theta[index] - 0.0
            cos_delta_theta = math.cos(delta_theta)
            cos_delta_theta_square = pow(cos_delta_theta, 2)
            cos_delta_theta_cubic = pow(cos_delta_theta, 3)
            sin_delta_theta = math.sin(delta_theta)

            self.solution_kappa.append(((self.solution_ddx[index] + (0.0 * self.solution_x[index] + 0.0 * self.solution_dx[index])*math.tan(delta_theta)) * cos_delta_theta_square / (1-0.0 * self.solution_x[index]) + 0.0) * cos_delta_theta/(1 - 0.0 * self.solution_x[index]))
            
            delta_kappa = self.solution_kappa[index] - 0.0
            self.solution_dkappa.append((self.solution_dddx[index] + \
                                         (0.0 * self.solution_x[index] + 2 * 0.0 * self.solution_dx[index] + 0.0 * self.solution_ddx[index]) * math.tan(delta_theta) + \
                                         (0.0 * self.solution_x[index] + 0.0 * self.solution_dx[index] * (delta_kappa)/cos_delta_theta_square) -\
                                         (-(0.0 * self.solution_x[index] + 0.0 * self.solution_dx[index])/cos_delta_theta_square + 2 * (1 - 0.0 * self.solution_x) * sin_delta_theta*delta_kappa/cos_delta_theta_cubic) * ((1-0.0 * self.solution_x[index]/cos_delta_theta) * self.solution_kappa[index] - 0.0) -\
                                         ((1 - 0.0 * self.solution_x[index]) / cos_delta_theta_square * ((sin_delta_theta * delta_kappa * (1 - 0.0 * self.solution_x)/cos_delta_theta_square - (0.0 * self.solution_x[index] + 0.0 * self.solution_dx[index])/cos_delta_theta) * self.solution_kappa[index] - 0.0))/ \
                                         (math.pow((1 - 0.0 * self.solution_x[index]),2)/cos_delta_theta_cubic)))
        
        plt.subplot(3, 1, 1)
        plt.plot(self.ref_path_s, self.solution_theta,'b')
        plt.grid()
        plt.legend(["solution_theta"])
        plt.title("PieceWise Jerk Path Optimization Solution")

        plt.subplot(3, 1, 2)
        plt.plot(self.ref_path_s, self.solution_kappa,'b')
        plt.legend(["solution_kappa"])
        plt.grid()

        plt.subplot(3, 1, 3)
        plt.plot(self.ref_path_s, self.solution_dkappa,'b')
        plt.legend(["solution_dkappa"])
        plt.grid()
        plt.show()

if __name__ == "__main__":
    # weight parameter
    w_l = 5
    w_dl = 10
    w_ddl = 30
    w_dddl = 40
    w_ref_l = 100
    
    # car paramter
    wheel_base = 2.8
    max_delta = numpy.deg2rad(29.375)
    max_delta_rate = numpy.deg2rad(31.25)

    # Reference path
    step = 0.5
    ref_path_length = 10.0
    ref_path_s = numpy.arange(0,ref_path_length,step)
    if(ref_path_s[-1] != ref_path_length):
        ref_path_s = numpy.append(ref_path_s,ref_path_length)

    ref_path_boundary_upper_l = numpy.array([])
    ref_path_boundary_lower_l = numpy.array([])
    ref_path_kappa = []
    for i in range(len(ref_path_s)):
        ref_path_boundary_upper_l = numpy.append(ref_path_boundary_upper_l,0.0 + 2.0)
        ref_path_boundary_lower_l = numpy.append(ref_path_boundary_lower_l,0.0 - 2.0)
        ref_path_kappa.append(0.0)

    ref_path_boundary_upper_l[1] = 3.0
    ref_path_boundary_lower_l[1] = -1.0
    ref_path_boundary_upper_l[5] = 3.0
    ref_path_boundary_lower_l[5] = 0.0
    ref_path_boundary_upper_l[6] = 3.0
    ref_path_boundary_lower_l[6] = 0.0
    ref_path_boundary_upper_l[7] = 3.0
    ref_path_boundary_lower_l[7] = 0.0
    ref_path_boundary_upper_l[8] = 3.0
    ref_path_boundary_lower_l[8] = 0.0
    ref_path_boundary_upper_l[14] = 2.0
    ref_path_boundary_lower_l[14] = -1.0
    ref_path_boundary_upper_l[15] = 2.0
    ref_path_boundary_lower_l[15] = -1.0

    ref_path_size = len(ref_path_s)
    ref_path_s_step = [ref_path_s[i+1] - ref_path_s[i] for i in range(ref_path_size - 1)]
    ref_path_l = [0.5 * (ref_path_boundary_lower_l[i] + ref_path_boundary_upper_l[i]) for i in range(ref_path_size)]
    
    init_state = [ref_path_l[0],0.3,0.1]
    end_state = [ref_path_l[-1],0.0,0.0]

    # bound
    l_lower_bound = ref_path_boundary_lower_l
    l_upper_bound = ref_path_boundary_upper_l

    # dl = (1 - kappa * l) * tan(delta_theta)
    dl_bound = math.tan(numpy.deg2rad(30))
    dl_lower_bound = [-dl_bound for i in range(ref_path_size)]
    dl_upper_bound = [dl_bound for i in range(ref_path_size)]

    # ddl = tan(max_delta)/wheel_base - k_r
    ddl_bound = (math.tan(max_delta)/wheel_base - 0.0)
    ddl_lower_bound = [-ddl_bound for i in range(ref_path_size)]
    ddl_upper_bound = [ddl_bound for i in range(ref_path_size)]

    # dddl
    dddl_bound = max_delta_rate / wheel_base / 2.0
    dddl_lower_bound = [-dddl_bound for i in range(ref_path_size)]
    dddl_upper_bound = [dddl_bound for i in range(ref_path_size)]

    path_optimize = PieceJerkPathOptimize(len(ref_path_s))
    path_optimize.SetWeight(w_l, w_dl, w_ddl, w_dddl, w_ref_l)
    path_optimize.SetReferencePathS(ref_path_s)
    path_optimize.SetReferenceX(ref_path_l)
    path_optimize.SetInitState(init_state)
    path_optimize.SetEndState(end_state)
    path_optimize.SetStep(ref_path_s_step)
    path_optimize.SetXBound(l_upper_bound, l_lower_bound)
    path_optimize.SetDXBound(dl_upper_bound, dl_lower_bound)
    path_optimize.SetDDXBound(ddl_upper_bound, ddl_lower_bound)
    path_optimize.SetDDDXBound(dddl_upper_bound, dddl_lower_bound)
    path_optimize.Optimize()
    path_optimize.VizResult()