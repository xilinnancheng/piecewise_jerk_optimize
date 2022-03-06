import math
import numpy
import osqp
from scipy import sparse
from matplotlib import pyplot as plt
from piecewise_jerk_optimize import PieceJerkOptimize

class PieceJerkSpeedOptimize(PieceJerkOptimize):
    def __init__(self, num_of_point):
        super().__init__(num_of_point)
        self.ref_path_t = []
        
    def SetReferencePathT(self, ref_path_t):
        self.ref_path_t = ref_path_t

    def CalculateQ(self):
        row = []
        col = []
        data = []

        # s cost
        for i in range(self.num_of_point):
            row.append(i)
            col.append(i)
            data.append(2.0 * self.w_x)

        # ds cost
        for i in range(self.num_of_point):
            row.append(self.num_of_point + i)
            col.append(self.num_of_point + i)
            data.append(2.0 * self.w_dx)

        # dds cost
        for i in range(self.num_of_point):
            row.append(2 * self.num_of_point + i)
            col.append(2 * self.num_of_point + i)
            data.append(2.0 * self.w_ddx)
        
        # ddds cost
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

        # s_ref cost
        for i in range(self.num_of_point):
            data[i] += 2.0 * self.w_ref_x
        
        Q = sparse.csc_matrix((data, (row, col)), shape=(
            self.num_of_variable, self.num_of_variable))
        
        return Q

    def VizResult(self):        
        plt.subplot(4, 1, 1)
        plt.plot(self.ref_path_t, self.x_upper_bound,'r',marker="x")
        plt.plot(self.ref_path_t, self.x_lower_bound,'r',marker="x")
        plt.plot(self.ref_path_t, self.ref_x,'g',marker="x")
        plt.plot(self.ref_path_t, self.solution_x,'b')
        plt.grid()
        plt.legend(["upper_bound","lower_bound","ref_path_s","solution_s"])
        plt.title("PieceWise Jerk Path optimization")

        plt.subplot(4, 1, 2)
        plt.plot(self.ref_path_t, self.solution_dx,'b')
        plt.plot(self.ref_path_t, self.dx_upper_bound,'r')
        plt.plot(self.ref_path_t, self.dx_lower_bound,'r')
        plt.grid()
        plt.legend(["solution_ds","ds_bound"])

        plt.subplot(4, 1, 3)
        plt.plot(self.ref_path_t, self.solution_ddx,'b')
        plt.plot(self.ref_path_t, self.ddx_upper_bound,'r')
        plt.plot(self.ref_path_t, self.ddx_lower_bound,'r')
        plt.legend(["solution_dds","dds_bound"])
        plt.grid()

        plt.subplot(4, 1, 4)
        plt.plot(self.ref_path_t, self.solution_dddx,'b')
        plt.plot(self.ref_path_t, self.dddx_upper_bound,'r')
        plt.plot(self.ref_path_t, self.dddx_lower_bound,'r')
        plt.legend(["solution_ddds","ddds_bound"])
        plt.grid()
        plt.show()
        
if __name__ == "__main__":
    # weight parameter
    w_s = 5
    w_ds = 10
    w_dds = 30
    w_ddds = 40
    w_ref_s = 100
    
    # car paramter
    max_jerk = 5.0
    min_jerk = -5.0
    max_acc = 1.0
    max_dec = -4.0
    max_v = 5.0
    min_v = 0.0

    # Reference path speed
    step = 1.0
    total_time = 10.0
    ref_path_length = 20.0
    ref_path_t = numpy.arange(0,total_time,step)
    if(ref_path_t[-1] != total_time):
        ref_path_t = numpy.append(ref_path_t,total_time)

    ref_path_size = len(ref_path_t)
    ref_path_t_step = [ref_path_t[i+1] - ref_path_t[i] for i in range(ref_path_size - 1)]
    ref_path_s = [ref_path_length for i in range(ref_path_size)]
    
    init_state = [0.0, 0.0, 0.0]
    end_state = [ref_path_length, 0.0, 0.0]
    
    # s bound
    s_lower_bound = [-1e-5 for i in range(ref_path_size)]
    s_upper_bound = [ref_path_length + 1e-5 for i in range(ref_path_size)]

    # ds bound
    ds_lower_bound = [min_v for i in range(ref_path_size)]
    ds_upper_bound = [max_v for i in range(ref_path_size)]

    # dds bound
    dds_lower_bound = [max_dec for i in range(ref_path_size)]
    dds_upper_bound = [max_acc for i in range(ref_path_size)]

    # ddds bound
    ddds_lower_bound = [min_jerk for i in range(ref_path_size)]
    ddds_upper_bound = [max_jerk for i in range(ref_path_size)]

    path_optimize = PieceJerkSpeedOptimize(len(ref_path_t))
    path_optimize.SetWeight(w_s, w_ds, w_dds, w_ddds, w_ref_s)
    path_optimize.SetReferencePathT(ref_path_t)
    path_optimize.SetReferenceX(ref_path_s)
    path_optimize.SetInitState(init_state)
    path_optimize.SetEndState(end_state)
    path_optimize.SetStep(ref_path_t_step)
    path_optimize.SetXBound(s_upper_bound, s_lower_bound)
    path_optimize.SetDXBound(ds_upper_bound, ds_lower_bound)
    path_optimize.SetDDXBound(dds_upper_bound, dds_lower_bound)
    path_optimize.SetDDDXBound(ddds_upper_bound, ddds_lower_bound)
    path_optimize.Optimize()
    path_optimize.VizResult()