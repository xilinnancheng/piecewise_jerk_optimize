import math
import numpy
import osqp
from scipy import sparse
from matplotlib import pyplot as plt

class PieceJerkPathOptimize:
    def __init__(self, ref_path_s, ref_path_upper_l, ref_path_lower_l, ref_path_kappa, w_l, w_dl, w_ddl, w_dddl, w_ref_l, wheel_base, max_delta, max_delta_rate):
        self.num_of_point = len(ref_path_s)
        self.num_of_variable = 3 * self.num_of_point
        self.ref_path_s = ref_path_s
        self.ref_path_upper_l = ref_path_upper_l
        self.ref_path_lower_l = ref_path_lower_l
        self.ref_path_kappa = ref_path_kappa
        self.w_l = w_l
        self.w_dl = w_dl
        self.w_ddl = w_ddl
        self.w_dddl = w_dddl
        self.w_ref_l = w_ref_l
        self.wheel_base = wheel_base
        self.max_delta = max_delta
        self.max_delta_rate = max_delta_rate
        self.s_step = []
        self.s_step_square = []
        self.ref_path_l = []
        for i in range(len(ref_path_s) - 1):
            delta_s = ref_path_s[i+1] - ref_path_s[i]
            self.s_step.append(delta_s)
            self.s_step_square.append(pow(delta_s, 2))
        
        for i in range(len(ref_path_s)):
            self.ref_path_l.append(0.5 * (ref_path_upper_l[i] + ref_path_lower_l[i]))
        
        self.dl_upper_bound = []
        self.dl_lower_bound = []
        self.ddl_upper_bound = []
        self.ddl_lower_bound = []
        self.solution_l = []
        self.solution_dl = []
        self.solution_ddl = []
        self.solution_dddl = []
        
    def CalculateQ(self):
        row = []
        col = []
        data = []

        # l cost
        for i in range(self.num_of_point):
            row.append(i)
            col.append(i)
            data.append(2.0 * self.w_l)

        # dl cost
        for i in range(self.num_of_point):
            row.append(self.num_of_point + i)
            col.append(self.num_of_point + i)
            data.append(2.0 * self.w_dl)

        # ddl cost
        for i in range(self.num_of_point):
            row.append(2 * self.num_of_point + i)
            col.append(2 * self.num_of_point + i)
            data.append(2.0 * self.w_ddl)
        
        # dddl cost
        for i in range(self.num_of_point - 1):
            row.append(2 * self.num_of_point + i)
            col.append(2 * self.num_of_point + i)
            data.append(data[2 * self.num_of_point + i] + self.w_dddl /self.s_step_square[i])
            row.append(2 * self.num_of_point + i + 1)
            col.append(2 * self.num_of_point + i + 1)
            data.append(data[2 * self.num_of_point + i + 1] + self.w_dddl /self.s_step_square[i])
            row.append(2 * self.num_of_point + i + 1)
            col.append(2 * self.num_of_point + i)
            data.append(-2.0 * self.w_dddl /self.s_step_square[i])
        
        row.append(2 * self.num_of_point + self.num_of_point - 1)
        col.append(2 * self.num_of_point + self.num_of_point - 1)
        data.append(data[2 * self.num_of_point + self.num_of_point - 1] /
            + self.w_dddl /self.s_step_square[self.num_of_point - 2])

        # l_ref cost
        for i in range(self.num_of_point):
            data[i] += 2.0 * self.w_ref_l
        
        Q = sparse.csc_matrix((data, (row, col)), shape=(
            self.num_of_variable, self.num_of_variable))
        
        return Q

    def CalculateP(self):
        P = []
        for i in range(self.num_of_point):
            P.append(-2.0 * self.ref_path_l[i] * self.w_ref_l)

        for i in range(2 * self.num_of_point):
            P.append(0.0)
        
        P = numpy.array(P)
        return P

    def CalculateAffineConstraint(self):
        row = []
        col = []
        data = []

        # l constrait.
        for i in range(self.num_of_point):
            row.append(i)
            col.append(i)
            data.append(1.0)
        
        # dl constraint.
        for i in range(self.num_of_point):
            row.append(self.num_of_point + i)
            col.append(self.num_of_point + i)
            data.append(1)

        # ddl constraint.
        for i in range(self.num_of_point):
            row.append(2 * self.num_of_point + i)
            col.append(2 * self.num_of_point + i)
            data.append(1)
        
        # dddl constraint.
        for i in range(self.num_of_point - 1):
            row.append(3 * self.num_of_point + i)
            col.append(2 * self.num_of_point + i)
            data.append(-1.0)
            row.append(3 * self.num_of_point + i)
            col.append(2 * self.num_of_point + i + 1)
            data.append(1.0)

        # start/end l limit
        row.append(4 * self.num_of_point - 1)
        col.append(0)
        data.append(1)

        row.append(4 * self.num_of_point)
        col.append(self.num_of_point - 1)
        data.append(1)

        # ddl consistency
        for i in range(self.num_of_point - 1):
            row.append(4 * self.num_of_point + 1 + i)
            col.append(self.num_of_point + i)
            data.append(1)

            row.append(4 * self.num_of_point + 1 + i)
            col.append(self.num_of_point + i + 1)
            data.append(-1)

            row.append(4 * self.num_of_point + 1 + i)
            col.append(2 * self.num_of_point + i)
            data.append(0.5 * self.s_step[i])

            row.append(4 * self.num_of_point + 1 + i)
            col.append(2 * self.num_of_point + i + 1)
            data.append(0.5 * self.s_step[i])

        # dl consistency
        for i in range(self.num_of_point - 1):
            row.append(5 * self.num_of_point + i)
            col.append(i)
            data.append(1)

            row.append(5 * self.num_of_point + i)
            col.append(i + 1)
            data.append(-1)

            row.append(5 * self.num_of_point + i)
            col.append(self.num_of_point + i)
            data.append(self.s_step[i])

            row.append(5 * self.num_of_point + i)
            col.append(self.num_of_point + i + 1)
            data.append(self.s_step[i])

            row.append(5 * self.num_of_point + i)
            col.append(2 * self.num_of_point + i)
            data.append(1.0/3.0 * self.s_step[i] * self.s_step[i])

            row.append(5 * self.num_of_point + i)
            col.append(2 * self.num_of_point + i + 1)
            data.append(1.0/6.0 * self.s_step[i] * self.s_step[i])

        A = sparse.csc_matrix((data, (row, col)), shape=(
            6 * self.num_of_point - 1, self.num_of_variable))

        numpy.set_printoptions(linewidth=numpy.inf)
        lb = []
        ub = []
        # l
        for i in range(self.num_of_point):
            lb.append(self.ref_path_lower_l[i])
            ub.append(self.ref_path_upper_l[i])
        
        # dl = (1 - kappa * l) * tan(delta_theta)
        for i in range(self.num_of_point):
            self.dl_lower_bound.append(-math.tan(numpy.deg2rad(30)))
            self.dl_upper_bound.append(math.tan(numpy.deg2rad(30)))
            lb.append(-math.tan(numpy.deg2rad(30)))
            ub.append(math.tan(numpy.deg2rad(30)))
        
        # ddl = tan(max_delta)/wheel_base - k_r
        for i in range(self.num_of_point):
            self.ddl_lower_bound.append(-(math.tan(self.max_delta)/self.wheel_base - 0.0))
            self.ddl_upper_bound.append(math.tan(self.max_delta)/self.wheel_base - 0.0)
            lb.append(-(math.tan(self.max_delta)/self.wheel_base - 0.0))
            ub.append(math.tan(self.max_delta)/self.wheel_base - 0.0)
        
        # dddl
        for i in range(self.num_of_point - 1):
            lb.append(-self.max_delta_rate / self.wheel_base / 2.0)
            ub.append(self.max_delta_rate / self.wheel_base / 2.0)

        ## dl consistency
        for i in range(self.num_of_point - 1):
            lb.append(-10e-5)
            ub.append(10e-5)
        
        ## l consistency
        for i in range(self.num_of_point - 1):
            lb.append(-10e-5)
            ub.append(10e-5)

        # start/end l
        lb.append(self.ref_path_l[0] - 1e-5)
        ub.append(self.ref_path_l[0] + 1e-5)
        lb.append(self.ref_path_l[self.num_of_point-1] - 1e-5)
        ub.append(self.ref_path_l[self.num_of_point-1] + 1e-5)

        return A,lb,ub

    def Optimize(self):
        Q = self.CalculateQ()
        P = self.CalculateP()

        A, lb, ub = self.CalculateAffineConstraint()

        prob = osqp.OSQP()
        prob.setup(Q, P, A, lb, ub, polish=True, eps_abs=1e-5, eps_rel=1e-5,
                   eps_prim_inf=1e-5, eps_dual_inf=1e-5, verbose=True)

        var_warm_start = numpy.array(self.ref_path_l + [0.0 for n in range(2 * self.num_of_point)])
        prob.warm_start(x = var_warm_start)
        res = prob.solve()

        self.solution_l = res.x[0:self.num_of_point]
        self.solution_dl = res.x[self.num_of_point:2 *self.num_of_point]
        self.solution_ddl = res.x[2 *self.num_of_point:3 * self.num_of_point]
        
        plt.subplot(3, 1, 1)
        plt.plot(self.ref_path_s, self.ref_path_upper_l,'r',marker="x")
        plt.plot(self.ref_path_s, self.ref_path_lower_l,'r',marker="x")
        plt.plot(self.ref_path_s, self.ref_path_l,'g',marker="x")
        plt.plot(self.ref_path_s, self.solution_l,'b')
        plt.grid()
        plt.legend(["upper_bound","lower_bound","ref_path_l","solution_l"])
        plt.title("PieceWise Jerk Path optimization")

        plt.subplot(3, 1, 2)
        plt.plot(self.ref_path_s, self.solution_dl,'b')
        plt.plot(self.ref_path_s, self.dl_upper_bound,'r')
        plt.plot(self.ref_path_s, self.dl_lower_bound,'r')
        plt.grid()
        plt.legend(["solution_dl","dl_bound"])

        plt.subplot(3, 1, 3)
        plt.plot(self.ref_path_s, self.solution_ddl,'b')
        plt.plot(self.ref_path_s, self.ddl_upper_bound,'r')
        plt.plot(self.ref_path_s, self.ddl_lower_bound,'r')
        plt.legend(["solution_ddl","ddl_bound"])
        plt.grid()
        plt.show()
        
if __name__ == "__main__":
    # parameter
    w_l = 5
    w_dl = 10
    w_ddl = 30
    w_dddl = 40
    w_ref_l = 100
    wheel_base = 2.8
    max_delta = numpy.deg2rad(29.375)
    max_delta_rate = numpy.deg2rad(31.25)

    # Reference path
    s_step = 0.5
    ref_path_length = 10.0
    ref_path_s = numpy.arange(0,ref_path_length,s_step)
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
    ref_path_boundary_upper_l[5] = 5.0
    ref_path_boundary_lower_l[5] = 1.0
    ref_path_boundary_upper_l[6] = 5.0
    ref_path_boundary_lower_l[6] = 1.0
    ref_path_boundary_upper_l[7] = 5.0
    ref_path_boundary_lower_l[7] = 1.0
    ref_path_boundary_upper_l[8] = 5.0
    ref_path_boundary_lower_l[8] = 1.0
    ref_path_boundary_upper_l[14] = 0.0
    ref_path_boundary_lower_l[14] = -4.0
    ref_path_boundary_upper_l[15] = 0.0
    ref_path_boundary_lower_l[15] = -4.0
    
    path_optimize = PieceJerkPathOptimize(ref_path_s,ref_path_boundary_upper_l,\
                                          ref_path_boundary_lower_l, ref_path_kappa, w_l,\
                                          w_dl, w_ddl, w_dddl, w_ref_l, wheel_base, max_delta,\
                                          max_delta_rate)
    path_optimize.Optimize()