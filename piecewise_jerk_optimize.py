import math
import numpy
import osqp
from scipy import sparse
from matplotlib import pyplot as plt

class PieceJerkOptimize:
    def __init__(self, num_of_point):
        self.num_of_point = num_of_point
        self.num_of_variable = 3 * num_of_point

        self.w_x = 0.0
        self.w_dx = 0.0
        self.w_ddx = 0.0
        self.w_dddx = 0.0
        self.w_ref_x = 0.0

        self.step = []
        self.step_square = []
        self.ref_x = []
        self.init_state = []
        self.end_state = []
        self.x_upper_bound = []
        self.x_lower_bound = []
        self.dx_upper_bound = []
        self.dx_lower_bound = []
        self.ddx_upper_bound = []
        self.ddx_lower_bound = []
        self.dddx_upper_bound = []
        self.dddx_lower_bound = []
        self.solution_x = []
        self.solution_dx = []
        self.solution_ddx = []
        self.solution_dddx = []
        pass

    def SetWeight(self, w_x, w_dx,w_ddx, w_dddx, w_ref_x):
        self.w_x = w_x
        self.w_dx = w_dx
        self.w_ddx = w_ddx
        self.w_dddx = w_dddx
        self.w_ref_x = w_ref_x

    def SetReferenceX(self, ref_x):
        self.ref_x = ref_x

    def SetStep(self, step):
        self.step = step
        self.step_square = [x for x in self.step]

    def SetXBound(self, upper_bound, lower_bound):
        self.x_lower_bound = lower_bound
        self.x_upper_bound = upper_bound

    def SetDXBound(self, upper_bound, lower_bound):
        self.dx_lower_bound = lower_bound
        self.dx_upper_bound = upper_bound

    def SetDDXBound(self, upper_bound, lower_bound):
        self.ddx_lower_bound = lower_bound
        self.ddx_upper_bound = upper_bound

    def SetDDDXBound(self, upper_bound, lower_bound):
        self.dddx_lower_bound = lower_bound
        self.dddx_upper_bound = upper_bound

    def SetInitState(self, init_state):
        self.init_state = init_state

    def SetEndState(self, end_state):
        self.end_state = end_state

    def CalculateQ(self):
        row = []
        col = []
        data = []
        return sparse.csc_matrix((data, (row, col)), shape=(
            self.num_of_variable, self.num_of_variable))

    def CalculateP(self):
        P = []
        for i in range(self.num_of_point):
            P.append(-2.0 * self.ref_x[i] * self.w_ref_x)

        for i in range(2 * self.num_of_point):
            P.append(0.0)
        
        P = numpy.array(P)
        return P

    def CalculateAffineConstraint(self):
        row = []
        col = []
        data = []

        # x constrait.
        for i in range(self.num_of_point):
            row.append(i)
            col.append(i)
            data.append(1.0)
        
        # dx constraint.
        for i in range(self.num_of_point):
            row.append(self.num_of_point + i)
            col.append(self.num_of_point + i)
            data.append(1)

        # ddx constraint.
        for i in range(self.num_of_point):
            row.append(2 * self.num_of_point + i)
            col.append(2 * self.num_of_point + i)
            data.append(1)
        
        # dddx constraint.
        for i in range(self.num_of_point - 1):
            row.append(3 * self.num_of_point + i)
            col.append(2 * self.num_of_point + i)
            data.append(-1.0)
            row.append(3 * self.num_of_point + i)
            col.append(2 * self.num_of_point + i + 1)
            data.append(1.0)

        # start/end x limit
        row.append(4 * self.num_of_point - 1)
        col.append(0)
        data.append(1)

        row.append(4 * self.num_of_point)
        col.append(self.num_of_point - 1)
        data.append(1)

        # start dx limit
        row.append(4 * self.num_of_point + 1)
        col.append(self.num_of_point)
        data.append(1)

        # start ddx limit
        row.append(4 * self.num_of_point + 2)
        col.append(2 * self.num_of_point)
        data.append(1)

        # ddx consistency
        for i in range(self.num_of_point - 1):
            row.append(4 * self.num_of_point + 3 + i)
            col.append(self.num_of_point + i)
            data.append(1)

            row.append(4 * self.num_of_point + 3 + i)
            col.append(self.num_of_point + i + 1)
            data.append(-1)

            row.append(4 * self.num_of_point + 3 + i)
            col.append(2 * self.num_of_point + i)
            data.append(0.5 * self.step[i])

            row.append(4 * self.num_of_point + 3 + i)
            col.append(2 * self.num_of_point + i + 1)
            data.append(0.5 * self.step[i])

        # dx consistency
        for i in range(self.num_of_point - 1):
            row.append(5 * self.num_of_point + 2 + i)
            col.append(i)
            data.append(1)

            row.append(5 * self.num_of_point + 2 + i)
            col.append(i + 1)
            data.append(-1)

            row.append(5 * self.num_of_point + 2 + i)
            col.append(self.num_of_point + i)
            data.append(self.step[i])

            row.append(5 * self.num_of_point + 2 + i)
            col.append(self.num_of_point + i + 1)
            data.append(self.step[i])

            row.append(5 * self.num_of_point + 2 + i)
            col.append(2 * self.num_of_point + i)
            data.append(1.0/3.0 * self.step_square[i])

            row.append(5 * self.num_of_point + 2 + i)
            col.append(2 * self.num_of_point + i + 1)
            data.append(1.0/6.0 * self.step_square[i])

        A = sparse.csc_matrix((data, (row, col)), shape=(
            6 * self.num_of_point + 1, self.num_of_variable))

        lb = []
        ub = []

        # x bound
        for i in range(self.num_of_point):
            lb.append(self.x_lower_bound[i])
            ub.append(self.x_upper_bound[i])
        
        # dx bound
        for i in range(self.num_of_point):
            lb.append(self.dx_lower_bound[i])
            ub.append(self.dx_upper_bound[i])
        
        # ddx bound
        for i in range(self.num_of_point):
            lb.append(self.ddx_lower_bound[i])
            ub.append(self.ddx_upper_bound[i])
        
        # dddx bound
        for i in range(self.num_of_point - 1):
            lb.append(self.dddx_lower_bound[i])
            ub.append(self.dddx_upper_bound[i])

        # start/end x
        lb.append(self.init_state[0] - 1e-2)
        ub.append(self.init_state[0] + 1e-2)
        lb.append(self.end_state[0] - 1e-2)
        ub.append(self.end_state[0] + 1e-2)

        # start dx
        lb.append(self.init_state[1] - 1e-2)
        ub.append(self.init_state[1] + 1e-2)

        # start ddx
        lb.append(self.init_state[2] - 1e-2)
        ub.append(self.init_state[2] + 1e-2)

        ## dx consistency
        for i in range(self.num_of_point - 1):
            lb.append(-10e-5)
            ub.append(10e-5)
        
        ## x consistency
        for i in range(self.num_of_point - 1):
            lb.append(-10e-5)
            ub.append(10e-5)

        numpy.set_printoptions(linewidth=numpy.inf)
        print(A.toarray())
        print(lb)
        print(ub)
        return A,lb,ub    

    def Optimize(self):
        Q = self.CalculateQ()
        P = self.CalculateP()

        A, lb, ub = self.CalculateAffineConstraint()

        prob = osqp.OSQP()
        prob.setup(Q, P, A, lb, ub, polish=True, eps_abs=1e-5, eps_rel=1e-5,
                   eps_prim_inf=1e-5, eps_dual_inf=1e-5, verbose=True)

        var_warm_start = numpy.array(self.ref_x + [0.0 for n in range(2 * self.num_of_point)])
        prob.warm_start(x = var_warm_start)
        res = prob.solve()

        self.solution_x = res.x[0:self.num_of_point]
        self.solution_dx = res.x[self.num_of_point:2 *self.num_of_point]
        self.solution_ddx = res.x[2 *self.num_of_point:3 * self.num_of_point]

    def VizResult(self):
        pass