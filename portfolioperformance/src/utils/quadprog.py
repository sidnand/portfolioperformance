import numpy as np
from qpsolvers import solve_qp

def quadprog(H, f, Aeq, beq, lb, ub):
   """
   minimize:
         (1/2)*x'*H*x + f'*x
   subject to:
         Aeq*x = beq 
         lb <= x <= ub
   """

   beq = np.array(beq)

   x = solve_qp(H, f, G = None, h = None, A = Aeq, b = beq, lb = lb, ub = ub, solver="quadprog")

   x = x[:, np.newaxis]
   return x