import numpy as np
from cvxopt import matrix as cvxmat, sparse, spmatrix
from cvxopt.solvers import qp, options

options['show_progress'] = False

def quadprog(H, f, Aeq, beq, lb, ub):
   """
   minimize:
         (1/2)*x'*H*x + f'*x
   subject to:
         Aeq*x = beq 
         lb <= x <= ub
   """
   P, q, G, h, A, b = _convert(H, f, Aeq, beq, lb, ub)
   results = qp(P, q, G, h, A, b)

   # Convert back to NumPy matrix
   # and return solution
   xstar = results['x']
   return np.matrix(xstar)


def _convert(H, f, Aeq, beq, lb, ub):
   """
   Convert everything to                                                                                              
   cvxopt-style matrices                                                                                              
   """
   P = cvxmat(H)
   q = cvxmat(f)
   if Aeq is None:
      A = None
   else:
      A = cvxmat(Aeq)
   if beq is None:
      b = None
   else:
      b = cvxmat(np.append([], beq))

   n = lb.size
   G = sparse([-speye(n), speye(n)])
   h = cvxmat(np.append([], [-lb, ub]))
   return P, q, G, h, A, b


def speye(n):
   """Create a sparse identity matrix"""
   r = range(n)
   return spmatrix(1.0, r, r)

# import numpy as np
# import cvxopt


# def quadprog(H, f, L=None, k=None, Aeq=None, beq=None, lb=None, ub=None):
#     """
#     Input: Numpy arrays, the format follows MATLAB quadprog function: https://www.mathworks.com/help/optim/ug/quadprog.html
#     Output: Numpy array of the solution
#     """
#     n_var = H.shape[1]

#     P = cvxopt.matrix(H, tc='d')
#     q = cvxopt.matrix(f, tc='d')

#     if L is not None or k is not None:
#         assert(k is not None and L is not None)
#         if lb is not None:
#             L = np.vstack([L, -np.eye(n_var)])
#             k = np.vstack([k, -lb])

#         if ub is not None:
#             L = np.vstack([L, np.eye(n_var)])
#             k = np.vstack([k, ub])

#         L = cvxopt.matrix(L, tc='d')
#         k = cvxopt.matrix(k, tc='d')

#     if Aeq is not None or beq is not None:
#         assert(Aeq is not None and beq is not None)
#         Aeq = cvxopt.matrix(Aeq, tc='d')
#         beq = cvxopt.matrix(beq, tc='d')

#     sol = cvxopt.solvers.qp(P, q, L, k, Aeq, beq)

#     return np.array(sol['x'])