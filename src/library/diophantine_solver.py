###############################################################################
# This library solve linear diophantine equations
# The method is using elementary mode calculation found 
# in Schuster S, et al. Trends Biotechnol. 1999
# Authors: Jean-loup Faulon jfaulon@gmail.com
# March 2023
###############################################################################

from library.imports import *

MAXSOLUTION = 1.0e7

def DiophantineSolver(C, verbose=False):
# Callable function
# ARGUMENTS:
# C:  m (contraints) x n (variables) matrix
# Vmin, Vmax: minimum (maximum) values for the variables
# RETURNS:
# R: an N x n array of modes where N is 
# the number of modes

    def ZeroPosNeg(R, pos, neg, cr):
        # each Rposneg repect the constraint (i.e. cr = 0)
        I, J = pos.shape[0], neg.shape[0]
        Rposneg = np.zeros((R.shape[0], I*J))
        for i in range(I):
            for j in range(J):
                r = cr[pos[i]] * R[:,neg[j]] - cr[neg[j]] * R[:,pos[i]]
                # get smallest values dividing by GCD
                r = r/np.gcd.reduce(r.astype(int)) 
                Rposneg[:,i*J+j] = r
        return Rposneg
 
    m, n = C.shape[0], C.shape[1]
    R = np.identity(n)
    for i in range(m):
        cr = R.T.dot(C[i])
        pos  = np.transpose((np.argwhere(cr > 0)))[0]
        neg  = np.transpose((np.argwhere(cr < 0)))[0]
        zero = np.transpose((np.argwhere(cr == 0)))[0]
        if pos.shape[0] * neg.shape[0] + neg.shape[0] > MAXSOLUTION:
            print(f'WARNING too many solutions: \
            {pos.shape[0] * neg.shape[0] + neg.shape[0]:.1E} > {MAXSOLUTION:.1E}')
            return np.asarray([[]])
        Rzero = R[:,zero]
        Rposneg = ZeroPosNeg(R, pos, neg, cr)
        R = np.concatenate((Rposneg, Rzero), axis=1)
        if verbose == 2: print(f'-------- constraint {i} ------------')
        if verbose == 2: print(f'C=\n {C[i]}')
        if verbose == 2: print(f'R=\n {R}')
        
    return R

