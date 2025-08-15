import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import lu_factor, lu_solve

class hydraulic_class():
    def __init__(self, N, L=1):
        self.L = L
        self.N = N
        self.x = np.linspace(self.L/self.N,1,self.N)
        self.dx = self.L/self.N
        self.source()

    def forward(self, a):
        a = np.exp(a)
        diag1 = -(a[1:] + a[:-1])
        diag1 = np.concatenate([diag1,[-a[-1]]])
        diag2 = a[1:]

        Dxx = np.diag(diag1) + np.diag(diag2,-1) + np.diag(diag2,1)
        Dxx /= self.dx*self.dx

        lu, piv = lu_factor(Dxx)

        sol = []
        for b in self.b_terms:
            sol.append( lu_solve((lu, piv), b) )

        return np.array(sol)

    def source(self, n_source=5, std=0.02):
        dist = self.L/(n_source+1)
        source_coords = np.linspace( dist,self.L-dist, n_source )

        self.b_terms = []
        for i in range(n_source):
            self.b_terms.append( np.exp( -0.5*(self.x - source_coords[i])**2/std/std )/std/np.sqrt(2*np.pi) )

if __name__ == '__main__':
    N = 128
    X = np.ones(128)

    problem = hydraulic_class(N)
    p = problem.forward(X)

    print(p.shape)
