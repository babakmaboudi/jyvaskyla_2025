import numpy as np
import matplotlib.pyplot as plt

# defining Montreal metro transition matrix
# each column represents the probability of transitioning to another station
# in other words p_ij is the probability of starting at station j and 
# arriving at station i
P = np.array([
    [0,     1/2,   0,     1/4,   0    ],  # Row 0
    [1/2,   0,     1/3,   0,     0    ],  # Row 1
    [0,     1/2,   0,     1/2,   0    ],  # Row 2
    [1/2,   0,     2/3,   0,     1/2  ],  # Row 3
    [0,     0,     0,     1/4,   1/2  ]   # Row 4
])

M = # number of stations
p0 = # choose the initial distribution distribution
 
print( #print the probability of being at the second station after one step )


p1 = # write this as a matrix-vector operation between P and p0

