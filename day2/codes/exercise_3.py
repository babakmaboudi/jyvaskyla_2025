
import numpy as np
import matplotlib.pyplot as plt

# defining Montreal metro transition matrix
# each column represents the probability of transitioning to another station
P = np.array([
    [0,     1/2,   0,     1/4,   0    ],  # Row 0
    [1/2,   0,     1/3,   0,     0    ],  # Row 1
    [0,     1/2,   0,     1/2,   0    ],  # Row 2
    [1/2,   0,     2/3,   0,     1/2  ],  # Row 3
    [0,     0,     0,     1/4,   1/2  ]   # Row 4
])

P2 = # compute the transition matrix for doing 2 steps in the Markov chain
print(P2)

P200 = # compute the transition matrix for doing 200 steps in the Markov chain 
