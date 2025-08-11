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


def roam(station_id):
    """
    This function will move from one station to the next according to 
    the transition matrix above
    """
    probabilities = P[:,station_id]
    next_station = np.random.choice(len(probabilities), p=probabilities)
    return next_station

x_current = # choose an initial station of your choise
samples = []
for i in range(100000):
    x_next = # apply the roam to move to the next station
    samples.append(x_next)
    x_current = x_next
samples = np.array(samples)

f,ax = plt.subplots(1)
ax.hist(samples)
plt.show()

