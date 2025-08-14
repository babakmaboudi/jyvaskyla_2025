import numpy as np
import matplotlib.pyplot as plt

from hydraulic import hydraulic_class

N = 100 # specify the number of discreitization points
s = np.linspace(0,1,N,endpoint=False)

hydraulic = hydraulic_class(100) # calling the hydraulic class

x = np.ones(N)
pressures = hydraulic.forward(x)

plt.figure()
for i, p in enumerate(pressures):
    plt.plot(s, p, label=f'injection no. {i+1}')
plt.title("pressure profiles")
plt.xlabel("s")
plt.ylabel("p")
plt.legend()
plt.grid(True)
plt.savefig('pressures.pdf')
plt.show()
