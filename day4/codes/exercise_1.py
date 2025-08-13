import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# create a discrete domain
dim = 100
s = np.linspace(0,1,dim,endpoint=False)
# creating a signal that is 1 in the interval (0.2, 0.35) and is 2 in the interval (0.5,0.7)
x = # use s to create a Numpy vector that is 1 in the interval (0.2, 0.35) and is 2 in the interval (0.5,0.7)


#plotting the signal
plt.plot(s,x, drawstyle='steps-post', label='true signal')
plt.xlabel("s")
plt.ylabel("intensity")
plt.grid(True)
plt.tight_layout()
plt.legend()

# creating a blurring operator
delta = 1
def A(signal, sigma=delta):
    """
    Apply Gaussian blur to a 1D signal using scipy.
    
    Parameters:
    - signal (np.ndarray): Original signal (1D array).
    - sigma (float): Standard deviation of the Gaussian kernel.
    
    Returns:
    - blurred (np.ndarray): Blurred signal.
    """
    return gaussian_filter1d(signal, sigma=sigma)

y = # apply blurring

plt.plot(s,y,label='blurred signal')

#sigma = 0.1*np.linalg.norm(y)/np.sqrt(dim)
sigma = 0.09197806689626055
sigma2 = sigma*sigma

y_obs = # create a noisy measurement using sigma. Hint: Follow the forward model

plt.plot(s,y_obs,label='noisy and blurred signal')
plt.xlabel("s")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
