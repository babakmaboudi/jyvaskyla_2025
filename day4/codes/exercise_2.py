import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# create a discrete domain
dim = 100
s = np.linspace(0,1,dim,endpoint=False)
# creating a signal that is 1 in the interval (0.2, 0.35) and is 2 in the interval (0.5,0.7)
x_true = ((0.2 < s) & (s < 0.35)).astype(float) + 2 * ((0.5 < s) & (s < 0.7)).astype(float) # use s to create a Numpy vector that is 1 in the interval (0.2, 0.35) and is 2 in the interval (0.5,0.7)


#plotting the signal
plt.plot(s,x_true, drawstyle='steps-post', label='true signal')
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

y = A(x_true)

plt.plot(s,y,label='blurred signal')

#sigma = 0.1*np.linalg.norm(y)/np.sqrt(dim)
sigma = 0.09197806689626055
sigma2 = sigma*sigma

e = np.random.standard_normal(dim)
y_obs = A(x_true) + sigma*e

plt.plot(s,y_obs,label='noisy and blurred signal')
plt.xlabel("s")
plt.grid(True)
plt.tight_layout()
plt.legend()

# defining the posterior distribution
log_prior = # write a function that computes the log of un-normalized prior density
log_likelihood = # write a function that computes the log of un-normalized likelihood density
log_posterior = # write a function that computes log of un-normalized posterior

#initiating random-walk Metropolis-Hastings method
x0 = # define the initial point in Markov chain
samples = [ x0 ]
acc = [True]
x = x0
c = # step size in random-walk Metropolis-Hastings
num_samples = # number of samples in the Markov chain
for i in range(num_samples):
    epsilon = np.random.randn(dim)
    x_star = x + c*epsilon

    ratio_nominator = log_posterior(x_star)
    ratio_denominator = log_posterior(x)

    alpha = min(0,ratio_nominator-ratio_denominator)
    u = np.log( np.random.rand() )
    if( u < alpha ):
        x = x_star
        accepted = True
    else:
        accepted = False
    samples.append( x )
    acc.append( accepted )
samples = np.array(samples)
acc = np.array(acc)

# compute the acceptance rate of the random walk method
print( # the acceptance rate )

# computing the mean of the posterior
x_mean = # compute the mean sample

# plotting the meain
plt.plot(s,x_mean,label='posterior mean')
plt.xlabel("s")
plt.grid(True)
plt.tight_layout()
plt.legend()

# plotting pixel-wise standard deviation
x_std = # compute the point-wise std or variance
plt.figure()
plt.plot(s,x_std,label='posterior point-wise std')
plt.xlabel("s")
plt.ylabel("std")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
