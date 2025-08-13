import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# create a discrete domain
dim = 100
s = np.linspace(0,1,dim,endpoint=False)
# creating a signal
x = ((0.2 < s) & (s < 0.35)).astype(float) + 2 * ((0.5 < s) & (s < 0.7)).astype(float)

#plotting the signal
plt.plot(s,x, drawstyle='steps-post', label='true signal')
plt.xlabel("s")
plt.ylabel("intensity")
plt.grid(True)
plt.tight_layout()
plt.legend()

# creating a blurring operator
sigma = 1
def A(signal, sigma=sigma):
    """
    Apply Gaussian blur to a 1D signal using scipy.
    
    Parameters:
    - signal (np.ndarray): Original signal (1D array).
    - sigma (float): Standard deviation of the Gaussian kernel.
    
    Returns:
    - blurred (np.ndarray): Blurred signal.
    """
    return gaussian_filter1d(signal, sigma=sigma)

y = A(x)

plt.plot(s,y,label='blurred signal')
plt.xlabel("s")
plt.grid(True)
plt.tight_layout()
plt.legend()

sigma = 0.1*np.linalg.norm(y)/np.sqrt(dim)
#sigma = 0.04584698788203243
sigma2 = sigma*sigma

e = np.random.standard_normal(dim)
y_obs = A(x) + sigma*e

plt.plot(s,y_obs,label='noisy and blurred signal')
plt.xlabel("s")
plt.grid(True)
plt.tight_layout()
plt.legend()

# defining the integration operator. This is used in the changing of variable x = Tz
T = np.tril(np.ones((dim, dim), dtype=float))

# defining the posterior distribution

#uncomment these lines for Gassian porior, best parameters are sigma_prior = 0.1 and c = 0.004
#sigma_prior = 0.1
#log_prior = # write a function that computes the log of a Gaussian density with standard deviation sigma_prior

#uncomment these lines for Laplace porior, best parameters are b = 0.05 and c = 0.004
#b = 0.05
#log_prior = # write a function that computes the log of a Laplace density with spreading parameter b

# uncomment these lines for Cauchy porior, best parameters are gamma = 0.005 and c = 0.002
#gamma = 0.005 
#log_prior = write a function that computes the log of a Cauchy density with spreading parameter gamma

log_likelihood = # write a function that comptues the log of the likelihood with the change of variable x = Tz
log_posterior = lambda z: log_prior(z) + log_likelihood(z)

#initiating random-walk Metropolis-Hastings method
x0 = # initial point in Markov chain
samples = [ x0 ]
acc = [True]
x = x0
c = # step size in random-walk Metropolis-Hastings
num_samples = 10000 # number of samples in the Markov chain
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
samples_transformed = # here, you should transform all the samples back to x using the change of variables x = Tz
acc = np.array(acc)
print('acceptance rate is: ', np.sum(acc)/num_samples)

# computing the mean of the posterior
x_mean = np.mean(samples_transformed, axis=0)

# plotting the meain
plt.plot(s,x_mean,label='posterior mean')
plt.xlabel("s")
plt.grid(True)
plt.tight_layout()
plt.legend()

# plotting pixel-wise standard deviation
x_std = np.std(samples_transformed, axis=0)
plt.figure()
plt.plot(s,x_std,label='posterior point-wise std')
plt.xlabel("s")
plt.ylabel("std")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
