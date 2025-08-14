import numpy as np
import matplotlib.pyplot as plt
import pickle

from hydraulic import hydraulic_class

N = # specify the number of discreitization points
s = np.linspace(0,1,N,endpoint=False)

hydraulic = hydraulic_class(N) # calling the hydraulic class

# defining the length scale for Gaussian covariance kernel
length_scale = 0.1

# defining Gaussian covariance function
def gaussian_cov_func(x, x_prime): # write a function that computes Gaussian covariance kernel between two given points

# creating a covariance matrix using covariance functions defined above
cov_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        cov_matrix[i, j] = # fill-in the covariance matrix using s and the kernels defined above
cov_matrix = cov_matrix + 1e-10 * np.eye(N) # this is to remove numerical errors

# creating the inverse of the covariance matrix
# generally this is a bad idea, but for the purpose of this course it is fine
cov_mat_chol = # find the cholesky factor of the covariance matrix

# creating log prior
log_prior = # function that evaluates the log of the prior

# loading measurement data 
with open('obs.pickle', 'rb') as handle:
    obs_data = pickle.load(handle)
y_obs = obs_data['y_obs']
sigmas = obs_data['sigmas']
sigma2s = sigmas**2


#creating log-likelihood function
def log_likelihood(z): # write a function that evaluates the log of the likelihood

log_posterior = # write a function that evaluates the log of the posterior

#write the code for random-walk Metropolis-Hastings method
x0 = np.zeros(N) # initial point in Markov chain
samples = [ x0 ]
acc = [True]
x = x0
c = 0.003 # step size in random-walk Metropolis-Hastings
num_samples = 10000 # number of samples in the Markov chain
for i in range(num_samples):
    x_star = # propose a state

    ratio_nominator = # nominator of the accpetance ratio
    ratio_denominator = # denominator of the acceptance ration

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
print('acceptance rate is: ', np.sum(acc)/num_samples)


# computing the mean of the posterior
x_mean = np.mean(samples, axis=0)
x_std = np.std( (cov_mat_chol@samples.T).T ,axis=0)

# plotting the meain
plt.figure()
plt.plot(s,cov_mat_chol@x_mean,label='posterior mean')
plt.xlabel("s")
plt.grid(True)
plt.tight_layout()
plt.legend()

plt.figure()
plt.plot(s,cov_mat_chol@x_mean,label='posterior point-wise std')
plt.xlabel("s")
plt.grid(True)
plt.tight_layout()
plt.legend()


plt.show()
