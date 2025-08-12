import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

prior = # function that computes the prior probability

A = # define the matrix A

#x_true = np.array([1,1])
#e = 0.3*np.random.randn( 3 )
#y_obs = A@x_true + e

sigma2 = 0.09
y_obs = np.array([ 0.08684452, -1.63278483,  2.9668183 ])

likelihood = # function that computes the likelihood probability

posterior = # function that computes the posterior probability

dim = # dimension of x
x0 = # choose an initial starting point for the Markov chain, you can choose zeros, ones or random.
samples = [ x0 ]
acc = [1]
x = x0
c = # step size of metropolis proposal
num_samples = 50000 # number of samples
# random walk metropolis-hastings loop 
for i in range(num_samples):
    x_star = # propose a new state folloing Normal(x, c*I), meaning x is the mean, c is the step size and I is the identity matrix

    ratio_nominator = # compute the nominator in the acceptance ratio
    ratio_denominator = # compute the denominator in the acceptance ratio

    alpha = min(1,ratio_nominator/ratio_denominator)
    u = # draw a sample from U([0,1])
    if( ):# if acceptance condition is True 
        x = x_star
        accepted = True
    else:
        accepted = False
    samples.append( x )
    acc.append( accepted )
samples = np.array(samples)
samples = # down sample or skip every 10 samples

mean = # compute the mean of the samples

f,ax = plt.subplots(1)

c =ax.hist2d(samples[:,0], samples[:,1], bins=50, cmap='plasma')
ax.plot( mean[0], mean[1], 'go', label='mean value' )
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
f.colorbar(c[3], ax=ax, label='counts')
ax.set_title('2D Histogram of the posterior')
ax.set_aspect('equal')
plt.legend()
plt.show()
