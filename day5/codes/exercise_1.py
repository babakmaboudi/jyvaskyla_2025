import numpy as np
import matplotlib.pyplot as plt

N = 100 # number of discretization points
s = # create a discretization of the interval [0,1], e.e., using np.linspace

# defining the length scale
length_scale = # choose a length scale

# defining Gaussian covariance function
def gaussian_cov_func(x, x_prime): # write a function that computes Gaussian covariance kernel between two given points

# defining exponential covariance function
def exponential_cov_func(x, x_prime):  # write a function that computes exponential covariance kernel between tow given poitns

# creating a covariance matrix using covariance functions defined above
cov_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        cov_matrix[i, j] = # fill-in the covariance matrix using s and the kernels defined above

# sampling from the multivariate normal
mean = np.zeros(N) # mean for sampling
samples = np.random.multivariate_normal(mean, cov_matrix, size=5)

# plotting the samples
plt.figure()
for i, sample in enumerate(samples):
    plt.plot(s, sample, label=f'Sample {i+1}')
plt.title("Samples from a Gaussian Process")
plt.xlabel("s")
plt.ylabel("x")
plt.legend()
plt.grid(True)
plt.savefig('exponential_0.05.pdf')
plt.show()

cov_matrix = cov_matrix + 1e-10 * np.eye(N) # this is to remove numerical errors
cov_mat_chol = np.linalg.cholesky(cov_matrix)

# plotting the samples
plt.figure()
for i in range(5):
    z_sample = np.random.standard_normal(N)
    x_sample = cov_mat_chol@z_sample
    plt.plot(s, x_sample, label=f'Sample {i+1}')
plt.title("Samples from a Gaussian Process")
plt.xlabel("s")
plt.ylabel("x")
plt.legend()
plt.grid(True)
plt.savefig('exponential_0.05.pdf')
plt.show()

# do this only for the Gaussian covariance kernel with length_scale = 0.1
cov_matrix = cov_matrix + 1e-10 * np.eye(N) # this is to remove numerical errors
cov_mat_chol = # compute the Cholesky factor of cov_matrix

samples_z = # compute 5 samples of size N
samples_x = (cov_mat_chol@samples_z.T).T # this is to tranform z to x following x = C^{1/2}z

# plotting the samples
plt.figure()
for i, sample in enumerate(samples_x):
    plt.plot(s, sample, label=f'Sample {i+1}')
plt.title("Samples from a Gaussian Process")
plt.xlabel("s")
plt.ylabel("x")
plt.legend()
plt.grid(True)
plt.savefig('exponential_0.05.pdf')
plt.show()


plt.show()

