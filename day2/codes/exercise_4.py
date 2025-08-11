import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy

# Target: Standard normal density function
def f(x):
    #norm.pdf(x)

# Proposal: Standard Cauchy density function
def g(x):
    # cauchy.pdf(x)

# Proposal sampler using scipy to sample from the standard Cauchy distribution
def sample_from_g(): return cauchy.rvs()

# Pre-generate samples
c = # steo-size must satisfy f(x) <= C * g(x) for all x
n_samples = # set number of samples
samples = []
while len(samples) < n_samples:
    x = # sample from g, use the function you defined above
    u = # sample a random number between 0 and 1, i.e., u ~ U([0,1])
    if( [write acceptance condition] is True ):
        accepted = True
    else:
        accepted = False
    samples.append((x, u, accepted))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ploting density functions
x_vals = np.linspace(-6, 6, 1000)
fx = f(x_vals)
gx = g(x_vals)

# Plot target and scaled proposal densities
axes[0].plot(x_vals, fx, label='Target f(x): Normal', color='blue')
axes[0].plot(x_vals, c * gx, label='C × Proposal g(x): Cauchy', color='orange')

axes[0].set_xlim(-6, 6)
axes[0].set_ylim(0, max(c * gx) * 1.1)
axes[0].set_xlabel('x')
axes[0].set_ylabel('Density')
axes[0].set_title('Acceptance-Rejection Sampling')
axes[0].legend()

# plotting the accepted samples with green and rejected ones with red
accepted_x = []
for i in range(len(samples)):
    x, u, accepted = samples[i]
    if accepted:
        accepted_x.append(x)
        axes[0].plot(x, u, 'go')
    else:
        axes[0].plot(x, u, 'rx')
    

# Histogram
bins = np.linspace(-6, 6, 100)
hist = axes[1].hist([], bins=bins, density=True, alpha=0.6, color='skyblue')[2]
pdf_line, = axes[1].plot(x_vals, fx, 'k--', label="Target Normal PDF")

axes[1].hist(accepted_x, bins=bins, density=True, alpha=0.6, color='skyblue')
axes[1].plot(x_vals, fx, 'k--', label="Target Normal PDF")
axes[1].set_xlim(-6, 6)
axes[1].set_ylim(0, 0.45)
axes[1].set_title("Histogram of Accepted Samples")
axes[1].set_xlabel("Sampled x")
axes[1].set_ylabel("Density")
axes[1].legend()


acceptance_ratio = len(accepted_x) / len(samples)
print(f"Acceptance ratio: {len(accepted_x)} / {len(samples)} = {acceptance_ratio:.2f}")

plt.show()
