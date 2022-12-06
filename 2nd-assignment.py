import numpy as np
from sympy import *
from scipy.constants import pi
import matplotlib.pyplot as plt

def probability_distribution(x, theta):
    value = np.sum((1/pi) * 1 / (1 + pow(x[i] - theta, 2)) for i in range(len(x)))
    return value

def calculate_log_likelihood(theta_array, d):
    LL = [] #log likelihood
    for theta in theta_array:
        loglike = np.log(probability_distribution(d, theta))
        LL.append(loglike)
    return LL

d1 = np.array([2.8, -0.4, -0.8, 2.3, -0.3, 3.6, 4.1])
d2 = np.array([-4.5, -3.4, -3.1, -3.0, -2.3])
theta_array = np.geomspace(0.000001, 10, 100000)

LL1 = calculate_log_likelihood(theta_array, d1)
LL2 = calculate_log_likelihood(theta_array, d2)

plt.plot(theta_array, LL1)
plt.xlabel('theta')
plt.ylabel('Log-likelihood')
plt.title('Log likelihood over a range of theta values for d1')
plt.show()

plt.plot(theta_array, LL2)
plt.xlabel('theta')
plt.ylabel('Log-likelihood')
plt.title('Log likelihood over a range of theta values for d2')
plt.show()