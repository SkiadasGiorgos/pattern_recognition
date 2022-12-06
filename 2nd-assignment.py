import numpy as np
from scipy.constants import pi
import matplotlib.pyplot as plt


def log_probability_distribution(x, theta):
    value = np.sum(np.log((1 / pi) * 1 / (1 + pow(x[i] - theta, 2))) for i in range(len(x)))
    return value


def probability_distribution(x, theta):
    value = (1 / pi) * 1 / (1 + pow(x - theta, 2))
    return value


def calculate_log_likelihood(theta_array, d):
    LL = []  # log likelihood
    for theta in theta_array:
        loglike = log_probability_distribution(d, theta)
        LL.append(loglike)
    return LL


def predict(d, theta_1, theta_2, a_priori_1, a_priori_2):
    class_1 = []
    class_2 = []
    values = []
    for x in d:
        value = np.log(probability_distribution(x, theta_1)) - np.log(probability_distribution(x, theta_2)) + np.log(
            a_priori_1) - np.log(a_priori_2)
        values.append(value)
        if value > 0:
            class_1.append(x)
        else:
            class_2.append(x)
    return class_1, class_2, values


d1 = np.array([2.8, -0.4, -0.8, 2.3, -0.3, 3.6, 4.1])
d2 = np.array([-4.5, -3.4, -3.1, -3.0, -2.3])
a_priori_1 = 7 / 12
a_priori_2 = 5 / 12
theta_array = np.arange(-10, 10, 0.001)

LL1 = calculate_log_likelihood(theta_array, d1)
LL2 = calculate_log_likelihood(theta_array, d2)

best_theta_1 = theta_array[LL1.index(max(LL1))]  # Find the x value corresponding to the maximum y value
print("Estimation of theta 1: ", best_theta_1)

max_ll2 = max(LL2)  # Find the maximum y value
best_theta_2 = theta_array[LL2.index(max(LL2))]  # Find the x value corresponding to the maximum y value
print("Estimation of theta 2: ", best_theta_2)

class_1, class_2, values = predict(np.concatenate((d1, d2)), best_theta_1, best_theta_2, a_priori_1, a_priori_2)

print(class_1, class_2)
print(np.concatenate((d1, d2)))
print(values)

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
x_axis = np.concatenate((d1,d2))
y_axis = values
color = ['red' if x in class_1 else 'blue' for x in x_axis]
#labels = ['$ class: \omega_1$', '$ class: \omega_2$', 'Decision rule']
plt.scatter(x_axis,y_axis,color=color)
plt.axhline(y = 0, color = "black", linestyle = '--')
plt.xlabel('x')
plt.ylabel('g(x)')
#plt.legend(labels=labels)
plt.show()
