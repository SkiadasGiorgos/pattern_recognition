import numpy as np
from scipy.constants import pi
import matplotlib.pyplot as plt


def log_likelihood(x, theta):
    value = np.sum(np.log((1 / pi) * 1 / (1 + pow(x[i] - theta, 2))) for i in range(len(x)))
    return value


def likelihood(x, theta):
    x = [(1 / pi) * 1 / (1 + pow(x[i] - theta, 2)) for i in range(len(x))]
    value = np.prod(x)
    return value


def probability_distribution(x, theta):
    value = (1 / pi) * 1 / (1 + pow(x - theta, 2))
    return value


def calculate_log_likelihood(theta_array, d):
    LL = []  # log likelihood
    for theta in theta_array:
        loglike = log_likelihood(d, theta)
        LL.append(loglike)
    return LL


def calculate_likelihood(theta_array, d):
    likelihood_list = []  # log likelihood
    for theta in theta_array:
        like = likelihood(d, theta)
        likelihood_list.append(like)
    return likelihood_list


def predict(d, theta_1, theta_2, a_priori_1, a_priori_2):
    class_1 = []
    class_2 = []
    values_class_1 = []
    values_class_2 = []
    for x in d:
        value = np.log(probability_distribution(x, theta_1)) - np.log(probability_distribution(x, theta_2)) + np.log(
            a_priori_1) - np.log(a_priori_2)
        if value > 0:
            class_1.append(x)
            values_class_1.append(value)
        else:
            class_2.append(x)
            values_class_2.append(value)
    return class_1, class_2, values_class_1, values_class_2


def prior_distribution(theta_array):
    p_theta = []
    for theta in theta_array:
        value = (1 / (10 * pi)) * 1 / (1 + pow(theta / 10, 2))
        p_theta.append(value)
    return p_theta


def posterior_distribution(likelihood, p_theta, theta_array):
    numerator = np.multiply(likelihood, p_theta)
    denominator = np.trapz(numerator, x=theta_array)
    posterior = numerator / denominator
    return posterior


d1 = np.array([2.8, -0.4, -0.8, 2.3, -0.3, 3.6, 4.1])
d2 = np.array([-4.5, -3.4, -3.1, -3.0, -2.3])
a_priori_1 = 7 / 12
a_priori_2 = 5 / 12
dx = 0.001
theta_array = np.arange(-10, 10, dx)

logLikelihood1 = calculate_log_likelihood(theta_array, d1)
logLikelihood2 = calculate_log_likelihood(theta_array, d2)

best_theta_1 = theta_array[
    logLikelihood1.index(max(logLikelihood1))]  # Find the x value corresponding to the maximum y value
print("Estimation of theta 1: ", best_theta_1)

max_ll2 = max(logLikelihood2)  # Find the maximum y value
best_theta_2 = theta_array[
    logLikelihood2.index(max(logLikelihood2))]  # Find the x value corresponding to the maximum y value
print("Estimation of theta 2: ", best_theta_2)

class_1, class_2, values_class_1, values_class_2  = predict(np.concatenate((d1, d2)), best_theta_1, best_theta_2, a_priori_1, a_priori_2)


likelihood1 = calculate_likelihood(theta_array, d1)
likelihood2 = calculate_likelihood(theta_array, d2)
p_theta = prior_distribution(theta_array)
posterior1 = posterior_distribution(likelihood1, p_theta, theta_array)
posterior2 = posterior_distribution(likelihood2, p_theta, theta_array)

print(max(p_theta))

# plt.plot(theta_array, posterior1)
# plt.plot(theta_array, posterior2)
# plt.plot(theta_array, p_theta)
# plt.xlabel('theta')
# plt.title("$ P(\theta | D1) $")
# plt.show()

# plt.plot(theta_array, LL1)
# plt.xlabel('theta')
# plt.ylabel('Log-likelihood')
# plt.title('Log likelihood over a range of theta values for d1')
# plt.show()
#
# plt.plot(theta_array, LL2)
# plt.xlabel('theta')
# plt.ylabel('Log-likelihood')
# plt.title('Log likelihood over a range of theta values for d2')
# plt.show()

# color = ['red' if x in d1 else 'blue' for x in class_1]
labels = ['$ class: \omega_1$', '$ class: \omega_2$', 'Decision rule']
plt.scatter(class_1, values_class_1)
plt.scatter(class_2, values_class_2)
plt.axhline(y=0, color="black", linestyle='--')
plt.xlabel('x')
plt.ylabel('g(x)')
plt.legend(labels=labels)
plt.show()
