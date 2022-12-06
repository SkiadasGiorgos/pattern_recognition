import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import matplotlib

mu_1 = np.array([0.4, 0.8])
mu_2 = np.array([1.5, 2.7])

p_omega_1 = 0.95
p_omega_2 = 0.05

sigma = np.array([[1.5, 0], [0, .8]])
step = .1
x_0 = np.arange(-10, 10, step)
x_1 = np.arange(-10, 10, step)
d = 2
x_0, x_1 = np.meshgrid(x_0, x_1)
x = np.array([x_0, x_1]).transpose()


# Answer to question 1
def distribution_values(x, mu, sigma):
    if not (isinstance(x, (int, np.integer, float))):
        p = np.zeros([x.shape[0], x.shape[1]])
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                p[i, j] = 1 / (pow((2 * np.pi), (d / 2)) * np.sqrt(np.linalg.det(sigma))) * np.exp(
                    -0.5 * np.matmul(np.matmul(np.transpose(x[i, j] - mu), np.linalg.inv(sigma)), (x[i, j] - mu)))
    else:
        p = 1 / (pow((2 * np.pi), (d / 2)) * np.sqrt(np.linalg.det(sigma))) * np.exp(
            -0.5 * np.matmul(np.matmul(np.transpose(x - mu), np.linalg.inv(sigma)), (x - mu)))
    return p


distribution_values_1 = distribution_values(x, mu_1, sigma)
distribution_values_2 = distribution_values(x, mu_2, sigma)


# Answer to question 2
def probability_distribution(distribution_values_1, distribution_values_2):
    probability_distrib = np.zeros([distribution_values_1.shape[0], distribution_values_1.shape[1]])

    for i in range(distribution_values_1.shape[0]):
        for j in range(distribution_values_1.shape[1]):
            probability_distrib[i][j] = distribution_values_1[i][j] * p_omega_1 + distribution_values_2[i][
                j] * p_omega_2

    return probability_distrib


probability_distribution = probability_distribution(distribution_values_1, distribution_values_2)

fig = plt.figure()
ax = plt.axes(projection='3d')
surface_probability_distribution = ax.plot_surface(x_0, x_1, probability_distribution)


def a_posteriori(x, mu, sigma, a_priori):
    a_posteriori = np.divide(distribution_values(x, mu, sigma) * a_priori,
                             probability_distribution)

    return a_posteriori


a_posteriori_1 = a_posteriori(x, mu_1, sigma, p_omega_1)
a_posteriori_2 = a_posteriori(x, mu_2, sigma, p_omega_2)

fig = plt.figure()
fig.add_subplot()
ax = plt.axes(projection='3d')
surface_a_posteriori_1 = ax.plot_surface(x_0, x_1, a_posteriori_1)
surface_a_posteriori_2 = ax.plot_surface(x_0, x_1, a_posteriori_2)

plt.show()


def error(a_posteriori_1, a_posteriori_2):
    distrib_values_1 = distribution_values_1
    distrib_values_2 = distribution_values_2

    for i in range(a_posteriori_1.shape[0]):
        for j in range(a_posteriori_1.shape[1]):

            if a_posteriori_2[i][j] > a_posteriori_1[i][j] > 0:
                distrib_values_2[i][j] = 0
            elif a_posteriori_1[i][j] > a_posteriori_2[i][j] > 0:
                distrib_values_1[i][j] = 0

    distrib_values_1 *= pow(step, 2)
    distrib_values_2 *= pow(step, 2)

    error = np.sum(distrib_values_1) * p_omega_1 + np.sum(distrib_values_2) * p_omega_2
    return error


print(error(a_posteriori_1, a_posteriori_2))
