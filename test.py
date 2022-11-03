import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import matplotlib

mu_1 = np.array([0.4, 0.8])
mu_2 = np.array([1.5, 2.7])

p_omega_1 = 0.95
p_omega_2 = 0.05

sigma = np.array([[1.5, 0], [0, .8]])
# x_0 = np.random.normal(1, 2, 10)
# x_1 = np.random.normal(1.2, 1.8, 10)
x_0 = np.arange(-5, 5, 1)
x_1 = np.arange(-5, 5, 1)
x = np.array([x_0, x_1]).transpose()
d = x.shape[1]
print(x.shape[0])
x_0, x_1 = np.meshgrid(x_0, x_1)

# Answer to question 1
def distribution_values(x, mu, sigma):
    if not (isinstance(x, (int, np.integer, float))):
        p = np.zeros(x.shape[0])
        for i in range(x.shape[0] - 1):
            p[i] = 1 / (pow((2 * np.pi), (d / 2)) * np.sqrt(np.linalg.det(sigma))) * np.exp(
                -0.5 * np.matmul(np.matmul(np.transpose(x[i, :] - mu), np.linalg.inv(sigma)), (x[i, :] - mu)))
    else:
        p = 1 / (pow((2 * np.pi), (d / 2)) * np.sqrt(np.linalg.det(sigma))) * np.exp(
            -0.5 * np.matmul(np.matmul(np.transpose(x - mu), np.linalg.inv(sigma)), (x - mu)))
    return p


print(distribution_values(x, mu_1, sigma))
print(distribution_values(x, mu_2, sigma))


# Answer to question 2
def probability_distribution(distribution_values_1, distribution_values_2):

    probability_distrib = np.array([])

    for i in range(distribution_values_1.shape[0]):
        probability_distrib = np.append(probability_distrib, distribution_values_1[i] * p_omega_1 + distribution_values_2[i] * p_omega_2)

    return probability_distrib


distribution_values_1 = distribution_values(x, mu_1, sigma)
distribution_values_2 = distribution_values(x, mu_2, sigma)
print(probability_distribution(distribution_values_1, distribution_values_2))


def discriminant_function(x_0, x_1, mu, sigma, a_priori):
    g = np.zeros([x_0.shape[0], x_0.shape[1]])

    for i in range(x_0.shape[0]):
        for j in range(x_1.shape[0]):
            g[i][j] = -0.5 * np.matmul(np.matmul(np.transpose(([x_0[i][j], x_1[i][j]] - mu)), np.linalg.inv(sigma)),
                                       ([x_0[i][j], x_1[i][j]] - mu)) + np.log(a_priori)

    return g


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

x = discriminant_function(x_0, x_1, mu_2, sigma, p_omega_2)
y = discriminant_function(x_0, x_1, mu_1, sigma, p_omega_1)

fig = plt.figure()
fig.add_subplot()
surf = ax.plot_surface(x_0, x_1, x)
surf = ax.plot_surface(x_0, x_1, y)
plt.rcParams['figure.figsize'] = [10, 5]

ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()


# def bayes_error(p_omega_1, p_omega_2, mu_1, mu_2, sigma):
#     case_1 = discriminant_function(x, mu_1, sigma, p_omega_1)
#     case_2 = discriminant_function(x, mu_2, sigma, p_omega_2)
#
#     # case =
