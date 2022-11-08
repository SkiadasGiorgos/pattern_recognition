import numpy as np

mu_1 = np.array([0.4, 0.8])
mu_2 = np.array([1.5, 2.7])
p_omega_1 = 0.95
p_omega_2 = 0.05

sigma = np.array([[1.5, 0], [0, .8]])
sigma_2 = np.divide(sigma, 4)

step = .1
d = 2
x = np.array([1, 0])
x_0 = np.arange(-10, 10, step)
x_1 = np.arange(-10, 10, step)
x_0, x_1 = np.meshgrid(x_0, x_1)


def distribution_prices(x_0, x_1, mu, sigma):
    d = mu.shape[0]
    p = np.zeros([x_0.shape[0], x_0.shape[1]])
    for i in range(x_0.shape[0]):
        for j in range(x_1.shape[0]):
            p[i][j] = 1 / (pow((2 * np.pi), (d / 2)) * np.sqrt(np.linalg.det(sigma))) * np.exp(
                -0.5 * np.matmul(np.matmul(np.transpose(np.subtract([x_0[i][j], x_1[i][j]], mu)), np.linalg.inv(sigma)),
                                 np.subtract([x_0[i][j], x_1[i][j]], mu)))
    return p


# print(distribution_prices(x_0, x_1, mu_1, sigma))


# fig = plt.figure()
# fig.add_subplot()
# ax = plt.axes(projection='3d')
# surface_mu_1 = ax.plot_surface(x_0, x_1, distribution_prices_1)
# surface_mu_2 = ax.plot_surface(x_0, x_1, distribution_prices_2)
# plt.show()


def probability_distribution(x_0, x_1, mu_1, mu_2, sigma_1, sigma_2):
    d = mu_1.shape[0]

    probability_distrib = distribution_prices(x_0, x_1, mu_1, sigma_1) * p_omega_1 + distribution_prices(x_0, x_1, mu_2,
                                                                                                         sigma_2) * p_omega_2
    return probability_distrib


# distribution = probability_distribution(x_0, x_1, mu_1, mu_2, sigma, sigma_2)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# probability_of_surface = ax.plot_surface(x_0, x_1, distribution)
# plt.show()


def discriminant_function(x_0, x_1, mu, sigma, a_priori):
    a_posteriori = np.divide(distribution_prices(x_0, x_1, mu, sigma) * a_priori,
                             probability_distribution(x_0, x_1, mu_1, mu_2, sigma, sigma))

    return a_posteriori


# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

a_posteriori_1 = discriminant_function(x_0, x_1, mu_1, sigma, p_omega_1)
a_posteriori_2 = discriminant_function(x_0, x_1, mu_2, sigma, p_omega_2)


# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# fig = plt.figure()
# fig.add_subplot()
# surf = ax.plot_surface(x_0, x_1, a_posteriori_1)
# surf = ax.plot_surface(x_0, x_1, a_posteriori_2)
#
#
# ax.set_xlabel('x_0')
# ax.set_ylabel('x_1')
# ax.set_zlabel('a_posteriori')
# plt.show()


def region_of_alliance(a_posteriori_1, a_posteriori_2, x_0, x_1):
    distribution_prices_1 = distribution_prices(x_0, x_1, mu_1, sigma)
    distribution_prices_2 = distribution_prices(x_0, x_1, mu_2, sigma)

    for i in range(a_posteriori_1.shape[0]):
        for j in range(a_posteriori_1.shape[1]):

            if a_posteriori_2[i][j] > a_posteriori_1[i][j] > 0:
                distribution_prices_2[i][j] = 0
            elif a_posteriori_1[i][j] > a_posteriori_2[i][j] > 0:
                distribution_prices_1[i][j] = 0

    distribution_prices_2 *= pow(step,2)
    distribution_prices_1 *= pow(step, 2)


    final = np.sum(distribution_prices_1) * p_omega_1 + np.sum(distribution_prices_2) * p_omega_2

    return final


print(region_of_alliance(a_posteriori_1, a_posteriori_2, x_0, x_1))


