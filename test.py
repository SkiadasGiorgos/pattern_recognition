import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import matplotlib


mu_1 = np.array([0.4, 0.8])
mu_2 = np.array([1.5, 2.7])
print(mu_2.shape)
p_omega_1 = 0.95
p_omega_2 = 0.05

sigma = np.array([[1.5, 0], [0, .8]])
x = np.array([1, 0])
x_0 = np.arange(-5,6,1)
x_1 = np.arange(-5,6,1)

def distribution_prices(x, mu, sigma):
    d = mu.shape[0]
    p = np.zeros(x.shape[0])
    p = 1 / (pow((2 * np.pi), (d / 2)) * np.sqrt(np.linalg.det(sigma))) * np.exp(
        -0.5 * np.matmul(np.matmul(np.transpose(x - mu), np.linalg.inv(sigma)), (x - mu)))
    # for i in range(x.shape[0]):
    #     p[i] = 1 / (pow((2 * np.pi), (d / 2)) * np.sqrt(np.linalg.det(sigma))) * np.exp(
    #         -0.5 * np.matmul(np.matmul(x[:, i] - mu, np.transpose(sigma)), (x[:, i] - mu)))
    return p


# print(distribution_prices(x, mu_1, sigma))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

def probability_distribution(x, mu_1, mu_2, sigma_1, sigma_2):
    d = mu_1.shape[0]
    # probability_distrib = 1 / (pow((2 * np.pi), (d / 2)) * np.sqrt(np.linalg.det(sigma))) * np.exp(
    #     -0.5 * np.matmul(np.matmul(np.transpose(x - mu_1), np.linalg.inv(sigma)), (x - mu_1)))*p_omega_1 + 1 / (
    #                               pow((2 * np.pi), (d / 2)) * np.sqrt(np.linalg.det(sigma))) * np.exp(
    #     -0.5 * np.matmul(np.matmul(np.transpose(x - mu_2), np.linalg.inv(sigma)), (x - mu_2)))*p_omega_2


    probability_distrib = np.array([])

    for i in range(x.shape[0]-1):
        for j in range(x.shape[1]-1):
            probability_distrib = [(1 / (pow((2 * np.pi), (d / 2)) * np.sqrt(np.linalg.det(sigma))) * np.exp(
                -0.5 * np.matmul(np.matmul(np.transpose(x[i, j] - mu_1), np.linalg.inv(sigma)),
                                 (x[i, j] - mu_1))) + 1 / (
                                               pow((2 * np.pi), (d / 2)) * np.sqrt(np.linalg.det(sigma))) * np.exp(
                -0.5 * np.matmul(np.matmul(np.transpose(x[i, j] - mu_2), np.linalg.inv(sigma)), (x[i, j] - mu_2)))),probability_distrib]

    return probability_distrib

# print(probability_distribution(np.array([x_0,x_1]).transpose(),mu_1,mu_2,sigma,sigma))



def discriminant_function(x, mu, sigma, a_priori):
    g = np.zeros(x.shape[0])
    g = -0.5 * np.matmul(np.matmul(np.transpose((x - mu)), np.linalg.inv(sigma)), (x - mu)) + np.log(
        a_priori)

    # for i in range(x.shape[0]):
    #     g[i] = -0.5 * np.matmul(np.matmul(np.transpose((x[:, i] - mu)), np.linalg.inv(sigma)), (x[:, i] - mu)) + np.log(
    #         a_priori)

    return g


# print(discriminant_function(x, mu_2, sigma, p_omega_2))

def bayes_error(p_omega_1,p_omega_2,mu_1,mu_2,sigma):
    case_1 = discriminant_function(x, mu_1, sigma, p_omega_1)
    case_2 = discriminant_function(x, mu_2, sigma, p_omega_2)

    # case =
    pass