import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import matplotlib


mu_1 = np.array([0.4, 0.8])
mu_2 = np.array([1.5, 2.7])
p_omega_1 = 0.95
p_omega_2 = 0.05

sigma = np.array([[1.5, 0], [0, .8]])
x = np.array([1, 0])
x_0 = np.arange(-5,6,1)
x_1 = np.arange(-5,6,1)
x_0,x_1 = np.meshgrid(x_0,x_1)

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



def probability_distribution(x, mu_1, mu_2, sigma_1, sigma_2):
    d = mu_1.shape[0]

    probability_distrib = np.array([])

    for i in range(x.shape[0]-1):
        for j in range(x.shape[1]-1):
            probability_distrib = np.append(probability_distrib, 1 / (pow((2 * np.pi), (d / 2)) * np.sqrt(np.linalg.det(sigma))) * np.exp(
                -0.5 * np.matmul(np.matmul(np.transpose(x[i, j] - mu_1), np.linalg.inv(sigma)),
                                 (x[i, j] - mu_1))) + 1 / (
                                               pow((2 * np.pi), (d / 2)) * np.sqrt(np.linalg.det(sigma))) * np.exp(
                -0.5 * np.matmul(np.matmul(np.transpose(x[i, j] - mu_2), np.linalg.inv(sigma)), (x[i, j] - mu_2))))


    return probability_distrib

# print(probability_distribution(np.array([x_0,x_1]).transpose(),mu_1,mu_2,sigma,sigma))



def discriminant_function(x_0,x_1, mu, sigma, a_priori):
    g = np.zeros([x_0.shape[0],x_0.shape[1]])

    for i in range(x_0.shape[0]):
        for j in range(x_1.shape[0]):
            g[i][j] = -0.5 * np.matmul(np.matmul(np.transpose(([x_0[i][j],x_1[i][j]] - mu)), np.linalg.inv(sigma)),
                                              ([x_0[i][j],x_1[i][j]] - mu)) + np.log(a_priori)

    return g

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

x = discriminant_function(x_0,x_1, mu_2, sigma, p_omega_2)
y = discriminant_function(x_0,x_1, mu_1, sigma, p_omega_1)

fig = plt.figure()
fig.add_subplot()
surf = ax.plot_surface(x_0,x_1,x)
surf = ax.plot_surface(x_0,x_1,y)

ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

def bayes_error(p_omega_1,p_omega_2,mu_1,mu_2,sigma):
    case_1 = discriminant_function(x, mu_1, sigma, p_omega_1)
    case_2 = discriminant_function(x, mu_2, sigma, p_omega_2)

    # case =
    pass