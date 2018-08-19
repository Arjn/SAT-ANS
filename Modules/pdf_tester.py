import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import inv


def process_func(x, h):
    """
    Holds the orbital propagation equations -  a RK4 method is used
    :param x: (estimated) state
    :param h: time step
    :return: new state
    """
    y0_ = x[0:3]
    k1 = np.multiply(h, derivs(y0_))
    k2 = np.multiply(h, derivs(y0_ + k1 / 2))
    k3 = np.multiply(h, derivs(y0_ + k2 / 2))
    k4 = np.multiply(h, derivs(y0_ + k3))
    v_y = x[3:6] + np.divide((k1 + np.multiply(2, k2) + np.multiply(2, k3) + k4), 6)
    y = y0_ + v_y * h
    out = np.array([y, v_y])

    return out.flatten()


def derivs(x):
    r = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
    dvx = - mu * x[0] / (r ** 3)
    dvy = - mu * x[1] / (r ** 3)
    dvz = - mu * x[2] / (r ** 3)

    return np.array([dvx, dvy, dvz])


def add_noise(X, Q):
    noise = np.diag(Q)
    y = [0,0,0,0,0,0]
    for x in X:
        y[list(X).index(x)] = x + np.random.normal(0, noise[list(X).index(x)])
    return(y)

state = [-6045, -3490, 2500, -3.457, 6.618, 2.533]
mu = 398600.44180000003
dt = 0.01
time_end = 50

true_state = []
X = state
true_state.append(X)
for i in range(0,int(time_end/dt)):
    X = process_func(X, dt)
    true_state.append(X)

true_state = np.array(true_state)
q = 0.001
Q = np.diag([0, 0, 0, q, q, q])
mean_error = []
num_iter = []
for iter in range(0,4):
    num_iterations = 10 ** iter
    num_iter.append(num_iterations)
    mean_monte_state =np.zeros(true_state.shape)
    for i in range(0,num_iterations):
        print(i)
        X = state
        noisy_state = []
        X = add_noise(X, Q)
        noisy_state.append(X)
        for j in range(0, int(time_end / dt)):
            X = process_func(X, dt)
            X = add_noise(X, Q)
            noisy_state.append(X)

        noisy_state = np.array(noisy_state)
        if i == 0:
            mean_monte_state = noisy_state
        for k in range(0,len(true_state)-1):
            for l in range(0,6):
                mean_monte_state[k,l] = np.mean([mean_monte_state[k,l], noisy_state[k,l]])
    mean_error.append(np.mean(abs(true_state[:,0:3] - mean_monte_state[:,0:3])))
plt.figure(1)
plt.loglog(num_iter, mean_error)
plt.ylabel('mean error [km]')
plt.xlabel('number of iterations')
plt.legend()
plt.show()