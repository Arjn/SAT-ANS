import numpy as np
import matplotlib.pyplot as plt
import numpy.random as ran
import doctest


class ClockModel(object):

    """
    Contains a 3rd order clock model for use in SAT-ANS.
    Uses Newtonian format for error propagation (d2E/dt2 -> dE/dt -> E)
    Clock error state is equal to error, error drift, and error drift rate
    """

    def __init__(self, state, noise_spectrum, seed, M=None):

        ran.seed(seed)

        self.state = np.array(state)

        self.q1, self.q2, self.q3 = noise_spectrum


        self.M = M if M else None

    def update_time(self, t):
        """

        >>> Test.update_time(1)

        >>> print(Test.state[0])
        3.0000210705158313e-06
        >>> Test.update_time(10)

        >>> print(Test.state[0])
        3.000328222877443e-06
        >>> Test.update_time(100)

        >>> print(Test.state[0])
        3.0033270669398155e-06

        """

        self.Q = np.array(
            [[t ** 5 * self.q3 / 20 + t ** 3 * self.q2 / 3 + t * self.q1, t ** 4 * self.q3 / 8 + t ** 2 * self.q2 / 2, t ** 3 * self.q3 / 6],
             [t ** 4 * self.q3 / 8 + t ** 2 * self.q2 / 2, t ** 3 * self.q3 / 3 + t * self.q2, t ** 2 * self.q3 / 2],
             [t ** 3 * self.q3 / 6, t ** 2 * self.q3 / 2, t * self.q3]])

        phi = np.array([[1, t, t ** 2 / 2],
                        [0, 1, t],
                        [0, 0, 1]])
        B = np.array([[t, t ** 2 / 2, t ** 3 / 6],
                      [0, t, t ** 2 / 2],
                      [0, 0, t]])

        # AWGN = np.array([[ran.normal(0, self.Q[0, 0]), ran.normal(0, self.Q[1, 0]), ran.normal(0, self.Q[2, 0])],
        #                  [ran.normal(0, self.Q[0, 1]), ran.normal(0, self.Q[1, 1]), ran.normal(0, self.Q[2, 1])],
        #                  [ran.normal(0, self.Q[0, 2]), ran.normal(0, self.Q[1, 2]), ran.normal(0, self.Q[2, 2])]])

        AWGN = np.array([ran.normal(0,self.q1), ran.normal(0,self.q2), ran.normal(0,self.q3)])

        if self.M:
            self.state = np.dot(phi, self.state) + np.dot(B, self.M) + AWGN
        else:
            self.state = np.dot(phi, self.state) + AWGN

# Test parameters
# t0 = 0
# h0 = 8e-21
# h_1 = 2.9e-23
# h_2 = 6.1e-27
# delta = 0
# t = 0
# t_true = []
# t_est = []
# dt = 100
# deltas = []
#
# sf = (1/2)*h0
# sg = 2*np.pi**2*h_2
#
# xd = 0
# xb = 0
# q1 = 1.11e-11
# q2 = 2.22e-18
# q3 = 6.66e-21
#
# phi = np.array([[1,dt,(dt**2)/2],
#                [0, 1, dt],
#                [0, 0, 1]])
#
# Q = np.array([[dt**5*q3/20 + dt**3*q2/3 + dt*q1, dt**4*q3/8 + dt**2*q2/2, dt**3*q3/6],
#               [dt**4*q3/8 + dt**2*q2/2, dt**3*q3/3 + dt*q2, dt**2*q3/2],
#               [dt**3*q3/6,dt**2*q3/2, dt*q3]])
#
# state = np.array([3e-6,3e-11,6e-18])
# state_saver = []
# M = np.array([0,0,0])
# clock = ClockModel(state, [q1,q2,q3], 1000)
# #
# for i in range(1, 314712):
#     clock.update_time(dt)
#     state_saver.append(clock.state[0])
# #
# state_saver = np.array(state_saver)
# # deltas = np.array(deltas)
# # t_est = np.array(t_est)
# plt.figure(1)
# plt.plot(state_saver*3e5)
# plt.xlabel('Time [100s]')
# plt.ylabel('Position error [km]')
# plt.title('light time error from clock drift')
# plt.show()


if __name__ == "__main__":
    state = np.array([3e-6, 3e-11, 6e-18])
    q1 = 1.11e-11
    q2 = 2.22e-18
    q3 = 6.66e-21
    doctest.testmod(extraglobs={'Test': ClockModel(state, [q1,q2,q3], seed=1000)})
