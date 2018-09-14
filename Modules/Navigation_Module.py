from Filter import MerweScaledSigmaPoints, UKF
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from poliastro import constants
import ClockModule
import doctest


class NavModule(object):
    """
    Holds the navigation components, pertaining to the filter
    """
    def __init__(self, update_rate, number_sensors, Sensor_objects, starting_conds, clock_params, random_seed):
        """
        ADD TEXT HERE
        :param update_rate:
        :param number_sensors:
        :param Sensor_objects:
        :param starting_conds:
        :param random_seed:
        """
        np.random.seed(random_seed)
        self.sensor_num = number_sensors
        self.h_func = []
        self.dt = update_rate
        self.sensors = Sensor_objects
        self.start_conds = starting_conds #[orbital body, start_state, filter]
        self.covar_store = []
        self.time = []
        if starting_conds[0] is 'sun':
            self.mu = constants.GM_sun
        elif starting_conds[0] is 'earth':
            self.mu = constants.GM_earth
        elif starting_conds[0] is 'mars':
            self.mu = constants.GM_mars
        else:
            raise NameError(starting_conds[0])

        self.clock_add_noise = clock_params[2]
        self.clock = ClockModule.ClockModel(clock_params[0], clock_params[1], seed=random_seed)
        self.time_storage = []

        # make sure mu is in the right units - use the distance unit of start state, and time unit from timing
        dist_unit = starting_conds[1][0][1].unit ** 3
        inv_time_unit = (starting_conds[1][1][1].unit/starting_conds[1][0][1].unit) ** 2
        self.mu = self.mu.to(dist_unit * inv_time_unit)

    def update_clock(self, dt):
        # Add noise model for the clock
        if self.clock_add_noise:
            self.clock.update_time(dt.to(u.s).value)
            self.time += dt + self.clock.state[0] * dt.unit
        else:
            self.time += dt
        self.time_storage.append(self.time)

    # def process_func(self, x, t):
    #     """"
    #     Process for the dynamics used in the navigation filter
    #
    #     >>> Test.process_func(np.array([100,100,100,10,10,10]), 10).astype(int)
    #     array([      200,       200,       200, -25540511, -25540511, -25540511])
    #
    #     """
    #     # contains the process model for the s/c - currently only the 2-body problem
    #     y = np.zeros(6)
    #     r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    #     y[3] = x[3] - (self.mu.value*x[0]/(r**3) * t)
    #     y[4] = x[4] - (self.mu.value*x[1]/(r**3) * t)
    #     y[5] = x[5] - (self.mu.value*x[2]/(r**3) * t)
    #     y[0] = x[0] + (x[3] * t)
    #     y[1] = x[1] + (x[4] * t)
    #     y[2] = x[2] + (x[5] * t)
    #     return(y)

    def process_func(self, x, h):
        """
        Holds the orbital propagation equations -  a RK4 method is used
        :param x: (estimated) state
        :param h: time step
        :return: new state
        """
        y0_ = x[0:3]
        k1 = np.multiply(h, self.derivs(y0_))
        k2 = np.multiply(h, self.derivs(y0_ + k1 / 2))
        k3 = np.multiply(h, self.derivs(y0_ + k2 / 2))
        k4 = np.multiply(h, self.derivs(y0_ + k3))
        v_y = x[3:6] + np.divide((k1 + np.multiply(2, k2) + np.multiply(2, k3) + k4), 6)
        y = y0_ + v_y * h
        out = np.array([y, v_y])

        return out.flatten()
    #
    def derivs(self, x):
        r = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        dvx = - self.mu.value * x[0] / (r ** 3)
        dvy = - self.mu.value * x[1] / (r ** 3)
        dvz = - self.mu.value * x[2] / (r ** 3)

        return np.array([dvx, dvy, dvz])

    # def keplar2cartes(mu, RA, inc, AP, a, e, M, i):
    #
    #     # %Inputs:
    #     # % RA = Right ascention of ascending node (deg)
    #     # % inc = inclination of orbit (deg)
    #     # % AP = Argument of perigee (deg)
    #     # % a = semi major axis (m)
    #     # % e = eccentricity (-)
    #     # % M = mean anomaly (deg)
    #
    #     # %Outputs:
    #     # % x = position in inertial x-direction (m)
    #     # % y = position in inertial y-direction (m)
    #     # % z = position in inertial z-direction (m)
    #     # % xdot = velocity in inertial x-direction (m/s)
    #     # % ydot = velocity in inertial y-direction (m/s)
    #     # % zdot = velocity in inertial z-direction (m/s)
    #
    #     # %convert angular input from degrees to radians
    #
    #     conv = np.pi / 180
    #
    #     RA = RA * conv
    #     inc = inc * conv
    #     AP = AP * conv
    #     M = M
    #
    #     # %Gravitational parameter mu
    #
    #     # %calculate true anomaly from mean
    #     E_new = 0
    #     E_old = M
    #
    #     def E_find(x, e, M):
    #         return (x - e * np.sin(x) - M) ** 2
    #
    #     temp = (10 ** (i * 2))
    #     q = (1e-4 / temp)
    #     res = minimize(lambda x: E_find(x, e, M), x0=M, tol=q)
    #     E = res.x[0]
    #     theta = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
    #
    #     # %Calculate radius
    #     r = (a * (1 - e ** 2)) / (1 + e * np.cos(theta))
    #
    #     # %Calculate ang momentum, H
    #     H = np.sqrt(mu * a * (1 - e ** 2))
    #
    #     # %polar sections
    #     nu = r * np.cos(theta)
    #     eta = r * np.sin(theta)
    #
    #     # %Calculation of elements
    #
    #     L1 = np.cos(RA) * np.cos(AP) - np.sin(RA) * np.sin(AP) * np.cos(inc)
    #     L2 = -np.cos(RA) * np.sin(AP) - np.sin(RA) * np.cos(AP) * np.cos(inc)
    #     M1 = np.sin(RA) * np.cos(AP) + np.cos(RA) * np.sin(AP) * np.cos(inc)
    #     M2 = -np.sin(RA) * np.sin(AP) + np.cos(RA) * np.cos(AP) * np.cos(inc)
    #     N1 = np.sin(AP) * np.sin(inc)
    #     N2 = np.cos(AP) * np.sin(inc)
    #
    #     # %position transformations
    #
    #     x = L1 * nu + L2 * eta
    #     y = M1 * nu + M2 * eta
    #     z = N1 * nu + N2 * eta
    #
    #     # %velocity transformations
    #     xdot = (mu / H) * (-L1 * np.sin(theta) + L2 * (e + np.cos(theta)))
    #     ydot = (mu / H) * (-M1 * np.sin(theta) + M2 * (e + np.cos(theta)))
    #     zdot = (mu / H) * (-N1 * np.sin(theta) + N2 * (e + np.cos(theta)))
    #
    #     return np.array([x, y, z, xdot, ydot, zdot])

    # def dfdx(self, x, dt):
    #     diff = []
    #     r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    #
    #     diff.append([1,0,0,-(2*self.mu.value*x[0])/r ** 3 + (3*self.mu.value*(x[0]**3))/r ** 5,
    #                      (3 * self.mu.value * (x[1] ** 2)*x[0]) / r ** 5,
    #                     (3 * self.mu.value * (x[2] ** 2)*x[0]) / r ** 5])
    #
    #     diff.append([0,1,0, (3 * self.mu.value * (x[0] ** 2)*x[1]) / r ** 5,
    #                      -(2*self.mu.value*x[1])/r ** 3 + (3*self.mu.value*(x[1]**3))/r ** 5,
    #                      (3 * self.mu.value * (x[2] ** 2) * x[1]) / r ** 5])
    #
    #     diff.append([0,0,1, (3 * self.mu.value * (x[0] ** 2)*x[2]) / r ** 5,
    #                      (3 * self.mu.value * (x[1] ** 2) * x[2]) / r ** 5,
    #                      -(2 * self.mu.value * x[2]) / r ** 3 + (3 * self.mu.value * (x[2] ** 3)) / r ** 5])
    #
    #     diff.append([0,0,0,1,0,0])
    #
    #     diff.append([0,0,0,0,1,0])
    #
    #     diff.append([0,0,0,0,0,1])
    #
    #     return np.array(diff)


    def dfdx(self, x, dt):
        dt = 1
        diff = []
        r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)

        dxdX = [1,0,0,dt,0,0]
        diff.append(dxdX)

        dydX = [0,1,0,0,dt,0]
        diff.append(dydX)

        dzdX = [0,0,1,0,0,dt]
        diff.append(dzdX)

        dvxdX=[(-(self.mu.value)/r ** 3 + (3*self.mu.value*(x[0]**2))/r ** 5)*dt,
                ((3 * self.mu.value * x[1]*x[0]) / r ** 5)*dt,
                ((3 * self.mu.value * x[2] * x[0]) / r ** 5) * dt,
                1,0,0]
        diff.append(dvxdX)

        dvydX = [((3 * self.mu.value * x[1]*x[0]) / r ** 5)*dt,
                (-(self.mu.value) / r ** 3 + (3 * self.mu.value * (x[1] ** 2)) / r ** 5) * dt,
                ((3 * self.mu.value * x[2] * x[1]) / r ** 5) * dt,
                0,1,0]
        diff.append(dvydX)

        dvzdX = [((3 * self.mu.value * x[1]*x[2]) / r ** 5)*dt,
                ((3 * self.mu.value * x[2] * x[1]) / r ** 5) * dt,
                 (-(self.mu.value) / r ** 3 + (3 * self.mu.value * (x[2] ** 2)) / r ** 5) * dt,
                0,0,1]

        diff.append(dvzdX)

        return np.array(diff)

    def obs_func(self, sensors):
        """
        Collects all of the observation equations from the sensors, checks if the observation is ready and then stores
        the corresponding results. The noise matrices needed for the UKF are dynamically updated every call.

        :param sensors:
        """
        # if the observations are available, the relevant observation equations are added for the UKF
        self.h_func = []
        self.num_obs = []
        self.all_measurements = []
        self.residual_h = []
        self.z_mean = []
        self.R_params = []
        for sens in sensors:
            if sens.measurement_ready:
                self.h_func.append(sens.h) # holds the observation equations
                self.all_measurements.append(sens.measurements) # holds the measurements
                self.residual_h.append(sens.residual_h)
                self.z_mean.append(sens.z_mean)
                self.num_obs.append(len(sens.measurements.values())*sens.vec_length)
                self.R_params.extend(np.array(sens.R_params).flatten()) # creates an array containing the values
                # for the R matrix based on the sensor values\
                # TODO: REMOVE THE LIBRARY NAMES TO MAKE THE R MATRIX


    def makeFilter(self):
        """
        Generates the UKF based on the starting conditions. The sigma points have not been
        :return:
        """
        self.sigpoints =  MerweScaledSigmaPoints(n=6, alpha=1, beta=2., kappa=0)
        self.ukf = UKF(dim_x=6, dim_z=1, fx=self.process_func, hx=self.obs_func, dt=self.dt.value,
                  points=self.sigpoints)
        temp = []
        temp.extend(self.start_conds[1][0].value)
        temp.extend(self.start_conds[1][1].value)
        self.ukf.x = np.array(temp)
        self.ukf.P = self.start_conds[2]
        self.ukf.Q = self.start_conds[3]

    def updateFilter(self, update=True):
        """
        ADD TEXT HERE
        :return:
        """
        self.ukf.residual_x = None
        self.ukf.dim_z = sum(self.num_obs) # total number of observations per sensor, summed
        self.ukf.R = np.diag(np.array(self.R_params))
        self.ukf.hx = self.h_func
        self.ukf.residual_z = self.residual_h
        self.ukf.z_mean = self.z_mean
        self.ukf.residual_x = np.subtract
        # print('num measurements = %d' % self.ukf.dim_z)
        self.ukf.predict()
        self.ukf.update(self.all_measurements, self.num_obs, clock=self.time) if update is True else None
        self.filter_state = self.ukf.x
        diag_elem = np.diag(self.ukf.P)
        self.covar_store.append(np.array([i for i in diag_elem]))
        #print(self.dfdx(self.ukf.x, self.dt.value))


def repeat_check(string_array, string):
    """
    Check if an element in an array of strings is repeated
    :param string_array:
    :param string:
    :return:
    """
    for str in string_array:
        if str == string:
            return True




def h_observer(x, marks):
    """Measurement function -
    measuring only position"""
    estimated_obs_angles = []
    for i in range(0, len(marks)):
        dX = [marks[i][0] - x[0], marks[i][1] - x[1], marks[i][2] - x[2]]
        r = np.sqrt(dX[0]**2 + dX[1]**2 + dX[2]**2)
        theta = np.arccos(dX[2]/r)
        phi = np.arctan2(dX[1], dX[0])
        estimated_obs_angles.extend([theta, phi])
    #print(np.array(estimated_obs_angles))
    return np.array(estimated_obs_angles)


if __name__ == "__main__":
    doctest.testmod(extraglobs={'Test': NavModule(10*u.s, 0, [], ['Sun', [[418.92428299, 74.17981514, -9167.91088469] * u.km,
                                                                          [-8.47776753, -2.5416425 , -3.28798827] *(u.km/u.s)],
                                                                  [0], [0]], 200)})