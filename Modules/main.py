# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 09:59:48 2018

@author: ArjenJ
Currently only 2-body orbitting has been implemented



"""

import OrbitModule
import SensorModule
import Navigation_Module
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import inv
from numpy.random import randn
from astropy import units as u
plt.interactive(False)
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

ref_time_mapper = {'s': u.s, 'min': u.min, 'hr': u.hr, 'yr': u.yr}

def zero_inv(Q):
    """
    This function is used in the FIM calculation process. For the gravitational dynamics, the noise comes from unknown
    force perturbations which affect the velocity directly - not the position. However to calculate the FIM, the inverse
    of the Q matrix is required. To get around this, the matrix is resized and only the noise contributing factors (non-
    zero) components are inverted.
    :param Q: noise matrix
    :return: inverted noise matrix
    """

    temp = np.diag([Q[3,3], Q[4,4], Q[5,5]])
    temp = inv(temp)
    return np.diag([Q[0,0], Q[1,1], Q[2,2], temp[0,0], temp[1,1],temp[2,2]])

class Main(object):
    def __init__(self, TIMING, ORBITAL, SENSORS, NAVIGATION):
        """
        :param TIMING: -Generates the timing aspects of the the model
                       -[timestep, simulation length]

        :param ORBITAL: - Defines the orbital aspects of the model
                        - [orbit definition, reference orbital body, relevant orbital parameters]
                        - defined orbital types are kepler, position/velocity and ephemeris

        :param SENSORS: - Defines the sensor aspects of the model
                        - [number of sensors, [type, [update rate, noise, library]]]

        :param NAVIGATION:  - Defines the navigational aspects of the model
                            - [update time step, initial state uncertanty, process noise,
                              std of estimated starting state, random_seed]

        """
        # ------GLOBAL TIMING ------
        self.global_timer = 0.0
        self.global_dt = TIMING[0]
        self.sim_length = TIMING[1]

        # ------ORBITAL ------
        self.orbit_type = ORBITAL[0]
        self.ref_body = ORBITAL[1]
        self.ref_time = ORBITAL[2]
        if self.orbit_type is 'kepler':
            self.orbit_info = ORBITAL[3]
        elif self.orbit_type is 'position_velocity':
            self.orbit_info = ORBITAL[4]
        elif self.orbit_type is 'ephemeris':
            raise Exception("Not implemented yet")
        else:
            raise NameError(ORBITAL[0])

        # ------SENSORS ------
        self.num_sensors = SENSORS[0]
        if self.num_sensors != len(SENSORS[1]):
            raise ValueError("Number of defined sensors (%d) not equal to sensor info (%d)!" % (self.num_sensors, len(SENSORS[1])))
        self.sensor_objs = []
        self.sensor_raw = []
        for i in SENSORS[1]:
            self.sensor_raw.append(i)

        # ------ NAVIGATION ------
        self.nav_dt = NAVIGATION[0]
        self.nav_state = []
        self.P = NAVIGATION[1]
        self.Q = NAVIGATION[2]
        self.nav_start_uncert = NAVIGATION[3]
        self.seed = NAVIGATION[4]
        np.random.seed(self.seed)
        self.nav_timer = 0

        self.MSE_P = []
        self.MSE_V = []

        if self.global_dt > self.nav_dt:
            raise Exception("Global time step %f larger than navigation time step %f" % (self.global_dt.value, self.nav_dt.value))

    def orbital(self):
        """
        Creates the orbital object and generates the ephemeris for the simulation run
        """
        self.Orbit = OrbitModule.MakeOrbit(self.orbit_type, self.orbit_info, self.ref_body, self.ref_time, self.global_dt
                                           , self.sim_length, self.Q)
        self.Orbit.makeEphem()

    def sensors(self):
        """

        """
        # add generalisation of the sensors
        # LIMITATION: If mutliple sensors are added of the same type, these will all have the same error characteristics
        i = 0
        for sens in self.sensor_raw:
            if sens[0] == 'spectrometer':
                check = add_sensor(self.sensor_objs, 'SensorModule.Spectrometer')
                if not check:
                    self.sensor_objs.append(SensorModule.Spectrometer(sens[1]))
            elif sens[0] == 'angle_sensor':
                self.sensor_objs.append(SensorModule.AngSensor(sens[1]))
            i += 1

    def navigation(self):
        '''
        ADD TEXT HERE
        :return:
        '''
        # [orbital body, start_state, filter]
        starting_conds = [self.ref_body, self.nav_state, self.P, self.Q]
        self.nav_module = Navigation_Module.NavModule(self.nav_dt, self.num_sensors, self.sensor_objs, starting_conds, self.seed)
        self.nav_module.makeFilter()

    def Rnorm_array(self, Q):
        out = []
        for i in np.diag(Q):
            out.append(np.random.normal(0,i))
        return out


    def run_simulation(self):
        '''
        ADD TEXT HERE
        :return:
        '''
        self.orbital() #generate orbital components
        self.state = self.Orbit.updateState(noise=True) #initialise the orbit state
        self.nav_state = [self.Orbit.ephem.state.r + np.multiply(np.random.normal(0, self.nav_start_uncert[0]), np.ones(3))*self.Orbit.ephem.state.r.unit,
                          self.Orbit.ephem.state.v + np.multiply(np.random.normal(0, self.nav_start_uncert[1]), np.ones(3))*self.Orbit.ephem.state.v.unit]
        #initialise the navigation initial state estimate
        print("SC estimated start state")
        print(self.nav_state)
        self.sensors() #generate the sensor objects
        self.navigation() #initialise the filter
        self.nav_module.filter_state = self.nav_module.ukf.x
        storage_true = []
        storage_filter = []
        timer_storage = []
        test_storage = []
        self.J = inv(self.nav_module.ukf.P)
        self.CRLB = []
        self.global_timer = 0*self.global_dt.unit
        F_bar = np.zeros([6,6])
        global_counter = 0
        navigation_counter = 0

        while self.global_timer <= self.sim_length:
            self.state = self.Orbit.updateState(noise=True)
            #currently propagating an analyitcal state and adding noise to what is considered the true state
            # self.Orbit.state = self.state

            for sens in self.sensor_objs:
                sens.internal_clock = sens.sensor_timer_counter*self.global_dt  # update the sensor internal clock for updates
                sens.observe(self.state, self.global_timer)
                sens.sensor_timer_counter += 1

            H_bar = []


            self.nav_module.obs_func(self.sensor_objs)
            if self.nav_timer >= self.nav_dt:
                navigation_counter = 0
                # print("\nUPDATE\t %f \n" % self.global_timer.value) if self.global_timer > 0 else 0
                self.nav_module.updateFilter(update=True)
                R = self.nav_module.ukf.R

                # for sens in self.sensor_objs:
                #     sens.observe(
                #         self.Orbit.state)  # the expectation value of a gaussian is the mean - as the noise is white
                    # gaussian, the mean will be the true value
                #     H_bar.append(sens.derivs)
                # H_bar = np.concatenate(np.array(H_bar))
                if self.global_timer > 0:
                    for sens in self.sensor_objs:
                        # sens.observe(self.state)
                        if sens.measurement_ready:
                            H_bar.append(sens.derivs)
                    if len(H_bar) > 0:
                        H_bar = np.concatenate(np.array(H_bar))


                    #------------- CRLB -----------
                        #self.J = (np.dot(H_bar.T, inv(R)).dot(H_bar))
                        # self.J =  (inv( self.Q + np.dot(F_bar.T, self.nav_module.ukf.P).dot(F_bar)) + np.dot(H_bar.T, inv(R)).dot(H_bar))
                        self.J = inv(self.Q) + np.dot(H_bar.T, inv(R)).dot(H_bar) - np.dot(np.dot(inv(self.Q), F_bar),
                                       inv(self.J + np.dot(F_bar.T, inv(self.Q)).dot(F_bar)), np.dot(F_bar.T, inv(self.Q)))
                        # test_storage.append(np.array(np.dot(F_bar.T, self.Q).dot(F_bar)).flatten())
                        self.CRLB.append(np.diag(1/(self.J)))


                F_bar = self.nav_module.dfdx(self.Orbit.state, self.nav_module.dt.value)  # dynamics deriviative for FIM
                temp = np.mean(np.sqrt((self.Orbit.state[0:3] - self.nav_module.filter_state[0:3])**2))
                self.MSE_P.append(temp)
                temp = np.mean(np.sqrt((self.Orbit.state[3:] - self.nav_module.filter_state[3:])**2))
                self.MSE_V.append(temp)
                test_storage.append(temp.tolist())
                timer_storage.append(self.global_timer.value)
            storage_true.append(self.Orbit.state)
            storage_filter.append(self.nav_module.filter_state)
            error = abs(self.Orbit.state - self.nav_module.filter_state)
            # print(error)


            # print(self.global_timer)
            global_counter += 1
            self.global_timer = self.global_dt*global_counter
            navigation_counter += 1
            self.nav_timer = self.global_dt*navigation_counter
        storage_filter = np.array(storage_filter)
        timer_storage = np.array(timer_storage)
        storage_true = np.array(storage_true)
        self.CRLB = np.array(self.CRLB)
        self.nav_module.covar_store = np.array(self.nav_module.covar_store)
        test_storage = np.array(test_storage)



        plt.figure(6)
        CRLB_P = pd.read_csv('CRLB_P.csv')
        CRLB_V = pd.read_csv('CRLB_V.csv')

        printer = ['x', 'y', 'z', 'vx', 'vy', 'vz']
        # for i in range(0,3):
        plt.plot(timer_storage, self.MSE_P[:], label='model error position', linewidth=0.5)
        print(np.mean(self.MSE_P))
        print(np.mean(self.MSE_V))

            # plt.plot(timer_storage, abs(self.nav_module.covar_store[:, i])*3, color='r', label='%s covar' % printer[i])
            # plt.plot(timer_storage, abs(self.nav_module.covar_store[:, i])*-3, color='r')
        plt.plot(timer_storage[1:], CRLB_P.values[:,0], label='MC CRLB')
        plt.figure(10)
        for i in range(0, 3):
            plt.semilogy(timer_storage[:], (self.CRLB[:, i]), label='analytical CRLB %s' % printer[i])
        plt.semilogy(timer_storage[1:], CRLB_P.values[:, 0], label='numerical CRLB')
        plt.title(' position')
        plt.legend()

        plt.figure(7)
        # for i in range(3,6):

        plt.plot(timer_storage[:], self.MSE_V[:], label='model error velocity', linewidth=0.5)

            # plt.plot(timer_storage, abs(self.nav_module.covar_store[:, i]), label='%s' % printer[i])
            # plt.plot(timer_storage, abs(self.nav_module.covar_store[:, i])*-1, label='%s' % printer[i])
        plt.plot(timer_storage[1:], CRLB_V.values[:,0], label='MC CRLB')
        plt.figure(9)
        for i in range(3, 6):
            plt.semilogy(timer_storage[:], (self.CRLB[:, i]), label='analytical CRLB %s' % printer[i])
            # plt.plot(timer_storage, storage_true[:, i], 'o', markersize=1, label=('filter'))
        plt.title('speed')
        plt.semilogy(timer_storage[1:], CRLB_V.values[:, 0], label='Numerical CRLB')
        plt.legend()

        # print("RMS pos : mean error = %f \t  RMS vel = %f" % (np.mean(abs(test_storage[:,0:3])),
        #                                                   np.mean(abs(test_storage[:, 3:]))))

        plt.figure(8)
        plt.title('covariances')
        for i in range(0,6):
            plt.semilogy(abs(self.nav_module.covar_store[:,i]), label=' Covar %s'%printer[i])
        plt.legend()


        plt.show()


def add_sensor(sensor_object, name):
    '''
    ADD TEXT HERE
    :param sensor_object:
    :param name:
    :return:
    '''
    check = False
    if len(sensor_object)>0:
        for obj in sensor_object:
           if obj is name:
               obj.params[0[0]] += 1
               check = True
    return check


def remove_units(state):
    '''
    ADD TEXT HERE
    :param state:
    :return:
    '''
    result = []
    for x in state:
        result.append(x.value)
    return result