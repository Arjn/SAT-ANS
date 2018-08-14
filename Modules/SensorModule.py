# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 13:11:28 2018

@author: ArjenJ

General sensor module
library->observation eqns ->obs output
"""
from astropy.constants.codata2014 import c
import numpy as np
from astropy import units as u
from numpy.random import randn
import xml.etree.cElementTree as ElementTree

class Sensor(object):
    """
    The parent class for the sensors.
    Contains residual and mean functions for use within the UKF
    """
    def __init__(self):
        self.measurement_ready = False
        self.internal_clock = 0
        self.FIM = []
        self.sensor_timer_counter = 0

    def residual_h(self, a, b):
        return a-b

    def z_mean (self, sigmas, Wm):

       return np.dot(Wm, sigmas)

    def FIM_calc(self, derivatives, std):
        FIM = np.zeros([6])
        for i in derivatives:
            FIM = FIM + np.multiply(i, i.T)
        return FIM / ((std)**2)


# def readLib(self, ):

def norm_angles(ang, angle_scaler):
    """
    Generates the normalised angle (making sure it remains mod 2pi
    :param ang: input angle IN RADIANS
    :return: normalised angle
    """
    ang_temp = np.mod(ang/angle_scaler, 2 * np.pi)
    if ang_temp > np.pi and abs(ang/angle_scaler - (ang_temp - 2 * np.pi)) > 1e-4:
        ang = ang_temp - 2 * np.pi
    return ang*angle_scaler

class Xml2Dict(dict):
    """
    Reads an XML file and creates a dictionary from this
    """
    def __init__(self, parent):
        for child in parent.getchildren():
            dictA = {}
            for grandchild in child.getchildren():
                dictA.update({grandchild.get('name'): eval(grandchild.get('type'))(grandchild.get('value'))})
            self.update({child.get('name'): dictA})


class Spectrometer(Sensor):
    """
    Spectrometer sensor type:
        determines velocity by detecting changes in wavelength due to doppler shift
    """
    def __init__(self, input_params):
        """
        Add text here
        :param input_params:
                    1. update rate
                    2.
        """
        # ['spectrometer', [0.1 * u.s, [1e-11], 'spectrometer_test.xml']
        super().__init__()
        self.type = 'spectrometer'
        self.update_rate = input_params[0]
        self.l_std = input_params[1][0]
        self.R_params = input_params[1]
        self.vec_length = 1
        tree = ElementTree.parse(input_params[2])
        xml_tree_root = tree.getroot()
        self.lib_params = Xml2Dict(xml_tree_root)  # TO DO: CONVERSION OF POSITION VECTORS TO OTHER REF FRAMES
        for key in self.lib_params: # create a numpy array from the xml file
            pos = np.array([self.lib_params[key]['direction.x'], self.lib_params[key]['direction.y'],
                               self.lib_params[key]['direction.z']])
            del (self.lib_params[key]['direction.x'])
            del (self.lib_params[key]['direction.y'])
            del (self.lib_params[key]['direction.z'])
            self.lib_params[key]['direction'] = pos
        self.observed_wavelength = dict.fromkeys(self.lib_params)

    def h(self, x, derivs=False):
        """
        Observation equation used in the kalaman filter

        :param x: estimated/real state of the spacecraft
        :return: the observed wavelengths
        """
        self.measurement_ready = False
        # first calculate position vector of the stars wrt the s/c
        self.observables_SCframe = self.lib_params.copy()  # currently just copy the whole library but will need to
        # optimise
        self.derivs = []
        for key in self.lib_params:
            if self.lib_params[key]['parallax'] == 1:
                self.observables_SCframe[key]['direction'] = self.lib_params[key][
                    'direction']  # SURELY ONLY REQUIRED IF DISTANCES ARE CONSIDERED??
            else:
                dot_prod = np.dot(x[3:6], self.observables_SCframe[key]['direction'])
                self.observed_wavelength[key] = (1 + dot_prod / (c.value/1000)) * self.lib_params[key][
                    'base_wavelength_nm']  # Doppler equation scaled to nanometer wavelength
                if derivs: self.derivs.append(self.dhdx(self.observables_SCframe[key]['direction'], self.lib_params[key][
                    'base_wavelength_nm']))
        self.derivs = np.array(self.derivs)
        self.measurement_ready = True
        self.measurements = self.observed_wavelength
        return self.observed_wavelength

    def dhdx(self, direction, wavelength):
        """
        :param direction: direction vector of the star;/doppler object
        :param wavelength: base wavelength of the object
        :return: the derivative (velocity only - position is zero) wrt the state
        """
        temp = direction*(wavelength/(c.value/1000))
        return np.array([0,0,0, temp[0], temp[1], temp[2]])


    def observe(self, state=np.array([6, 1]), time=None):
        """
        Observation equation for spectrometer. Through the base wavelengths read from the library and the current state
        of the spacecraft, the doppler equation is employed to calculate and then output the 'observed' wavelength
        TODO:
            convert singular wavelength to spectrum

        :param state: current state of the SC
        :param time: Current time
        :return: observed wavelength
        """
        self.measurement_ready = False
        # first calculate position vector of the stars wrt the s/c
        self.observables_SCframe = self.lib_params.copy()  # currently just copy the whole library but will need to
        # optimise
        # self.derivs = []
        if time and self.internal_clock < self.update_rate:
            self.measurement_ready = False

        elif not time or self.internal_clock >= self.update_rate:
            self.sensor_timer_counter = 0
            # print('spect ready')
            for key in self.lib_params:
                if self.lib_params[key]['parallax'] == 1:
                    self.observables_SCframe[key]['direction'] = self.lib_params[key][
                        'direction']  # SURELY ONLY REQUIRED IF DISTANCES ARE CONSIDERED??
                else:
                    dot_prod = np.dot(state[3:6], self.observables_SCframe[key]['direction'])
                    # self.derivs.append(self.dhdx(self.observables_SCframe[key]['direction'],  self.lib_params[key][
                    #     'base_wavelength_nm']))
                    self.observed_wavelength[key] = ((1 + dot_prod / (c.value/1000)) * self.lib_params[key][
                        'base_wavelength_nm']) + np.random.normal(0, self.l_std)  # Doppler equation scaled to nanometer wavelength
            # self.derivs = np.array(self.derivs)
            self.measurement_ready = True
            self.measurements = self.observed_wavelength
            return self.observed_wavelength



class AngSensor(Sensor):
    """
    Sensor which determines the angles (and if required, range) to beacons
    """
    def __init__(self, input_params):
        """
        ADD TEXT HERE
        :param input_params:
                1. update rate
                2. std for theta
                3. std for phi (generally the same)

        """
        super().__init__()
        self.vec_length = 2
        self.type = 'ang_sensor'
        self.r_std = 0#stds[0]
        self.theta_std = input_params[1][0] # std converted from arcseconds!
        self.phi_std = input_params[1][1] # std converted from arcseconds!
        self.R_params = [input_params[1][0], input_params[1][1]]
        self.r = {}
        self.r_true = {}
        self.theta_true = {}
        self.phi_true = {}
        self.theta = {}
        self.phi = {}
        self.update_rate = input_params[0]
        self.angle_scaler = 1



        tree = ElementTree.parse(input_params[2])
        xml_tree_root = tree.getroot()
        self.lib_params = Xml2Dict(xml_tree_root)  # TO DO: CONVERSION OF POSITION VECTORS TO OTHER REF FRAMES
        for key in self.lib_params:  # create a numpy array from the xml file
            self.pos = np.array([self.lib_params[key]['direction.x']*self.lib_params[key]['range_m'], self.lib_params[key]['direction.y']*self.lib_params[key]['range_m'],
                            self.lib_params[key]['direction.z']*self.lib_params[key]['range_m']])
            del (self.lib_params[key]['direction.x'])
            del (self.lib_params[key]['direction.y'])
            del (self.lib_params[key]['direction.z'])
            self.lib_params[key]['direction'] = self.pos

    def dhdx(self, dx, array):
        """
        Computes the derivative of the observation equations for use in the Fisher information matrix
        :param dx:
        :return:
        """
        out = []
        r = (np.sqrt((dx[0]) ** 2 + (dx[1]) ** 2 + (dx[2]) ** 2))
        dR = [(dx[0]/r), (dx[1]/r), (dx[2]/r), 0, 0, 0]

        dtheta = [(dx[2]*dx[0])/(r ** 2 * np.sqrt(dx[0] ** 2 + dx[1] ** 2)),
                           (dx[2] * dx[1]) / (r ** 2 * np.sqrt(dx[0] ** 2 + dx[1] ** 2)),
                           np.sqrt(dx[0] ** 2 + dx[1] ** 2)/r ** 2, 0, 0, 0]
        array.append(np.multiply(dtheta,self.angle_scaler))

        dphi =[(dx[1]/(dx[0] ** 2 + dx[1] ** 2)), (dx[1]/(dx[0] ** 2 + dx[1] ** 2)),
                         0, 0, 0, 0]
        array.append(np.multiply(dphi,self.angle_scaler))


    def FIM_calc(self, derivatives, std):
        FIM = np.zeros([6])
        for i in derivatives:
            k = 0
            for j in i:
                FIM = FIM + np.multiply(j, j.T)/(std[k] ** 2)
                k = 0 if k == 1 else 1
        return FIM

    def h(self, x, derivs=False):
        """
        observation equation used in the kalman filter
        :param x:
        :param derivs: as the FIM uses the state estimation, need a flag for this
        :return:
        """
        self.derivs = []
        self.output = {}
        self.r_true = {}
        self.theta_true = {}
        self.phi_true = {}
        for key in self.lib_params:
            #run through the elements in the library and take a measurement from the s/c's ppersepctive
            self.dX = [self.lib_params[key]['direction'][0]-x[0], self.lib_params[key]['direction'][1]-x[1],
                       self.lib_params[key]['direction'][2]-x[2]]
            self.r_true[key] = (np.sqrt((self.dX[0]) ** 2 + (self.dX[1]) ** 2 + (self.dX[2]) ** 2))
            self.theta_true[key] = (np.arccos(self.dX[2] / np.sqrt((self.dX[0]) ** 2 + (self.dX[1]) ** 2 + (self.dX[2]) ** 2)))*self.angle_scaler
            self.phi_true[key] =(np.arctan2(self.dX[1], self.dX[0]))*self.angle_scaler
            mix = np.array([float(self.theta_true[key]), float(self.phi_true[key])])
            self.output[key] = mix
            if derivs: self.dhdx(self.dX, self.derivs)

        #self.FIM = self.FIM_calc(self.derivs, [self.theta_std, self.phi_std])
        # print()
        #print(self.FIM)
        self.derivs = np.array(self.derivs)
        return self.output

    def residual_h(self, a, b):
        """
        Function used by the KF to find the residuals between observation and estimated
        needed due to using angles.
        :param a:
        :param b:
        :return:
        """
        if any(isinstance(el, list) for el in a):
            a = [item for sublist in a for item in sublist]
        if any(isinstance(el, list) for el in b):
            b = [item for sublist in b for item in sublist]

        y = a - b
        # data in format [[theta1, phi1], [theta2, phi2], ...]
        for ang in y:
            q = ang
            ang = norm_angles(ang, self.angle_scaler)
            # if q != y[i]:y
            #     print("NORMALISED ANGLES!", b, "!=", y[i])
        return y

    def z_mean(self, sigmas, Wm):
        """
        Function used by the KF to find the mean between observation and estimations or sigmas
        needed due to using angles.
        :param sigmas:
        :param Wm:
        :return:
        """
        z_count = sigmas.shape[1]
        x = np.zeros(z_count)

        for z in range(0, z_count):
            # sum_sin1 = np.sum(np.dot(np.sin(sigmas[:, z]/1000), Wm))
            # sum_cos1 = np.sum(np.dot(np.cos(sigmas[:, z]/1000), Wm))
            #
            # x[z] = np.arctan2(sum_sin1, sum_cos1)*1000
            sum_sin1 = np.sum(np.dot(np.sin(sigmas[:, z]/self.angle_scaler), Wm))
            sum_cos1 = np.sum(np.dot(np.cos(sigmas[:, z]/self.angle_scaler), Wm))

            x[z] = np.arctan2(sum_sin1, sum_cos1)*self.angle_scaler
        return x

    def observe(self, x, time=None):
        """
        Observation equation of the angle sensor. Determines the (body centered) atitude and azimuth to the beacons
        from the positions of the beacons in space
        TODO:
            add ephemeris input to the position of the beacons for if required
        :param x:
        :param time:
        :return:
        """

        self.measurements = {}
        self.theta = {}
        self.phi = {}
        if time and self.internal_clock < self.update_rate:
            #if time is being measured, and the current time is less than the update cycle, skip
            self.measurement_ready = False

        elif not time or self.internal_clock >= self.update_rate:
            #If not measuring time or the update cycle is ready
            self.h(x) #noise free observations
            # print('ang ready')
            self.sensor_timer_counter = 0 #reset timer counter
            for key in self.lib_params:
                #run through the elements of the library and add (white) noise to observations
                self.r[key] = self.r_true[key] + np.random.normal(0, self.r_std)
                self.theta[key] =norm_angles(self.theta_true[key] + np.random.normal(0, self.theta_std)*self.angle_scaler, self.angle_scaler)#*self.angle_scaler)
                self.phi[key] =norm_angles(self.phi_true[key] + np.random.normal(0, self.phi_std)*self.angle_scaler, self.angle_scaler)#*self.angle_scaler)
                mix = np.array([float(self.theta[key]), float(self.phi[key])])

                self.measurements[key] = mix
            # for i in range(0, len(self.r_true),2):
            #     self.storage_theta[i].append(self.measure[i])
            #     self.storage_phi[i].append(self.measure[i+1])
            self.measurement_ready = True
            return self.measurements



class Xray_Sensor(Sensor):

    def __init__(self, input_params):
        super().__init__()
        self.type = 'Xray'
        self.update_rate = input_params[0]
        self.l_std = input_params[1][0]
        self.R_params = input_params[1]
        self.vec_length = 1
        tree = ElementTree.parse(input_params[2])
        xml_tree_root = tree.getroot()
        self.lib_params = Xml2Dict(xml_tree_root)  # TO DO: CONVERSION OF POSITION VECTORS TO OTHER REF FRAMES
        for key in self.lib_params: # create a numpy array from the xml file
            pos = np.array([self.lib_params[key]['direction.x'], self.lib_params[key]['direction.y'],
                               self.lib_params[key]['direction.z']])
            del (self.lib_params[key]['direction.x'])
            del (self.lib_params[key]['direction.y'])
            del (self.lib_params[key]['direction.z'])
            self.lib_params[key]['direction'] = pos


class Radio_Sensor(Sensor):

    def __init__(self, input_params):
        super().__init__()
        self.type = 'Xray'
        self.update_rate = input_params[0]
        self.l_std = input_params[1][0]
        self.R_params = input_params[1]
        self.vec_length = 1
        tree = ElementTree.parse(input_params[2])
        xml_tree_root = tree.getroot()
        self.lib_params = Xml2Dict(xml_tree_root)  # TO DO: CONVERSION OF POSITION VECTORS TO OTHER REF FRAMES
        for key in self.lib_params:  # create a numpy array from the xml file
            pos = np.array([self.lib_params[key]['direction.x'], self.lib_params[key]['direction.y'],
                            self.lib_params[key]['direction.z']])
            del (self.lib_params[key]['direction.x'])
            del (self.lib_params[key]['direction.y'])
            del (self.lib_params[key]['direction.z'])
            self.lib_params[key]['direction'] = pos



class test_position_sensor(Sensor):
    """
   Generic test sensor which ouputs the true (positional) state of the S/C
    """

    def observe(self, x):
        self.measurements = x[0:3]
        return x[0:3]