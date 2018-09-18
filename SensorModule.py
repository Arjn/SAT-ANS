# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 13:11:28 2018

@author: ArjenJ

General sensor module
library->observation eqns ->obs output
"""
from astropy.constants.codata2014 import c, k_B
import numpy as np
from astropy import units as u
from astropy.coordinates import solar_system
from numpy.random import randn
import xml.etree.cElementTree as ElementTree
from poliastro import coordinates
from poliastro.bodies import Earth, Mars, Sun
ref_body_mapper = {'sun': Sun, 'earth': Earth, 'mars': Mars}

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

def spher2cart(inc_az):
    """
    Spherical coordinate to cartesian converter
    """
    if len(inc_az) == 2:
        # [inc, az]
        x = np.sin(np.deg2rad(inc_az[0])) * np.cos(np.deg2rad(inc_az[1]))
        y = np.sin(np.deg2rad(inc_az[0])) * np.sin(np.deg2rad(inc_az[1]))
        z = np.cos(np.deg2rad(inc_az[0]))
    elif len(inc_az) == 3:
        # [r, inc az]
        x = inc_az[0]*np.sin(np.deg2rad(inc_az[1])) * np.cos(np.deg2rad(inc_az[2]))
        y = inc_az[0]*np.sin(np.deg2rad(inc_az[1])) * np.sin(np.deg2rad(inc_az[2]))
        z = inc_az[0]*np.cos(np.deg2rad(inc_az[1]))
    else:
        raise ValueError('length of spherical coord %d not 2 or 3' % len(inc_az))

    return np.array([x,y,z])

def norm_vec(vec):
    """
    :param vec: vector
    :return: normalised vector
    """
    return(np.sqrt(np.dot(vec,vec)))

def norm_angles(ang, angle_scaler):
    """
    Generates the normalised angle (making sure it remains mod 2pi
    :param ang: input angle IN RADIANS
    :return: normalised angle
    """

    ang_temp = np.mod(ang/angle_scaler, 2 * np.pi)
    if ang_temp > np.pi: #and abs(ang/angle_scaler - (ang_temp - np.pi)) > 1e-4:
        ang_temp -= 2 * np.pi
    return ang_temp*angle_scaler

class Xml2Dict(dict):
    """
    Reads an XML file and creates a dictionary from this
    - if units are present in the attributes of the observables - add the units flag
    """
    def __init__(self, parent, units=False):
        if not units:
            for child in parent.getchildren():
                dictA = {}
                for grandchild in child.getchildren():
                    dictA.update({grandchild.get('name'): eval(grandchild.get('type'))(grandchild.get('value'))})
                self.update({child.get('name'): dictA})
        else:
            for child in parent.getchildren():
                dictA = {}
                for grandchild in child.getchildren():
                    dictA.update({grandchild.get('name'):[eval(grandchild.get('type'))(grandchild.get('value')),
                                                          str(grandchild.get('units'))]})
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
                    2. standard deviation in wavelength measurement
                    3. sensor library
        """
        # ['spectrometer', [0.1 * u.s, [1e-11], 'spectrometer_test.xml']
        super().__init__()
        self.type = 'spectrometer'
        self.update_rate = input_params[0]
        self.l_std = input_params[1][0]
        self.R_params = []
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

    def h(self, x, derivs=False, epoch=None):
        """
        Observation equation used in the Kalman filter

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
                self.observed_wavelength[key] = 1/(1 + dot_prod / (c.value/1000)) * self.lib_params[key][
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
        temp = direction*((1/wavelength)/(c.value/1000))
        return np.array([0,0,0, temp[0], temp[1], temp[2]])


    def observe(self, state=np.array([6, 1]), time=None , epoch=None):
        """
        Observation equation for spectrometer. Through the base wavelengths read from the library and the current state
        of the spacecraft, the doppler equation is employed to calculate and then output the 'observed' wavelength
        TODO:
            convert singular wavelength to spectrum

        :param state: current state of the SC
        :param time: Current time
        :param epoch: current time in epoch - note used for the angle sensor
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
            self.R_params = []
            for key in self.lib_params:
                if self.lib_params[key]['parallax'] == 1:
                    self.observables_SCframe[key]['direction'] = self.lib_params[key][
                        'direction']  # SURELY ONLY REQUIRED IF DISTANCES ARE CONSIDERED??
                else:
                    dot_prod = np.dot(state[3:6], self.observables_SCframe[key]['direction'])
                    # self.derivs.append(self.dhdx(self.observables_SCframe[key]['direction'],  self.lib_params[key][
                    #     'base_wavelength_nm']))
                    self.observed_wavelength[key] = 1/(1 + dot_prod / (c.value/1000)) * self.lib_params[key][
                        'base_wavelength_nm'] + np.random.normal(0, self.l_std)  # Doppler equation scaled to nanometer wavelength
                    self.R_params.append(self.l_std**2)
            # self.derivs = np.array(self.derivs)
            self.measurement_ready = True
            self.measurements = self.observed_wavelength
            return self.observed_wavelength



class AngSensor(Sensor):
    """
    Sensor which determines the angles (and if required, range) to beacons

    KNOWN BUG:
        If using a beacon which is likely to change sign in observation angle over the course of a trajectory, or remain
        around zero - get instabilities due to angular singularity.

        test case:
                    a = 7136.635444 * u.km  # semi-major axis [km]
                    ecc = 0.3 * u.one# eccentricity [-]
                    inc = 90. * u.deg# inclination [deg]
                    raan = 175. * u.deg   # Right ascension of the ascending node [deg]
                    argp = 90. * u.deg # Argument of perigee [deg]
                    nu = 178. * u.deg # True anaomaly [deg]
                    kep = [a, ecc, inc, raan, argp, nu]

                    sensors = [['spectrometer', [10. * u.s, [0.1], 'spectrometer_test.xml']],
                    ['angle_sensor', [10. * u.s, [4e-6, 4e-6], 'ang_sensor_lib.xml']]]

                    beacons at [1,0,0], [0,1,0], [0,0,1]

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
        self.R_params = []
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

    def h(self, x, derivs=False, epoch=None):
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
            #run through the elements in the library and take a measurement from the s/c's persepctive
            self.dX = [self.lib_params[key]['direction'][0]-x[0], self.lib_params[key]['direction'][1]-x[1],
                       self.lib_params[key]['direction'][2]-x[2]]
            self.r_true[key] = (np.sqrt((self.dX[0]) ** 2 + (self.dX[1]) ** 2 + (self.dX[2]) ** 2))
            self.theta_true[key] = (np.arccos(self.dX[2] / np.sqrt((self.dX[0]) ** 2 + (self.dX[1]) ** 2 + (self.dX[2]) ** 2)))
            self.phi_true[key] =(np.arctan2(self.dX[1], self.dX[0]))

            mix = np.array([float(self.theta_true[key])*self.angle_scaler,
                            float(self.phi_true[key])* self.angle_scaler])
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
        out = []
        for ang in y:
            out.append(norm_angles(ang, self.angle_scaler))
            x = norm_angles(ang, self.angle_scaler)
            if x > 2*np.pi:
                raise ValueError('angle difference %f is greater than 2 pi!'%(x))
        return np.array(out)

    def z_mean(self, sigmas, Wm):
        """
        Function used by the KF to find the mean between observation and estimations or sigmas
        needed due to using angles.
        :param sigmas:
        :param Wm: weights for the angle estimates
        :return:
        """
        z_count = sigmas.shape[1]
        x = np.zeros(z_count)

        for z in range(0, z_count):
            # sum_sin1 = np.sum(np.dot(np.sin(sigmas[:, z]), Wm))
            # sum_cos1 = np.sum(np.dot(np.cos(sigmas[:, z]), Wm))
            #
            # x[z] = np.arctan2(sum_sin1, sum_cos1)
            sum_sin1 = np.sum(np.dot(np.sin(sigmas[:, z]/self.angle_scaler), Wm))
            sum_cos1 = np.sum(np.dot(np.cos(sigmas[:, z]/self.angle_scaler), Wm))

            x[z] = np.arctan2(sum_sin1, sum_cos1)*self.angle_scaler
        return x

    def observe(self, x, time=None, epoch=None):
        """
        Observation equation of the angle sensor. Determines the (body centered) atitude and azimuth to the beacons
        from the positions of the beacons in space
        TODO:
            add ephemeris input to the position of the beacons for if required
        :param epoch: current time in epoch - note used for the angle sensor
        :param x:
        :param time:
        :return:
        """

        self.measurements = {}
        self.theta = {}
        self.phi = {}
        self.R_params = []
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
                self.theta[key] =self.theta_true[key] + np.random.normal(0, self.theta_std)
                self.phi[key] =self.phi_true[key] + np.random.normal(0, self.phi_std)
                # mix = np.array([float(self.theta[key]), float(self.phi[key])])
                mix = np.array([np.mod(float(self.theta[key]), 2 * np.pi) * self.angle_scaler,
                                np.mod(float(self.phi[key]), 2 * np.pi) * self.angle_scaler])

                self.measurements[key] = mix
                self.R_params.append([self.theta_std**2, self.phi_std**2])
            # for i in range(0, len(self.r_true),2):
            #     self.storage_theta[i].append(self.measure[i])
            #     self.storage_phi[i].append(self.measure[i+1])
            self.measurement_ready = True
            return self.measurements


class PulsarSensor(Sensor):

    def __init__(self, input_params):
        super().__init__()
        self.refbody = []
        self.mu = []
        tree = ElementTree.parse(input_params[1])
        xml_tree_root = tree.getroot()
        self.lib_objs = Xml2Dict(xml_tree_root, units=True)  # TODO: CONVERSION OF POSITION VECTORS TO OTHER REF FRAMES
        for key in self.lib_objs:  # create a numpy array from the xml file
            long_lat = np.array([self.lib_objs[key]['long_ICRS'][0], self.lib_objs[key]['lat_ICRS'][0]])
            self.lib_objs[key]['direction'] = long_lat

    def which_pulsar(self ):
        self.pulsar_names = []
        for name in self.lib_objs:
            # do optimisation
            self.pulsar_names.append(name)

    def true_phase(self, time):
        """
        Reference phase is taken as initial phase at SSB at T0 (usually assumed to be zero)
        Note that the phase is kept as a cumulative phase (not mod 2pi) so that it can be used
        for both navigation methods.
        :param time: current time
        """
        for pulsar in self.pulsar_names:
            self.lib_objs[pulsar]['phase'][0] = self.lib_objs[pulsar]['ref_phase'][0] \
                                                + (1/self.lib_objs[pulsar]['P'][0])*time.to(u.s)

    def state_transform(self, time_ref, state, ref_body):
        #To do the time transform, first need the state in barycentric coords

        self.state_SBB = coordinates.body_centered_to_icrs(state[0:3]*u.km, state[3:]*(u.km/u.s), ref_body_mapper[ref_body], time_ref,
                                          ephemeris='de430')
        self.state_SBB = np.array([self.state_SBB[0].value, self.state_SBB[1].value]).flatten()
        self.pos_SSB = self.state_SBB[0:3]
        self.sun_pos_SBB = solar_system.get_body_barycentric_posvel('sun', time_ref, ephemeris='de430')
        self.sun_pos_SBB = self.sun_pos_SBB[0].xyz.value

    def time_transform(self, sim_time, mu):
        self.SC_pulsar_times = {}
        c_temp = (c.to(u.km / u.s)).value
        mu = mu.to(u.km**3/u.s**2).value
        for pulsar in self.pulsar_names:
            norm_r = norm_vec(self.state_SBB)
            norm_b = norm_vec(self.sun_pos_SBB)
            n = spher2cart(np.array([self.lib_objs[pulsar]['direction'][0],self.lib_objs[pulsar]['direction'][1]]))
            self.SC_pulsar_times[pulsar] = sim_time.value + np.dot(n, self.pos_SSB-self.sun_pos_SBB)/c_temp + (1/(2*c_temp*self.lib_objs[pulsar]['dist'][0]*3.086e19)) \
                                                * (-norm_r**2 + np.dot(n, self.pos_SSB) **2 + norm_b**2 - np.dot(self.sun_pos_SBB, n)**2) \
                                                + (2*mu/(c_temp**3))*np.log(abs(1 + (np.dot((self.pos_SSB - self.sun_pos_SBB), n)
                                                                              + norm_vec(self.pos_SSB - self.sun_pos_SBB))/
                                                                              np.dot(self.sun_pos_SBB, n) + norm_b))
        return self.SC_pulsar_times


class XraySensor(PulsarSensor):

    def __init__(self, input_params):
        super().__init__(input_params)
        self.type = 'XPNAV'
        self.int_time = input_params[0][0]
        self.detec_area = input_params[0][1]
        self.R_params = []
        self.vec_length = 1
        for key in self.lib_objs:  # create a numpy array from the xml file
            self.lib_objs[key]['Fx'][0] = self.lib_objs[key]['Fx'][0] / (8 * 1.6e-9) # conversion from ergs to photons
            self.lib_objs[key]['Bx'][0] = self.lib_objs[key]['Bx'][0] / (8 * 1.6e-9) # Assume 8 KeV bandwidth for all fluxes
            self.lib_objs[key]['Fx'][1] = 'J_s-1_cm-2'  # change units TODO: add specific bandwidth for all pulsars
            self.lib_objs[key]['Bx'][1] = 'J_s-1_cm-2'

    def timing_noise(self):
        """
        Sigma error specifically for xray pulsars
        :return: variance of timing estimates
        """
        self.sigma = []
        self.R_params = []
        self.SC_pulsar_times_noisy = {}
        for pulsar in self.pulsar_names:
            d = self.lib_objs[pulsar]['puls_width'][0]/self.lib_objs[pulsar]['P'][0]
            S = self.lib_objs[pulsar]['Fx'][0]*self.detec_area*self.lib_objs[pulsar]['puls_frac'][0]*self.int_time.value
            N = (self.lib_objs[pulsar]['Bx'][0] + self.lib_objs[pulsar]['Fx'][0]*(1-self.lib_objs[pulsar]['puls_frac'][0]))\
                * (self.detec_area*self.int_time.value*d)
            SNR = S/(np.sqrt(N + S))
            self.sigma.append(self.lib_objs[pulsar]['puls_width'][0]/2*SNR)
            self.R_params.append((self.lib_objs[pulsar]['puls_width'][0]/2*SNR))
            self.SC_pulsar_times_noisy[pulsar] = self.SC_pulsar_times[pulsar] + np.random.normal(0, np.sqrt( self.lib_objs[pulsar]['puls_width'][0]/2*SNR))
        return self.sigma

    def dhdx(self):
        """
        TODO: add the derivatives of the observation equations
        :return: 1
        """
        return 1

    def h(self, x, epoch=None, derivs=False):
        self.derivs = []
        self.output = []
        self.state_transform(epoch, x, self.refbody)
        self.output = self.time_transform(self.sim_time, self.mu)

        return self.output

    def observe(self, x, sim_time=None, epoch=None):
        self.derivs = []
        self.sim_time = sim_time
        self.measurements =[]
        if sim_time and self.internal_clock < self.int_time:
            #if time is being measured, and the current time is less than the update cycle, skip
            self.measurement_ready = False

        elif not sim_time or self.internal_clock >= self.int_time:
            self.which_pulsar()
            self.state_transform(epoch, x, self.refbody)
            self.time_transform(self.sim_time, self.mu)
            self.timing_noise()
            self.measurement_ready = True
            self.measurements = self.SC_pulsar_times_noisy
            self.dhdx()
            return self.measurements


class RadioSensor(PulsarSensor):

    def __init__(self, input_params):
        super().__init__(input_params)
        self.type = 'RPNAV'
        self.int_time = input_params[0][0]
        self.alpha = input_params[0][1]
        self.A = input_params[0][2]
        self.v_rec = input_params[0][3]
        self.B = input_params[0][4]
        self.atten = input_params[0][5]
        self.T_rec = input_params[0][6]
        self.R_params = []
        self.vec_length = 1
        self.beta = -1.8
        self.v_ref = 1.4


    def timing_noise(self):
        """
        Sigma error specifically for radio pulsars
        :return: variance of timing estimates
        """
        self.sigma = []
        self.R_params = []
        self.SC_pulsar_times_noisy = {}
        d = norm_vec((self.pos_SSB[0:3] * u.km).to(u.AU).value)
        for pulsar in self.pulsar_names:
            S = self.alpha*self.A*1e-26*self.lib_objs[pulsar]['Sp'][0]*1e-3*(self.v_rec/self.v_ref)**self.beta
            N = k_B.value*(self.T_rec + 2.7 + 6*self.v_rec**(-2.2) + (72*self.v_rec + 0.058)*self.A*10**(self.atten/10)*d**(-2))
            SNR = S/N
            err = 1/((2*np.pi)**2*SNR**2*self.lib_objs[pulsar]['Q'][0]*self.B*self.int_time.value)
            self.SC_pulsar_times_noisy[pulsar] = self.SC_pulsar_times[pulsar] + np.random.normal(0, np.sqrt(err))
            self.sigma.append(np.sqrt(err))
            self.R_params.append(np.sqrt(err))
        return self.sigma

    def dhdx(self):
        """
        TODO: add the derivatives of the observation equations
        :return: 1
        """
        return 1

    def h(self, x, epoch=None, derivs=False):
        self.derivs = []
        self.output = []
        self.state_transform(epoch, x, self.refbody)
        self.output = self.time_transform(self.sim_time, self.mu)
        return self.output


    def observe(self, x, sim_time=None, epoch=None):
        self.sim_time = sim_time
        self.measurements = []
        self.derivs = []
        if sim_time and self.internal_clock < self.int_time:
            #if time is being measured, and the current time is less than the update cycle, skip
            self.measurement_ready = False

        elif not sim_time or self.internal_clock >= self.int_time:
            self.which_pulsar()
            self.state_transform(epoch, x, self.refbody)
            self.time_transform(self.sim_time, self.mu)
            self.timing_noise()
            self.measurement_ready = True
            self.measurements = self.SC_pulsar_times_noisy
            self.dhdx()
            return self.measurements



class test_position_sensor(Sensor):
    """
   Generic test sensor which ouputs the true (positional) state of the S/C
    """

    def observe(self, x):
        self.measurements = x[0:3]
        return x[0:3]