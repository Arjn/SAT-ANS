# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 09:24:08 2018

@author: ArjenJ

This module is the orbital module for the INAV testbed blah blah blah
"""
import numpy as np
import matplotlib.pyplot as plt
import EphemerisModule

plt.ion()  # To immediately show plots

from astropy import units as u
from poliastro.bodies import Earth, Mars, Sun
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import cowell, kepler, RK4
from poliastro import coordinates
from astropy.time import Time
from astropy.coordinates import CartesianRepresentation, get_body_barycentric_posvel, \
    ICRS, GCRS, CartesianDifferential
from astropy.coordinates import solar_system_ephemeris
from collections import OrderedDict
from poliastro.plotting import plot
from numpy.random import normal


ref_body_mapper = {'sun': Sun, 'earth': Earth, 'mars': Mars}
ref_time_mapper = {'s': u.s, 'min': u.min, 'hr': u.hr, 'day': u.day, 'yr': u.yr}


class MakeOrbit(object):
    """ MakeOrbit class uses AstroPy and Poliastro to define and propagate an orbit in keplarian or r/v coords with a choice of
    central body. Output in r/v
    Parameters

    in_orbit_type : the type of orbit definition 'kepler' 'position_velocity' 'ephemeris'
    in_orbit_info : the parameters which define the orbit (kep elements, r/v state or ephemeris [TO DO])
    in_ref_body : the reference orbital body
    in_ref_time : the reference start time (in J2000 ref frame)
    in_timestep : the desired time step of propagation (in units of s, min, hr, yr)
    in_sim_length :the simulation length in specified units

    """
    def __init__(self, in_orbit_type, in_orbit_info, in_ref_body, in_ref_time, in_timestep, in_sim_length, noise):

        solar_system_ephemeris.set("de430")
        self.ref_ephem = "de430"


        self.orbitType = in_orbit_type
        self.orbitInfo = in_orbit_info
        self.refBody = ref_body_mapper[in_ref_body]
        self.refTime = in_ref_time
        self.dt = in_timestep
        self.simLength = in_sim_length
        # if(self.dt/self.simLength).decompose() == u.one:
        self.simStates = []
        self.endEpoch = 0
        self.startEpoch = 0
        self.noise = np.diag(noise)
        # else:
        #    print("Can't make simStates")
        self.i = 0
        self.barycentric_state = []

    def makeEphem(self):
        """
        Generates the ephemeris which will be propagated.
        Ephemeris type is dependent on the type of orbit. Currently only 2-body orbits are implemented, so all orbit
        types are converted to 2-body ephemerides
        """

        orbit_epoch = Time(self.refTime[0], scale=self.refTime[1], format=self.refTime[2])

        if self.orbitType is 'kepler':
            self.ephem = Orbit.from_classical(self.refBody, self.orbitInfo[0], self.orbitInfo[1],self.orbitInfo[2],
                                              self.orbitInfo[3], self.orbitInfo[4], self.orbitInfo[5], epoch=orbit_epoch)
            #plot(self.ephem)

        elif self.orbitType is 'position_velocity':
            self.ephem = Orbit.from_vectors(self.refBody, self.orbitInfo[0], self.orbitInfo[1], self.orbitInfo[2],
                                            self.orbitInfo[3], self.orbitInfo[4], self.orbitInfo[5], epoch=orbit_epoch)
            self.interim = self.ephem.state.to_classical()
            # self.ephem = Orbit.from_classical(self.refFrame, self.interim)
            self.ephem = Orbit.from_classical(self.refBody, self.interim.a, self.interim.ecc,
                                              self.interim.inc, self.interim.raan, self.interim.argp,
                                              self.interim.nu, epoch=orbit_epoch)
            #plot(self.ephem)

        else:
            print("error")
        self.startEpoch = self.ephem.epoch
        self.endEpoch = self.ephem.epoch + self.simLength
        #print(self.ephem.epoch.iso)
        #print(self.endEpoch.iso)

        # if isinstance(self.refBody, str):
        #     # Look up kernel chain for JPL ephemeris, based on name
        #     try:
        #         kernel_spec = BODY_NAME_TO_KERNEL_SPEC[self.refBody.lower()]
        #     except KeyError:
        #         raise KeyError("{0}'s position cannot be calculated with "
        #                        "the {1} ephemeris.".format(self.refBody, self.ref_ephem))
        # else:
        #     # otherwise, assume the user knows what their doing and intentionally
        #     # passed in a kernel chain
        #     kernel_spec = self.refBody
        self.ephem_kernel = EphemerisModule._get_kernel(self.ref_ephem)

    def add_noise(self, r,v):
        """
        Adds zero-mean gaussian noise to the ephemeris state with the user-defined noise parameters

        :param r: ephemeris range [x,y,z] with units
        :param v: ephemeris velocity
        :return:
        """
        r += [np.random.normal(0, self.noise[0]),np.random.normal(0, self.noise[1]),np.random.normal(0, self.noise[2])]*self.ephem.state.r.unit
        v += [np.random.normal(0, self.noise[3]),np.random.normal(0, self.noise[4]),np.random.normal(0, self.noise[5])] * self.ephem.state.v.unit

    def updateState(self, noise=True):
        """
        propagates the ephemeris by the timestep and updates the s/c state
        :return: True state of the spacecraft in cartestian coords
        """
        #print("start update")
        self.ephem = self.ephem.propagate(self.dt, method=RK4, rtol=1e-15)
        self.add_noise(self.ephem.r, self.ephem.v) if noise is True else None
        #print(Orbit.from_body_ephem(self.refBody, self.ephem.epoch))
        r = self.ephem.r.value
        v = self.ephem.v.value
        self.state = np.array([r[0], r[1], r[2], v[0], v[1], v[2]])
        # print(self.state)
        # self.barycentric_state = (EphemerisModule.body_centered_to_icrs(self.ephem.r, self.ephem.v, self.refBody, self.ephem.epoch,
        #                                         ephemeris=self.ref_ephem, Ephemkernel=self.ephem_kernel))
        self.simStates.append(self.state)
        return(self.state)

print(__doc__)

BODY_NAME_TO_KERNEL_SPEC = OrderedDict(
                                      (('sun', [(0, 10)]),
                                       ('mercury', [(0, 1), (1, 199)]),
                                       ('venus', [(0, 2), (2, 299)]),
                                       ('earth-moon-barycenter', [(0, 3)]),
                                       ('earth', [(0, 3), (3, 399)]),
                                       ('moon', [(0, 3), (3, 301)]),
                                       ('mars', [(0, 4)]),
                                       ('jupiter', [(0, 5)]),
                                       ('saturn', [(0, 6)]),
                                       ('uranus', [(0, 7)]),
                                       ('neptune', [(0, 8)]),
                                       ('pluto', [(0, 9)]))
                                      )