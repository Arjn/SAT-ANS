import main
import numpy as np
from astropy import units as u
from numpy.random import randint
"""
SAT-ANS: Spacecraft Analysis Tool for Autonomous Navigation and Sizing

Author: Arjen Jongschaap

This tool assesses the navigation performance of selected sensors with navigation filter in a user defined s/c
trajectory.

To date the following has been implemented:
Orbit
    Keplerian orbit around Earth, Mars or Sun
    
Navigation
    UKF filter
    
Sensors
    Angle sensor
    Spectrometer
    Radio Pulsar Sensor
    X-ray Pulsar Sensor  
"""


# -------------TIMING ------------------

# TIMING UNITS MUST BE THE SAME!!
dt = 10. * u.s  # choice of s, min, hour, day, year
Simulation_length = 500000. * u.s

TIMING = [dt, Simulation_length]

# ------------ ORBITAL -------------------

Orbit_Type = 'kepler'  # 'kepler' 'position_velocity' 'ephemeris'
Reference_Body = 'earth'  # earth, mars, sun
Reference_Time = ["2018-01-01 00:00", 'tdb', 'iso']  # start epoch, timing reference, format]

# If using an ephemeris, link to file
File_Name = 'foo.txt'  # text file should be formatted appropriately TODO

# If the type kepler is chosen, change the values below:
a = 7136.635444 * u.km  # semi-major axis [km]
ecc = 0.3 * u.one# eccentricity [-]
inc = 90. * u.deg# inclination [deg]
raan = 175. * u.deg   # Right ascension of the ascending node [deg]
argp = 90. * u.deg # Argument of perigee [deg]
nu = 178. * u.deg # True anaomaly [deg]
kep = [a, ecc, inc, raan, argp, nu]

# if the type is position_velocity, need the position and velocity at starting
# epoch
state = [-6045 * u.km, -3490 * u.km, 2500 * u.km, -3.457 * u.km/u.s, 6.618 * u.km/u.s, 2.533 * u.km/u.s]  # position then velocity

ORBITAL = [Orbit_Type, Reference_Body, Reference_Time, kep, state]

# -------------------- SENSORS ----------------

# format of the the sensor list: [ TYPE, [UPDATE RATE, VARIANCE, LIBRARY]]

# Spectrometer variance is error in measurement of spectrum (wavelength)
spectrometer = ['spectrometer', [1000. * u.s, [0.1], 'spectrometer_test.xml']]

# Angle variance is error in measurement of theta and phi
angle_sensor = ['angle_sensor', [100. * u.s, [4e-9, 4e-9], 'ang_sensor_lib.xml']]

# Sizing parameters for RPNAV:

alpha = 0.5 # - polarization parameter [-]
Ae = 100    # - Effective detection area [m^2]
v_rec = 1.4 # - receiver central frequency [GHz]
B = 400e6   # - Bandwidth [Hz]
atten = -40 # - main sidelobe attenuation [dB]
T_rec = 15  # - Receiver noise temperature [K]
update_rate = 10000 * u.s

radio_PNAV = ['radio_pulsar', [update_rate, alpha, Ae, v_rec, B, atten, T_rec], 'radio_pulsar_lib.xml']

# Sizing parameters for XPNAV:

A = 100**2    # - Detector area [cm^2]
update_rate = 10000 * u.s

xray_PNAV = ['xray_pulsar', [update_rate, A], 'xray_pulsar_lib.xml']



number_sensors = 1
sensors = [xray_PNAV]

SENSORS = [number_sensors, sensors]


# ----------- NAVIGATION ----------------
starting_uncertanty = [10, 1]
P = np.diag([1**2, 1**2, 1**2, 0.1**2, 0.1**2, 0.1**2])
q = 1e-7 # white noise spectral density
dt_nav = 100. * u.s
random_seed = 20000
NAVIGATION = [dt_nav, P, q, starting_uncertanty, random_seed]

# ------------ON-BOARD CLOCK---------------
# note that the clock state determines the error in the time
# clock state = [error, error drift, drift rate]
initial_state = [3e-6, 3e-11, 6e-18]
noise_spectra = [1.11e-8, 2.22e-12, 6.66e-18]
add_noise = False
ONBOARD_CLOCK = [initial_state, noise_spectra, add_noise]

# [orbital body, timing, start_state, filter]

# MSEs_V = []
# MSEs_P = []
# for i in range(0,500):
# print(i)
# random_seed = randint(low=1, high=3000, size=1)

if __name__ == "__main__":
     import doctest
     doctest.testmod()
     sim = main.Main(TIMING, ORBITAL, SENSORS, NAVIGATION, ONBOARD_CLOCK)
     sim.run_simulation()

#     MSEs_V.append(sim.MSE_V)
#     MSEs_P.append(sim.MSE_P)
# np.savetxt('MSE_P.csv', np.array(MSEs_P))
# np.savetxt('MSE_V.csv', np.array(MSEs_V))
