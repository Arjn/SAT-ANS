import main
import numpy as np
from astropy import units as u
from numpy.random import randint
import copy as cp
import matplotlib.pyplot as plt

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
Simulation_length = 20000. * u.s

TIMING = [dt, Simulation_length]

# ------------ ORBITAL -------------------

Orbit_Type = 'kepler'  # 'kepler' 'position_velocity' 'ephemeris'
Reference_Body = 'earth'  # earth, mars, sun
Reference_Time = ["2018-01-01 00:00", 'tdb', 'iso']  # start epoch, timing reference, format]

# If using an ephemeris, link to file
File_Name = 'foo.txt'  # text file should be formatted appropriately TODO

# If the type kepler is chosen, change the values below:
a = 7136.635444 * u.km  # semi-major axis [km]
ecc = 0.3 * u.one  # eccentricity [-]
inc = 90. * u.deg  # inclination [deg]
raan = 175. * u.deg  # Right ascension of the ascending node [deg]
argp = 90. * u.deg  # Argument of perigee [deg]
nu = 178. * u.deg  # True anaomaly [deg]
kep = [a, ecc, inc, raan, argp, nu]

# if the type is position_velocity, need the position and velocity at starting
# epoch
state = [-6045 * u.km, -3490 * u.km, 2500 * u.km, -3.457 * u.km / u.s, 6.618 * u.km / u.s,
         2.533 * u.km / u.s]  # position then velocity

orbit_q = 1e-5

ORBITAL = [Orbit_Type, Reference_Body, Reference_Time, kep, state, orbit_q]

# -------------------- SENSORS ----------------

# format of the the sensor list: [ TYPE, [UPDATE RATE, VARIANCE, LIBRARY]]

# Spectrometer variance is error in measurement of spectrum (wavelength)
spectrometer = ['spectrometer', [10. * u.s, [1e-10], 'spectrometer_test.xml']]

# Angle variance is error in measurement of theta and phi
angle_sensor = ['angle_sensor', [10. * u.s, [4e-7, 4e-7], 'ang_sensor_lib.xml']]

# Sizing parameters for RPNAV:

alpha = 0.5  # - polarization parameter [-]
Ae = 100  # - Effective detection area [m^2]
v_rec = 1.4  # - receiver central frequency [GHz]
B = 400e6  # - Bandwidth [Hz]
atten = -40  # - main sidelobe attenuation [dB]
T_rec = 15  # - Receiver noise temperature [K]
update_rate = 10000 * u.s

radio_PNAV = ['radio_pulsar', [update_rate, alpha, Ae, v_rec, B, atten, T_rec], 'radio_pulsar_lib.xml']

# Sizing parameters for XPNAV:

A = 100 ** 2  # - Detector area [cm^2]
update_rate = 10000 * u.s

xray_PNAV = ['xray_pulsar', [update_rate, A], 'xray_pulsar_lib.xml']

number_sensors = 1
sensors = [xray_PNAV]

SENSORS = [number_sensors, sensors]

# ----------- NAVIGATION ----------------
starting_uncertanty = [10., 1.]
P = np.diag([100. ** 2, 100. ** 2, 100. ** 2, 10. ** 2, 10. ** 2, 10. ** 2])
nav_q = 1e-4  # white noise spectral density
dt_nav = 10. * u.s


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

sens = [[spectrometer], [spectrometer], [angle_sensor], [angle_sensor, spectrometer]]
num_sens = [1,1,1,2]
updates = [False, True, True, True]
global_storage = []

num_iter = 2
seedling = 2000
np.random.seed(seedling)

if __name__ == "__main__":
    import doctest

    doctest.testmod()
    for i in range(0,4):
        SENSORS = [num_sens[i], sens[i]]
        update = updates[i]

        covarx_store = []
        covary_store = []
        covarz_store = []
        covarvx_store = []
        covarvy_store = []
        covarvz_store = []
        mean_err_store = []

        for iter in range(0,num_iter):
            random_seed = int(abs(np.random.normal(0,seedling)))
            NAVIGATION = [dt_nav, P, nav_q, starting_uncertanty, random_seed]
            sim = main.Main(cp.deepcopy(TIMING), cp.deepcopy(ORBITAL), cp.deepcopy(SENSORS), cp.deepcopy(NAVIGATION),
                            cp.deepcopy(ONBOARD_CLOCK), cp.deepcopy(update))
            sim.run_simulation()
            covarx_store.append(sim.filter_covar[:,0])
            covary_store.append(sim.filter_covar[:, 1])
            covarz_store.append(sim.filter_covar[:, 2])
            covarvx_store.append(sim.filter_covar[:, 3])
            covarvy_store.append(sim.filter_covar[:, 4])
            covarvz_store.append(sim.filter_covar[:, 5])
            mean_err_store.append(sim.analysis.means)

        mean_err_store = np.array(mean_err_store)
        covarx_store = np.array(covarx_store)
        covary_store = np.array(covary_store)
        covarz_store = np.array(covarz_store)
        covarvx_store = np.array(covarvx_store)
        covarvy_store = np.array(covarvy_store)
        covarvz_store = np.array(covarvz_store)
        mean_covar = [np.mean(covarx_store, axis=0), np.mean(covary_store, axis=0) ,np.mean(covarz_store, axis=0),
                      np.mean(covarvx_store, axis=0), np.mean(covarvy_store, axis=0), np.mean(covarvz_store, axis=0)]
        mean_err = [np.mean(mean_err_store[:,0]), np.mean(mean_err_store[:,1]), np.mean(mean_err_store[:,2]),
                    np.mean(mean_err_store[:, 3]), np.mean(mean_err_store[:,4]), np.mean(mean_err_store[:,5])]

        global_storage.append([mean_err, mean_covar])

    print(len(global_storage))
plt.figure(1)
labels = ['X', 'Y', 'Z', 'Vx', 'Vy', 'Vz']
for i in range(0,6):
    plt.plot(['None', 'Spectrometer', 'Angle Sensor', 'Integrated'],
             [global_storage[0][0][i], global_storage[1][0][i], global_storage[2][0][i], global_storage[3][0][i]], label=labels[i])
    plt.ylabel('Error')
plt.legend()
plt.show()
#     MSEs_V.append(sim.MSE_V)
#     MSEs_P.append(sim.MSE_P)
# np.savetxt('MSE_P.csv', np.array(MSEs_P))
# np.savetxt('MSE_V.csv', np.array(MSEs_V))
