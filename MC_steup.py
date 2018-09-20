import main
import numpy as np
from astropy import units as u
from numpy.random import randint
import copy as cp
import matplotlib.pyplot as plt
import pickle as pkl
import time
from tqdm import tqdm
import cProfile
from profilestats import profile

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
dt = 100. * u.s  # choice of s, min, hour, day, year
Simulation_length = 200000. * u.s

TIMING = [dt, Simulation_length]

# ------------ ORBITAL -------------------

Orbit_Type = 'kepler'  # 'kepler' 'position_velocity' 'ephemeris'
Reference_Body = 'earth'  # earth, mars, sun
Reference_Time = ["2018-01-01 00:00", 'tdb', 'iso']  # start epoch, timing reference, format]

# If using an ephemeris, link to file
File_Name = 'foo.txt'  # text file should be formatted appropriately TODO

# If the type kepler is chosen, change the values below:
a = (1*u.AU).to(u.km)  # semi-major axis [km]
ecc = 0.0167 * u.one  # eccentricity [-]
inc = 0. * u.deg  # inclination [deg]
raan = 0. * u.deg  # Right ascension of the ascending node [deg]
argp = 0. * u.deg  # Argument of perigee [deg]
nu = 0. * u.deg  # True anaomaly [deg]
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
spectrometer = ['spectrometer', [100. * u.s, [1e-10], 'spectrometer_test.xml']]

# Angle variance is error in measurement of theta and phi
angle_sensor = ['angle_sensor', [100. * u.s, [4e-7, 4e-7], 'ang_sensor_lib.xml']]

# Sizing parameters for RPNAV:

alpha = 0.5  # - polarization parameter [-]
Ae = 100  # - Effective detection area [m^2]
v_rec = 1.4  # - receiver central frequency [GHz]
B = 400e6  # - Bandwidth [Hz]
atten = -40  # - main sidelobe attenuation [dB]
T_rec = 15  # - Receiver noise temperature [K]
update_rate = 2000 * u.s

radio_PNAV = ['radio_pulsar', [update_rate, alpha, Ae, v_rec, B, atten, T_rec], 'radio_pulsar_lib.xml']

# Sizing parameters for XPNAV:

A = 100 ** 2  # - Detector area [cm^2]
update_rate = 2000 * u.s

xray_PNAV = ['xray_pulsar', [update_rate, A], 'xray_pulsar_lib.xml']

number_sensors = 1
sensors = [xray_PNAV]

SENSORS = [number_sensors, sensors]

# ----------- NAVIGATION ----------------
starting_uncertanty = [10., 1.]
P = np.diag([100. ** 2, 100. ** 2, 100. ** 2, 10. ** 2, 10. ** 2, 10. ** 2])
nav_q = 1e-4  # white noise spectral density
dt_nav = 100. * u.s


# ------------ON-BOARD CLOCK---------------
# note that the clock state determines the error in the time
# clock state = [error, error drift, drift rate]
initial_state = [3e-6, 3e-11, 6e-18]
noise_spectra = [1.11e-8, 2.22e-12, 6.66e-18]
add_noise = True
ONBOARD_CLOCK = [initial_state, noise_spectra, add_noise]

# [orbital body, timing, start_state, filter]

# MSEs_V = []
# MSEs_P = []
# for i in range(0,500):
# print(i)
# random_seed = randint(low=1, high=3000, size=1)

sens = [[radio_PNAV], [spectrometer], [angle_sensor], [angle_sensor, spectrometer]]
num_sens = [1,1,1,2]
updates = [False, True, True, True]
global_storage = []

num_iter = 1
seedling = 20000
np.random.seed(seedling)

@profile(dump_stats=True)
def run():
    for i in range(0, 1):
        SENSORS = [num_sens[i], sens[i]]
        update = updates[i]
        print('\n\n\n\n SENSOR COMBINATION %d \n\n\n\n' % (i + 1))

        mean_err_store = [0, 0, 0, 0, 0, 0]

        random_seed = int(abs(np.random.normal(0, seedling)))
        NAVIGATION = [dt_nav, P, nav_q, starting_uncertanty, random_seed]
        sim = main.Main(cp.deepcopy(TIMING), cp.deepcopy(ORBITAL), cp.deepcopy(SENSORS), cp.deepcopy(NAVIGATION),
                        cp.deepcopy(ONBOARD_CLOCK), cp.deepcopy(update))
        sim.run_simulation()

        covarx_store = sim.filter_covar[:, 0]
        covary_store = sim.filter_covar[:, 1]
        covarz_store = sim.filter_covar[:, 2]
        covarvx_store = sim.filter_covar[:, 3]
        covarvy_store = sim.filter_covar[:, 4]
        covarvz_store = sim.filter_covar[:, 5]

        freqx_store = sim.analysis.state_fft[0, :]
        freqy_store = sim.analysis.state_fft[1, :]
        freqz_store = sim.analysis.state_fft[2, :]
        freqvx_store = sim.analysis.state_fft[3, :]
        freqvy_store = sim.analysis.state_fft[4, :]
        freqvz_store = sim.analysis.state_fft[5, :]

        mean_err_store[0] = np.sqrt(sim.analysis.err[:,0]**2)
        mean_err_store[1] = np.sqrt(sim.analysis.err[:,1]**2)
        mean_err_store[2] = np.sqrt(sim.analysis.err[:,2]**2)
        mean_err_store[3] = np.sqrt(sim.analysis.err[:,3]**2)
        mean_err_store[4] = np.sqrt(sim.analysis.err[:,4]**2)
        mean_err_store[5] = np.sqrt(sim.analysis.err[:,5]**2)

        for iter in tqdm(range(1, num_iter)):
            random_seed = int(abs(np.random.normal(0, seedling)))
            NAVIGATION = [dt_nav, P, nav_q, starting_uncertanty, random_seed]
            sim = main.Main(cp.deepcopy(TIMING), cp.deepcopy(ORBITAL), cp.deepcopy(SENSORS), cp.deepcopy(NAVIGATION),
                            cp.deepcopy(ONBOARD_CLOCK), cp.deepcopy(update))


            sim.run_simulation()
            covarx_store = covarx_store * (1 - iter / (iter + 1)) + sim.filter_covar[:, 0] / (iter + 1)
            covary_store = covary_store * (1 - iter / (iter + 1)) + sim.filter_covar[:, 1] / (iter + 1)
            covarz_store = covarz_store * (1 - iter / (iter + 1)) + sim.filter_covar[:, 2] / (iter + 1)
            covarvx_store = covarvx_store * (1 - iter / (iter + 1)) + sim.filter_covar[:, 3] / (iter + 1)
            covarvy_store = covarvy_store * (1 - iter / (iter + 1)) + sim.filter_covar[:, 4] / (iter + 1)
            covarvz_store = covarvz_store * (1 - iter / (iter + 1)) + sim.filter_covar[:, 5] / (iter + 1)

            freqx_store = freqx_store * (1 - iter / (iter + 1)) + sim.analysis.state_fft[0, :] / (iter + 1)
            freqy_store = freqy_store * (1 - iter / (iter + 1)) + sim.analysis.state_fft[1, :] / (iter + 1)
            freqz_store = freqz_store * (1 - iter / (iter + 1)) + sim.analysis.state_fft[2, :] / (iter + 1)
            freqvx_store = freqvx_store * (1 - iter / (iter + 1)) + sim.analysis.state_fft[3, :] / (iter + 1)
            freqvy_store = freqvy_store * (1 - iter / (iter + 1)) + sim.analysis.state_fft[4, :] / (iter + 1)
            freqvz_store = freqvz_store * (1 - iter / (iter + 1)) + sim.analysis.state_fft[5, :] / (iter + 1)

            mean_err_store[0] = mean_err_store[0] * (1 - iter / (iter + 1)) + np.sqrt(sim.analysis.err[:,0] **2) / (iter + 1)
            mean_err_store[1] = mean_err_store[1] * (1 - iter / (iter + 1)) + np.sqrt(sim.analysis.err[:,1] **2) / (iter + 1)
            mean_err_store[2] = mean_err_store[2] * (1 - iter / (iter + 1)) + np.sqrt(sim.analysis.err[:,2] **2) / (iter + 1)
            mean_err_store[3] = mean_err_store[3] * (1 - iter / (iter + 1)) + np.sqrt(sim.analysis.err[:,3] **2) / (iter + 1)
            mean_err_store[4] = mean_err_store[4] * (1 - iter / (iter + 1)) + np.sqrt(sim.analysis.err[:,4] **2) / (iter + 1)
            mean_err_store[5] = mean_err_store[5] * (1 - iter / (iter + 1)) + np.sqrt(sim.analysis.err[:,5] **2) / (iter + 1)

            pickle1 = open("crash_saver2.txt", "wb")
            pkl.dump(
                [sens[i], iter, [covarx_store, covary_store, covarz_store, covarvx_store, covarvy_store, covarvz_store],
                 [freqx_store, freqy_store, freqz_store, freqvx_store, freqvy_store, freqvz_store]], pickle1)
            pickle1.close()

        mean_err_store = np.array(mean_err_store)
        covarx_store = np.array(covarx_store)
        covary_store = np.array(covary_store)
        covarz_store = np.array(covarz_store)
        covarvx_store = np.array(covarvx_store)
        covarvy_store = np.array(covarvy_store)
        covarvz_store = np.array(covarvz_store)

        freqx_store = np.array(freqx_store)
        freqy_store = np.array(freqy_store)
        freqz_store = np.array(freqz_store)
        freqvx_store = np.array(freqvx_store)
        freqvy_store = np.array(freqvy_store)
        freqvz_store = np.array(freqvz_store)

        mean_freq = [freqx_store, freqy_store, freqz_store,
                     freqvx_store, freqvy_store, freqvz_store]

        mean_covar = [covarx_store, covary_store, covarz_store,
                      covarvx_store, covarvy_store, covarvz_store]

        mean_err = [mean_err_store]

        global_storage.append([mean_err, mean_covar, mean_freq])

        pickle2 = open("int.txt", "wb")
        pkl.dump(global_storage, pickle2)
        pickle2.close()

    print(len(global_storage))

    return





if __name__ == "__main__":
    import doctest

    doctest.testmod()
    run()




plt.figure(1)
labels = ['X', 'Y', 'Z', 'Vx', 'Vy', 'Vz']
# for i in range(0,6):
#     plt.plot(['None', 'Spectrometer', 'Angle Sensor', 'Integrated'],
#              [global_storage[0][0][i], global_storage[1][0][i], global_storage[2][0][i], global_storage[3][0][i]], label=labels[i])
#     plt.ylabel('Error')
# plt.legend()
# plt.show()
#     MSEs_V.append(sim.MSE_V)
#     MSEs_P.append(sim.MSE_P)
# np.savetxt('MSE_P.csv', np.array(MSEs_P))
# np.savetxt('MSE_V.csv', np.array(MSEs_V))
