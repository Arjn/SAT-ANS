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
Reference_Body = 'sun'  # earth, mars, sun
Reference_Time = ["2018-01-01 00:00", 'tdb', 'iso']  # start epoch, timing reference, format]

# If using an ephemeris, link to file
File_Name = 'foo.txt'  # text file should be formatted appropriately TODO

# # # If the type kepler is chosen, change the values below:
a = (1*u.AU).to(u.km)  # semi-major axis [km]
ecc = 0.0167 * u.one  # eccentricity [-]
inc = 0. * u.deg  # inclination [deg]
raan = 0. * u.deg  # Right ascension of the ascending node [deg]
argp = 0. * u.deg  # Argument of perigee [deg]
nu = 0. * u.deg  # True anaomaly [deg]
kep = [a, ecc, inc, raan, argp, nu]

# a = 7136.6 * u.km  # semi-major axis [km]
# ecc = 0.3 * u.one# eccentricity [-]
# inc = 90. * u.deg# inclination [deg]
# raan = 175. * u.deg   # Right ascension of the ascending node [deg]
# argp = 90. * u.deg # Argument of perigee [deg]
# nu = 178. * u.deg # True anaomaly [deg]
# kep = [a, ecc, inc, raan, argp, nu]


# if the type is position_velocity, need the position and velocity at starting
# epoch
state = [-6045 * u.km, -3490 * u.km, 2500 * u.km, -3.457 * u.km / u.s, 6.618 * u.km / u.s,
         2.533 * u.km / u.s]  # position then velocity

orbit_q = 1e-5

ORBITAL = [Orbit_Type, Reference_Body, Reference_Time, kep, state, orbit_q]


# ----------- NAVIGATION ----------------
starting_uncertanty = [10., 1.]
P = np.diag([100. ** 2, 100. ** 2, 100. ** 2, 10. ** 2, 10. ** 2, 10. ** 2])
nav_q = 1e-5  # white noise spectral density
dt_nav = dt


# ------------ON-BOARD CLOCK---------------
# note that the clock state determines the error in the time
# clock state = [error, error drift, drift rate]
initial_state = [6e-6, 3e-9, 6e-11]
noise_spectra = [1.11e-9, 2.22e-20, 6.66e-24]
add_noise = False
ONBOARD_CLOCK = [initial_state, noise_spectra, add_noise]

# -------------------- SENSORS ----------------

# format of the the sensor list: [ TYPE, [UPDATE RATE, VARIANCE, LIBRARY]]

# Spectrometer variance is error in measurement of spectrum (wavelength)
spectrometer = ['spectrometer', [dt, [1e-9], 'spectrometer_test.xml']]

# Angle variance is error in measurement of theta and phi
angle_sensor = ['angle_sensor', [dt, [1e-7, 1e-7], 'ang_sensor_lib.xml']]

# Sizing parameters for RPNAV:

alpha = 0.5  # - polarization parameter [-]
Ae = 100  # - Effective detection area [m^2]
v_rec = 1.4  # - receiver central frequency [GHz]
B = 400e6  # - Bandwidth [Hz]
atten = -40  # - main sidelobe attenuation [dB]
T_rec = 15  # - Receiver noise temperature [K]
update_rate_r = 1500 * u.s

radio_PNAV = ['radio_pulsar', [update_rate_r, alpha, Ae, v_rec, B, atten, T_rec], 'radio_pulsar_lib.xml']

# Sizing parameters for XPNAV:

A = 100**2 # - Detector area [cm^2]
update_rate_x = 1500 * u.s

xray_PNAV = ['xray_pulsar', [update_rate_x, A], 'xray_pulsar_lib.xml']

# sens = [[spectrometer], [spectrometer], [angle_sensor], [angle_sensor, spectrometer]]
# sens = [[radio_PNAV], [radio_PNAV, spectrometer], [radio_PNAV, angle_sensor], [radio_PNAV, angle_sensor, spectrometer]]
sens = [[xray_PNAV], [xray_PNAV, spectrometer], [xray_PNAV, angle_sensor], [xray_PNAV, angle_sensor, spectrometer]]
file_name = f"q1e-5_XNAV_sun_{str(int(dt.value))}_{str(A)}cm2_{str(add_noise)}_CLOCK_sen_int_{str(int(update_rate_x.value))}s_{str(nav_q)}.txt"
# file_name = f"q1e-5_RNAV_sun_{str(dt.value)}_{str(Ae)}m2_{str(add_noise)}_CLOCK_sen_int_{str(int(update_rate_r.value))}s_{str(nav_q)}.txt"
# file_name = f"1q_NOPNAV_sun_{str(dt.value)}_{str(angle_sensor[1][1][0])}_{str(spectrometer[1][1][0])}_{str(add_noise)}_CLOCK_sen_int__{str(nav_q)}.txt"

# file_name = '300_iteration_ang_sens.txt'

num_sens = [1,2,2,3]
updates = [True, True, True, True]
global_storage = []

num_iter = 100
seedling = 4000
np.random.seed(seedling)
# print(sens[0])
# @profile(dump_stats=True)
def run():
    for i in range(0,4):
        SENSORS = [num_sens[i], sens[i]]
        update = updates[i]
        print('\n\n\n\n SENSOR COMBINATION %d \n\n\n\n' % (i + 1))

        mean_SQerr_store = [0, 0, 0, 0, 0, 0]
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

        mean_err_store[0] = sim.analysis.err[:, 0]
        mean_err_store[1] = sim.analysis.err[:, 1]
        mean_err_store[2] = sim.analysis.err[:, 2]
        mean_err_store[3] = sim.analysis.err[:, 3]
        mean_err_store[4] = sim.analysis.err[:, 4]
        mean_err_store[5] = sim.analysis.err[:, 5]

        mean_SQerr_store[0] = sim.analysis.err[:,0]**2
        mean_SQerr_store[1] = sim.analysis.err[:,1]**2
        mean_SQerr_store[2] = sim.analysis.err[:,2]**2
        mean_SQerr_store[3] = sim.analysis.err[:,3]**2
        mean_SQerr_store[4] = sim.analysis.err[:,4]**2
        mean_SQerr_store[5] = sim.analysis.err[:,5]**2

        M = sim.analysis.err
        S = 0

        div = 1
        fail = 0

        for iter in (range(1, num_iter)):
            print("%d / %d" % (iter, num_iter))
            random_seed = int(abs(np.random.normal(0, seedling)))
            NAVIGATION = [dt_nav, P, nav_q, starting_uncertanty, random_seed]
            sim = main.Main(cp.deepcopy(TIMING), cp.deepcopy(ORBITAL), cp.deepcopy(SENSORS), cp.deepcopy(NAVIGATION),
                            cp.deepcopy(ONBOARD_CLOCK), cp.deepcopy(update))

            try:
                sim.run_simulation()
                div +=1

                covarx_store = covarx_store + sim.filter_covar[:, 0]
                covary_store = covary_store + sim.filter_covar[:, 1]
                covarz_store = covarz_store  + sim.filter_covar[:, 2]
                covarvx_store = covarvx_store + sim.filter_covar[:, 3]
                covarvy_store = covarvy_store  + sim.filter_covar[:, 4]
                covarvz_store = covarvz_store  + sim.filter_covar[:, 5]

                freqx_store = freqx_store + sim.analysis.state_fft[0, :]
                freqy_store = freqy_store  + sim.analysis.state_fft[1, :]
                freqz_store = freqz_store + sim.analysis.state_fft[2, :]
                freqvx_store = freqvx_store + sim.analysis.state_fft[3, :]
                freqvy_store = freqvy_store  + sim.analysis.state_fft[4, :]
                freqvz_store = freqvz_store + sim.analysis.state_fft[5, :]

                mean_err_store[0] = sim.analysis.err[:, 0]
                mean_err_store[1] = sim.analysis.err[:, 1]
                mean_err_store[2] = sim.analysis.err[:, 2]
                mean_err_store[3] = sim.analysis.err[:, 3]
                mean_err_store[4] = sim.analysis.err[:, 4]
                mean_err_store[5] = sim.analysis.err[:, 5]

                mean_SQerr_store[0] += sim.analysis.err[:,0]**2
                mean_SQerr_store[1] += sim.analysis.err[:,1]**2
                mean_SQerr_store[2] += sim.analysis.err[:,2]**2
                mean_SQerr_store[3] += sim.analysis.err[:,3]**2
                mean_SQerr_store[4] += sim.analysis.err[:,4]**2
                mean_SQerr_store[5] += sim.analysis.err[:,5]**2

                #iterative standard deviation (non RMS!)
                M_old = M
                S_old = S
                M = M + (sim.analysis.err - M_old)/((iter-fail)+1)
                S = S_old + (sim.analysis.err - M_old)*(sim.analysis.err - M)


                # pickle1 = open("crash_saver2.txt", "wb")
                # pkl.dump(
                #     [sens[i], iter, [covarx_store, covary_store, covarz_store, covarvx_store, covarvy_store, covarvz_store],
                #      [freqx_store, freqy_store, freqz_store, freqvx_store, freqvy_store, freqvz_store]], pickle1)
                # pickle1.close()

            except:
                fail += 1

        print(fail)


        mean_SQerr_store = np.sqrt(np.array(mean_SQerr_store)/(num_iter-fail))
        mean_err_store = np.array(mean_err_store) / (num_iter - fail)
        # print(mean_err_store)

        std_store = np.sqrt(S/(num_iter-fail-1))

        covarx_store = np.array(covarx_store)/(num_iter-fail)
        covary_store = np.array(covary_store)/(num_iter-fail)
        covarz_store = np.array(covarz_store)/(num_iter-fail)
        covarvx_store = np.array(covarvx_store)/(num_iter-fail)
        covarvy_store = np.array(covarvy_store)/(num_iter-fail)
        covarvz_store = np.array(covarvz_store)/(num_iter-fail)

        freqx_store = np.array(freqx_store)/(num_iter-fail)
        freqy_store = np.array(freqy_store)/(num_iter-fail)
        freqz_store = np.array(freqz_store)/(num_iter-fail)
        freqvx_store = np.array(freqvx_store)/(num_iter-fail)
        freqvy_store = np.array(freqvy_store)/(num_iter-fail)
        freqvz_store = np.array(freqvz_store)/(num_iter-fail)

        mean_freq = [freqx_store, freqy_store, freqz_store,
                     freqvx_store, freqvy_store, freqvz_store]

        mean_covar = [covarx_store, covary_store, covarz_store,
                      covarvx_store, covarvy_store, covarvz_store]

        mean_err = [mean_SQerr_store, mean_err_store]

        global_storage.append([mean_err, mean_covar, std_store, mean_freq])

        pickle2 = open(file_name, "wb")
        pkl.dump(global_storage, pickle2)
        pickle2.close()

    # print(sens[i])
    print("clock noise = %s" % str(add_noise))

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
