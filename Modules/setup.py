import main
import numpy as np
from astropy import units as u
from numpy.random import randint
'''
ADD TEXT HERE
'''
# -------------TIMING ------------------
# TIMING UNITS MUST BE THE SAME!!
dt = 1. * u.s  # choice of s, min, hour, day, year
Simulation_length = 5000. * u.s

TIMING = [dt, Simulation_length]
dt_nav = 10. * u.s
# ------------ ORBITAL -------------------
Orbit_Type = 'kepler'  # 'kepler' 'position_velocity' 'ephemeris'
Reference_Body = 'Earth'  # Éarth, Mars, SSB(NEED to add)
Reference_Time = ["2013-03-18 12:00", 'utc']  # start epoch, timing reference]

# If using an ephemeris, link to file
File_Name = 'foo.txt'  # text file should be formatted appropriately TODO

# If the type kepler is chosen, change the values below:
a = 7136.635444 * u.km  # semi-major axis [km]
ecc = 0.1 * u.one# eccentricity [-]
inc = 0. * u.deg# inclination [deg]
raan = 0. * u.deg   # Right ascension of the ascending node [deg]
argp = 0. * u.deg # Argument of perigee [deg]
nu = 0. * u.deg # True anaomaly [deg]
kep = [a, ecc, inc, raan, argp, nu]

# if the type is position_velocity, need the position and velocity at starting
# epoch
state = [-6045 * u.km, -3490 * u.km, 2500 * u.km, -3.457 * u.km/u.s, 6.618 * u.km/u.s, 2.533 * u.km/u.s]  # position then velocity

ORBITAL = [Orbit_Type, Reference_Body, Reference_Time, kep, state]

# ----------- SENSORS ----------------
# format of the the sensor list: [ TYPE, [UPDATE RATE, VARIANCE, LIBRARY]]
number_sensors = 2
sensors = [['spectrometer', [10. * u.s, [10.], 'spectrometer_test.xml']],
           ['angle_sensor', [10. * u.s, [4e-6, 4e-6], 'ang_sensor_lib.xml']]]

SENSORS = [number_sensors, sensors]


# ----------- NAVIGATION ----------------
dt_nav = 10.
starting_uncertanty = [1, 0.1]
P = np.diag([10**2, 10**2, 10**2, 1**2, 1**2, 1**2])
q = 0.0001
# Q = np.diag([0, 0, 0, q, q, q])
Q = np.multiply(np.array([[dt_nav**3/3, 0, 0, dt_nav**2/2, 0, 0],
     [0, dt_nav**3/3, 0, 0, dt_nav**2/2, 0],
     [0, 0, dt_nav**3/3, 0, 0, dt_nav**2/2],
     [dt_nav**2/2, 0, 0, dt_nav, 0, 0],
     [0, dt_nav**2/2, 0, 0, dt_nav, 0],
     [0, 0, dt_nav**2/2, 0, 0, dt_nav]]),q)

dt_nav = 10. * u.s
#np.diag([1e-1, 1e-1, 1e-1, 5e-2, 5e-2, 5e-2])
random_seed = randint(400)
seedling = 100
NAVIGATION = [dt_nav, P, Q, starting_uncertanty, random_seed]
# [orbital body, timing, start_state, filter]

# MSEs_V = []
# MSEs_P = []
# for i in range(0,500):
# print(i)
# random_seed = randint(low=1, high=3000, size=1)


sim = main.Main(TIMING, ORBITAL, SENSORS, NAVIGATION)
sim.run_simulation()
#     MSEs_V.append(sim.MSE_V)
#     MSEs_P.append(sim.MSE_P)
# np.savetxt('MSE_P.csv', np.array(MSEs_P))
# np.savetxt('MSE_V.csv', np.array(MSEs_V))
