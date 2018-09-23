import numpy as np
import matplotlib.pyplot as plt
import EphemerisModule
from astropy import units as u
from poliastro.bodies import Earth, Mars, Sun
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import cowell, kepler, RK4, mean_motion
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris
from poliastro import constants
import pickle as pkl


def norm(vec):
    return np.sqrt(np.dot(vec,vec))

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
    def __init__(self, in_orbit_type, in_orbit_info, in_ref_body, in_ref_time, in_timestep, in_sim_length, noise, intergrator):

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
        self.noise = np.diag(noise) if noise != 0 else None

        self.i = 0
        self.barycentric_state = []
        self.inter = intergrator
        self.M = []
        self.e = []

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

        elif self.orbitType is 'position_velocity':
            self.ephem = Orbit.from_vectors(self.refBody, self.orbitInfo[0], self.orbitInfo[1], self.orbitInfo[2],
                                            self.orbitInfo[3], self.orbitInfo[4], self.orbitInfo[5], epoch=orbit_epoch)
            self.interim = self.ephem.state.to_classical()
            # self.ephem = Orbit.from_classical(self.refFrame, self.interim)
            self.ephem = Orbit.from_classical(self.refBody, self.interim.a, self.interim.ecc,
                                              self.interim.inc, self.interim.raan, self.interim.argp,
                                              self.interim.nu, epoch=orbit_epoch)

        else:
            print("error")
        self.startEpoch = self.ephem.epoch
        self.endEpoch = self.ephem.epoch + self.simLength
        #print(self.ephem.epoch.iso)
        #print(self.endEpoch.iso)

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

    def updateState(self, noise=False):
        """
        propagates the ephemeris by the timestep and updates the s/c state
        :return: True state of the spacecraft in cartestian coords
        """
        self.ephem = self.ephem.propagate(self.dt, method=self.inter, rtol=1e-15)
        self.add_noise(self.ephem.r, self.ephem.v) if noise is True else None
        r = self.ephem.r.to(u.m).value
        v = self.ephem.v.to(u.m/u.s).value
        self.state = np.array([r[0], r[1], r[2], v[0], v[1], v[2]])
        self.simStates.append(self.state)
        return(self.state)


    def cart_to_mean_motion(self, t):
        r = self.state[0:3]
        v = self.state[3:]
        self.e.append(norm(((norm(v)**2 - self.ephem.attractor.k.value/norm(r))*r - np.dot(r,v)*v)/self.ephem.attractor.k.value))
        a = (-self.ephem.attractor.k.value)/(2*((norm(v)**2)/2 - self.ephem.attractor.k.value/norm(r)))
        # E = np.arccos((norm(r)/a - 1)/e)
        self.M.append(np.sqrt(self.ephem.attractor.k.value/abs(a)**3)*t)


Orbit_Type = 'kepler'  # 'kepler' 'position_velocity' 'ephemeris'
Reference_Body = 'earth'  # earth, mars, sun
Reference_Time = ["2018-01-01 00:00", 'tdb', 'iso']  # start epoch, timing reference, format]

# If the type kepler is chosen, change the values below:
a = 8000e3 * u.m  # semi-major axis [km]
ecc = 0.75 * u.one# eccentricity [-]
inc = 65. * u.deg# inclination [deg]
raan = 0. * u.deg   # Right ascension of the ascending node [deg]
argp = 0. * u.deg # Argument of perigee [deg]
nu = 0. * u.deg # True anaomaly [deg]
kep = [a, ecc, inc, raan, argp, nu]

dt = 10. * u.s  # choice of s, min, hour, day, year
Simulation_length = 200000. * u.s



if Reference_Body is 'sun':
    mu = constants.GM_sun
elif Reference_Body is 'earth':
    mu = constants.GM_earth
elif Reference_Body is 'mars':
    mu = constants.GM_mars
else:
    raise NameError(Reference_Body)

n = np.sqrt(mu.value/a.value**3)
M_true = []
time = []
times = [1*u.s, 10*u.s, 100*u.s, 1000*u.s, 10000*u.s]
times_graph = [1,10,100, 1000,10000]

mean_cowell_err_M = []
mean_kep_err_M = []
mean_RK4_err_M = []
mean_mean_motion_err_M = []

std_cowell_err_M = []
std_kep_err_M = []
std_RK4_err_M = []
std_mean_motion_err_M = []

mean_cowell_err_e = []
mean_kep_err_e = []
mean_RK4_err_e = []
mean_mean_motion_err_e = []

std_cowell_err_e = []
std_kep_err_e = []
std_RK4_err_e = []
std_mean_motion_err_e = []

for dt in times:

    M_true = []
    time = []

    ephem_cowell = MakeOrbit(Orbit_Type, kep, Reference_Body, Reference_Time, dt
                                               , Simulation_length, 0, cowell)
    ephem_kep = MakeOrbit(Orbit_Type, kep, Reference_Body, Reference_Time, dt
                                               , Simulation_length, 0, kepler)
    ephem_RK4 = MakeOrbit(Orbit_Type, kep, Reference_Body, Reference_Time, dt
                                               , Simulation_length, 0, RK4)
    ephem_mean_motion = MakeOrbit(Orbit_Type, kep, Reference_Body, Reference_Time, dt
                                               , Simulation_length, 0, mean_motion)
    ephem_cowell.makeEphem()
    ephem_kep.makeEphem()
    ephem_RK4.makeEphem()
    ephem_mean_motion.makeEphem()

    for i in range(1,int(Simulation_length.value/dt.value)):
        print("%d / %d"%(i, int(Simulation_length.value/dt.value)))
        time.append(i*dt.value)
        M_true.append(n*i*dt.value)
        ephem_cowell.updateState()
        ephem_cowell.cart_to_mean_motion(i*dt.value)
        ephem_kep.updateState()
        ephem_kep.cart_to_mean_motion(i*dt.value)
        ephem_RK4.updateState()
        ephem_RK4.cart_to_mean_motion(i*dt.value)
        ephem_mean_motion.updateState()
        ephem_mean_motion.cart_to_mean_motion(i*dt.value)


    M_true = np.array(M_true)
    ephem_cowell.M = np.array(ephem_cowell.M)
    ephem_kep.M = np.array(ephem_kep.M)
    ephem_RK4.M = np.array(ephem_RK4.M)
    ephem_mean_motion.M = np.array(ephem_mean_motion.M)

    ephem_cowell.e = np.array(ephem_cowell.e)
    ephem_kep.e = np.array(ephem_kep.e)
    ephem_RK4.e = np.array(ephem_RK4.e)
    ephem_mean_motion.e = np.array(ephem_mean_motion.e)

    mean_cowell_err_M.append(np.mean(abs(M_true - ephem_cowell.M)))
    mean_kep_err_M.append(np.mean(abs(M_true - ephem_kep.M)))
    mean_RK4_err_M.append(np.mean(abs(M_true - ephem_RK4.M)))
    mean_mean_motion_err_M.append(np.mean(abs(M_true - ephem_mean_motion.M)))

    std_cowell_err_M.append(np.std(abs(M_true - ephem_cowell.M)))
    std_kep_err_M.append(np.std(abs(M_true - ephem_kep.M)))
    std_RK4_err_M.append(np.std(abs(M_true - ephem_RK4.M)))
    std_mean_motion_err_M.append(np.std(abs(M_true - ephem_mean_motion.M)))

    mean_cowell_err_e.append(np.mean(abs(np.subtract(ephem_cowell.e,ecc))))
    mean_kep_err_e.append(np.mean(abs(np.subtract(ephem_kep.e,ecc))))
    mean_RK4_err_e.append(np.mean(abs(np.subtract(ephem_RK4.e,ecc))))
    mean_mean_motion_err_e.append(np.mean(abs(np.subtract(ephem_mean_motion.e,ecc))))

    std_cowell_err_e.append(np.std(abs(np.subtract(ephem_cowell.e,ecc))))
    std_kep_err_e.append(np.std(abs(np.subtract(ephem_kep.e,ecc))))
    std_RK4_err_e.append(np.std(abs(np.subtract(ephem_RK4.e,ecc))))
    std_mean_motion_err_e.append(np.std(abs(np.subtract(ephem_mean_motion.e, ecc))))

mean_cowell_err_M = np.array(mean_cowell_err_M)
mean_kep_err_M = np.array(mean_kep_err_M)
mean_RK4_err_M = np.array(mean_RK4_err_M)
mean_mean_motion_err_M = np.array(mean_mean_motion_err_M)

std_cowell_err_M = np.array(std_cowell_err_M)
std_kep_err_M = np.array(std_kep_err_M)
std_RK4_err_M = np.array(std_RK4_err_M)
std_mean_motion_err_M = np.array(std_mean_motion_err_M)

mean_cowell_err_e = np.array(mean_cowell_err_e)
mean_kep_err_e = np.array(mean_kep_err_e)
mean_RK4_err_e = np.array(mean_RK4_err_e)
mean_mean_motion_err_e = np.array(mean_mean_motion_err_e)

std_cowell_err_e = np.array(std_cowell_err_e)
std_kep_err_e = np.array(std_kep_err_e)
std_RK4_err_e = np.array(std_RK4_err_e)
std_mean_motion_err_e = np.array(std_mean_motion_err_e)

global_storage = [[mean_cowell_err_M, std_cowell_err_M, mean_cowell_err_e, std_cowell_err_e],
                  [mean_kep_err_M, std_kep_err_M, mean_kep_err_e, std_kep_err_e],
                  [mean_RK4_err_M, std_RK4_err_M, mean_RK4_err_e, std_RK4_err_e],
                  [mean_mean_motion_err_M, std_mean_motion_err_M, mean_mean_motion_err_e, std_mean_motion_err_e]]

pickle2 = open("orb_verif_8000km_e_0_75_i_65_cow_kep_rk4_MM.txt", "wb")
pkl.dump(global_storage, pickle2)
pickle2.close()


plt.figure(1)
plt.scatter(times_graph, mean_cowell_err_M, label='Cowell')
plt.scatter(times_graph, mean_kep_err_M, label='kepler')
plt.scatter(times_graph, mean_RK4_err_M, label='RK4')
plt.scatter(times_graph, mean_mean_motion_err_M, label='mean motion')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.grid()
plt.tight_layout()

plt.xlabel('time [s]')
plt.ylabel('Mean motion error [rad]')

plt.figure(2)
plt.scatter(times_graph, mean_cowell_err_e, label='Cowell')
plt.scatter(times_graph, mean_kep_err_e, label='kepler')
plt.scatter(times_graph, mean_RK4_err_e, label='RK4')
plt.scatter(times_graph, mean_mean_motion_err_e, label='mean motion')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.grid()
plt.tight_layout()

plt.xlabel('time [s]')
plt.ylabel('eccentricity error [rad]')



plt.show()
