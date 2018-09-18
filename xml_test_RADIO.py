from astropy.constants.codata2014 import c
import numpy as np
from astropy import units as u
from numpy.random import randn
import xml.etree.cElementTree as ElementTree
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord


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

tree = ElementTree.parse('radio_pulsar_lib.xml')
names = []
xml_tree_root = tree.getroot()
lib_objs = Xml2Dict(xml_tree_root, units=True)  # TODO: Add units to dictionary
for key in lib_objs:  # create a numpy array from the xml file
    long_lat = np.array([lib_objs[key]['long_ICRS'][0], lib_objs[key]['lat_ICRS'][0]])
    lib_objs[key]['direction'] = long_lat
    # c = SkyCoord(long_lat[0] * u.degree, long_lat[1] * u.degree, frame='galactic')
    # print(c)
    # print(c.icrs)
    # print()
    names.append(key)

alpha = 0.5
A = 100
v_rec = 1.4
B = 400e6
v_ref = 1.4
beta = -1.8
d = 1
atten = -40
k_b = 1.3806e-23
T_rec = 15

def sigma_error(int_time, lib_params):
    sigma = []
    for pulsar in lib_params:
        S = alpha*A*1e-26*lib_params[pulsar]['Sp'][0]*1e-3*(v_rec/v_ref)**beta
        N = k_b*(T_rec + 2.7 + 6*v_rec**(-2.2) + (72*v_rec + 0.058)*A*10**(atten/10)*d**(-2))
        SNR = S/N
        print(10*np.log10(SNR))
        err = 1/((2*np.pi)**2*SNR**2*lib_params[pulsar]['Q'][0]*B*int_time)
        sigma.append(np.sqrt(err))
    return sigma


int_time = [1, 10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
error = []


for t in int_time:
    error.append(sigma_error(t, lib_objs))

error = np.array(error)
plt.figure(1)
for i in range(0,len(error[0,:])):
    plt.loglog(int_time, error[:,i]*3e8/1e3, label=names[i])
plt.legend()
plt.xlabel('Integration Time [s]')
plt.ylabel('Distance error [km]')
plt.title('100 m^2 10 best radio pulsars distance error')
plt.show()
