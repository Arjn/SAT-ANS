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

tree = ElementTree.parse('xray_pulsar_lib.xml')
names = []
xml_tree_root = tree.getroot()
lib_objs = Xml2Dict(xml_tree_root, units=True)  # TODO: Add units to dictionary
for key in lib_objs:  # create a numpy array from the xml file
    long_lat = np.array([lib_objs[key]['long_ICRS'][0], lib_objs[key]['lat_ICRS'][0]])
    lib_objs[key]['direction'] = long_lat
    lib_objs[key]['Fx'][0] = lib_objs[key]['Fx'][0]/(8*1.6e-9) #conversion from ergs to photons
    lib_objs[key]['Bx'][0] = lib_objs[key]['Bx'][0]/(8*1.6e-9)
    lib_objs[key]['Fx'][1] = 'J_s-1_cm-2' # change units
    lib_objs[key]['Bx'][1] = 'J_s-1_cm-2'
    c = SkyCoord(long_lat[0] * u.degree, long_lat[1] * u.degree, frame='galactic')
    print(c)
    print(c.icrs)
    print()
    names.append(key)


h = 6.63e-34
hv1 = 2e3*1.6e-19
# eng_conversion
c = 3e8
def sigma_error(int_time, lib_params, detec_area):
    sigma = []
    for pulsar in lib_params:
        d = lib_params[pulsar]['puls_width'][0]/lib_params[pulsar]['P'][0]
        S = lib_params[pulsar]['Fx'][0]*detec_area*lib_params[pulsar]['puls_frac'][0]*int_time
        N = (lib_params[pulsar]['Bx'][0] + lib_params[pulsar]['Fx'][0]*(1-lib_params[pulsar]['puls_frac'][0]))\
            * (detec_area*int_time*d)
        SNR = S/(np.sqrt(N + S))
        sigma.append(lib_params[pulsar]['puls_width'][0]/(2*SNR))
    return sigma

area = 1800 #cm^2
int_time = [1, 10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
error = []


for t in int_time:
    error.append(sigma_error(t, lib_objs, area))

error = np.array(error)
plt.figure(1)
for i in range(0,len(error[0,:])):
    plt.loglog(int_time, error[:,i]*c/1e3, label=names[i])
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Distance error [km]')
plt.title('1800 cm^2 10 best x-ray pulsars distance error')
plt.show()
