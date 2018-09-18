import csv
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Create a function
# ==> First encounter with *whitespace* in Python <==

def make_float(string_list):
    return [float(i) for i in string_list]

# Generate fake data.
# Note: functions in random package, array arithmetic (exp)


# e = random.uniform(0.1, 1., n)
# Note: these error bars don't reflect the distribution from which
# they were drawn! Chi^2 of the fit will be poor.
pulsar_params = []

#1B937+21 first

with open('b1937_orig_1414.csv', 'r') as csvfile:
    int = csv.reader(csvfile, delimiter=',')
    for row in int:
        if float(row[3]) < 0: row[3] = 0
        pulsar_params.append(make_float(row))
pulsar_params = np.array(pulsar_params)

a = 2*np.pi/len(pulsar_params[:,2])
pulsar_period = np.multiply(pulsar_params[:,2],a)
normalised_profile = pulsar_params[:,3]/max(pulsar_params[:,3])

normalised_profile = np.array(normalised_profile)


data = normalised_profile

# plt.plot(pulsar_period, data, '.')

X = np.arange(0,max(pulsar_period),max(pulsar_period)/len(pulsar_period))
x = np.sum(X*data)/np.sum(data)
width = np.sqrt(np.abs(np.sum((X-x)**2*data)/np.sum(data)))

maximum = data.max()

fit = lambda t : maximum*np.exp(-(t-x)**2/(2*width**2))
X = np.arange(0, max(pulsar_period),0.00001)
# plt.plot(X, fit(X))



def gaussian(x, height, center, width):
    return height*np.exp(-(x - center)**2/(2*width**2))


def four_gaussian(x, height1, center1, width1,
                  height2, center2, width2,
                  height3, center3, width3,
                  height4, center4, width4):
    out = (gaussian(x, height1, center1, width1) +
             gaussian(x, height2, center2, width2) +
             gaussian(x, height3, center3, width3) +
             gaussian(x, height4, center4, width4))

    return out


def n_gaussian(x, *params):
    temp = 0
    for i in range(0,len(params),3):
        temp += gaussian(x, params[i], params[i+1], params[i+2])
    return temp

errfunc3 = lambda p, x, y: (n_gaussian(x, * p) - y) ** 2


gaus1 = gaussian(pulsar_period, 0.47, 1.551756371318595, 0.0761598219052071)
gaus2 = gaussian(pulsar_period, 0.05, 1.780235837034216, 0.024751942119192304)
gaus3 = gaussian(pulsar_period, 1, 4.5600693365742755, 0.0666398441670562)
gaus4 = gaussian(pulsar_period, 0.34, 4.702869002646539, 0.02284794657156213)

guess4 = [0.47, 0.1/0.33 * 2*np.pi, 0.005/0.33 * 2*np.pi, 0.05, 0.09/0.33 * 2*np.pi, 0.001/0.33 * 2*np.pi, 1,
          0.24/0.33 * 2*np.pi, 0.004/0.33 * 2*np.pi, 0.34, 0.25/0.33 * 2*np.pi, 0.001/0.33 * 2*np.pi]

optim3, success = optimize.leastsq(errfunc3, guess4[:], args=(pulsar_period, normalised_profile))


plt.plot(pulsar_period, normalised_profile, lw=4, c='b', label='Least Squares Data')
# plt.plot(pulsar_period, n_gaussian(pulsar_period, *optim3), lw=3, c='r', label='4 Gaussian fit')

# plt.plot(pulsar_period,four_gaussian(pulsar_period, *optim3), color='r')
plt.legend()
plt.xlabel('time [s]')
plt.ylabel('normalised flux')
plt.title('B1973+21 orginal (taken at 1414 MHz) and fitted profile')
plt.savefig('result.png')
plt.show()