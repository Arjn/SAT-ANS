import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd

P = pd.read_csv('MSE_P.csv', delim_whitespace=True)
V = pd.read_csv('MSE_V.csv', delim_whitespace=True)

CRLB_P = []
CRLB_V = []
plt.figure(1)
for i in range(0,len(P.values[1,:])):
    CRLB_P.append(np.mean(P.values[:,i]))
plt.semilogy(CRLB_P)
print(np.mean(CRLB_P))

np.savetxt('CRLB_P.csv', np.array(CRLB_P),  delimiter=',')

plt.figure(2)
for i in range(0,len(P.values[1,:])):
    CRLB_V.append(np.mean(V.values[:,i]))
plt.semilogy(CRLB_V)
print(np.mean(CRLB_V))
np.savetxt('CRLB_V.csv', np.array(CRLB_V), delimiter=',')

plt.show()