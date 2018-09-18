import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from scipy.signal import convolve
import matplotlib.transforms as mtransforms

from scipy.linalg import inv
from poliastro import constants
from astropy import units as u




class Analysis:

    def __init__(self, true_state, filter_state, filter_covariance, time):
        self.true = true_state
        self.filter = filter_state
        self.covar = filter_covariance
        self.time = time
        self.means = []
        self.std = []

        self.font = {'family': 'cursive',
                     'color': 'black',
                     'weight': 'bold',
                     'size': 10,
                     }

    def make_graph_true_compar_pos(self):
        fig1, (ax0, ax1, ax2) = plt.subplots(nrows=3)
        ax0.set_title('Position')
        ax0.plot(self.time, self.filter[:, 0], color='xkcd:medium blue')
        ax0.plot(self.time, self.true[:, 0], '--', color='xkcd:dark green')
        ax0.set_ylabel('x [km]')
        ax0.set_xlabel('time [s]')
        ax0.grid(True)

        ax1.plot(self.time, self.filter[:, 1], color='xkcd:medium blue')
        ax1.plot(self.time, self.true[:, 1], '--', color='xkcd:dark green')
        ax1.set_ylabel('y [km]')
        ax1.set_xlabel('time [s]')
        ax1.grid(True)

        ax2.plot(self.time, self.filter[:, 2], color='xkcd:medium blue')
        ax2.plot(self.time, self.true[:, 2], '--', color='xkcd:dark green')
        ax2.set_ylabel('z [km]')
        ax2.set_xlabel('time [s]')
        ax2.grid(True)

        plt.tight_layout()

    def make_graph_true_compar_vel(self):

        fig1, (ax0, ax1, ax2) = plt.subplots(nrows=3)
        ax0.set_title('Velocity')
        ax0.plot(self.time, self.filter[:, 3], color='xkcd:medium blue')
        ax0.plot(self.time, self.true[:, 3], '--', color='xkcd:dark green')
        ax0.set_ylabel('Vx [km/s]')
        ax0.set_xlabel('time [s]')
        ax0.grid(True)

        ax1.plot(self.time, self.filter[:, 4], color='xkcd:medium blue')
        ax1.plot(self.time, self.true[:, 4], '--', color='xkcd:dark green')
        ax1.set_ylabel('Vy [km/s]')
        ax1.set_xlabel('time [s]')
        ax1.grid(True)

        ax2.plot(self.time, self.filter[:, 5], color='xkcd:medium blue')
        ax2.plot(self.time, self.true[:, 5], '--', color='xkcd:dark green')
        ax2.set_ylabel('Vz [km/s]')
        ax2.set_xlabel('time [s]')
        ax2.grid(True)

        plt.tight_layout()

    def make_graph_pos(self):
        fig1, (ax0, ax1, ax2) = plt.subplots(nrows=3)
        ax0.set_title('Position Error')
        ax0.plot(self.time, self.true[:, 0] - self.filter[:, 0], color='xkcd:medium blue')
        ax0.plot(self.time, np.sqrt(self.covar[:, 0]), color='r')
        ax0.plot(self.time, np.sqrt(self.covar[:, 0]) * -1, color='r')
        ax0.text(0.7, 0.8, 'RMS = %f km\n Std = %f km' % (self.means[0], self.std[0]), fontdict=self.font,
                 transform=ax0.transAxes)
        ax0.set_ylabel('x-error [km]')
        ax0.set_xlabel('time [s]')
        ax0.grid(True)

        ax1.plot(self.time, self.true[:, 1] - self.filter[:, 1], color='xkcd:medium blue')
        ax1.plot(self.time, np.sqrt(self.covar[:, 1]), color='r')
        ax1.plot(self.time, np.sqrt(self.covar[:, 1]) * -1, color='r')
        ax1.text(0.7, 0.8, 'RMS = %f km \n Std = %f km' % (self.means[1], self.std[1]), fontdict=self.font,
                 transform=ax1.transAxes)
        ax1.set_ylabel('y-error [km]')
        ax1.set_xlabel('time [s]')
        ax1.grid(True)

        ax2.plot(self.time, self.true[:, 2] - self.filter[:, 2], color='xkcd:medium blue')
        ax2.plot(self.time, np.sqrt(self.covar[:, 2]), color='r')
        ax2.plot(self.time, np.sqrt(self.covar[:, 2]) * -1, color='r')
        ax2.text(0.7, 0.8, 'RMS = %f km \n Std = %f km' % (self.means[2], self.std[2]), fontdict=self.font,
                 transform=ax2.transAxes)
        ax2.set_ylabel('z-error [km]')
        ax2.set_xlabel('time [s]')
        ax2.grid(True)

        plt.tight_layout()

    def make_graph_vel(self):
        fig2, (ax0, ax1, ax2) = plt.subplots(nrows=3)
        ax0.set_title('Velocity Error')
        ax0.plot(self.time, self.true[:, 3] - self.filter[:, 3], color='xkcd:medium blue')
        ax0.plot(self.time, np.sqrt(self.covar[:, 3]), color='r')
        ax0.plot(self.time, np.sqrt(self.covar[:, 3]) * -1, color='r')
        ax0.text(0.7, 0.8, 'RMS = %f km/s \n Std = %f km/s' % (self.means[3], self.std[3]), fontdict=self.font,
                 transform=ax0.transAxes)
        ax0.set_ylabel('Vx-error [km/s]')
        ax0.set_xlabel('time [s]')
        ax0.grid(True)

        ax1.plot(self.time, self.true[:, 4] - self.filter[:, 4], color='xkcd:medium blue')
        ax1.plot(self.time, np.sqrt(self.covar[:, 4]), color='r')
        ax1.plot(self.time, np.sqrt(self.covar[:, 4]) * -1, color='r')
        ax1.text(0.7, 0.8, 'RMS = %f km/s\n Std = %f km/s' % (self.means[4], self.std[4]), fontdict=self.font,
                 transform=ax1.transAxes)
        ax1.set_ylabel('Vy-error [km/s]')
        ax1.set_xlabel('time [s]')
        ax1.grid(True)

        ax2.plot(self.time, self.true[:, 5] - self.filter[:, 5], color='xkcd:medium blue')
        ax2.plot(self.time, np.sqrt(self.covar[:, 5]), color='r')
        ax2.plot(self.time, np.sqrt(self.covar[:, 5]) * -1, color='r')
        ax2.text(0.7, 0.8, 'RMS = %f km/s\n Std = %f km/s' % (self.means[5], self.std[5]), fontdict=self.font,
                 transform=ax2.transAxes)
        ax2.set_ylabel('Vz-error [km/s]')
        ax2.set_xlabel('time [s]')
        ax2.grid(True)

        plt.tight_layout()

    def find_means(self):
        err = np.array(self.true - self.filter)
        self.means = []
        for i in range(0,6):
            self.means.append(np.mean(np.sqrt(err[:,i] ** 2)))
        print(self.means)

    def find_stds(self):
        err = np.array(self.true - self.filter)
        self.std = []
        for i in range(0,6):
            self.std.append(np.std(err[:,i]))
        print(self.std)

    def Fourier_trans(self, which):
        if which == 'True_pos':
            data = self.true
        elif which == 'Est_pos':
            data = self.filter
        elif which == 'Error':
            data = self.true - self.filter
        else:
            raise NameError('%s not implemented as data set' % which)

        N = len(self.time)
        T = self.time[1] - self.time[0]
        self.state_fft = []
        for i in range(0,6):
            self.state_fft.append(np.array(abs(fft(data[:,i])))[:N//2])
        self.state_fft = np.array(self.state_fft)
        self.freqs = np.linspace(0.0, 1.0/(2.0*T), N//2)
        peaks2 = []
        peaks_store = []
        for i in range(0,6):
            kernel = [1, -1]
            dY = convolve(self.state_fft[i,:], kernel, 'same')
            S = np.sign(dY)
            ddS = convolve(S, kernel, 'valid')
            candidates = np.where(dY > 0)[0]# + (len(kernel) - 1)
            alpha = 0.2
            peaks = sorted(set(candidates).intersection(np.where(ddS == -2)[0]))
            peaks_store.append(np.where(np.array(peaks)[self.state_fft[i, peaks] > max(self.state_fft[i, peaks])*alpha].flatten()))
            peaks2.append(np.array(peaks)[self.state_fft[i, peaks] > max(self.state_fft[i, peaks])*alpha])

        f, axarr = plt.subplots(2, 3)
        f.suptitle('%s'% which)
        labels = ['X', 'Y', 'Z', 'Vx', 'Vy', 'Vz']
        graph = 0
        for i in range(0,2):
            for j in range(0,3):
                ax = axarr[i, j]
                trans_offset = (mtransforms.offset_copy(ax.transData, fig=f, x=0.05, y=0.10))
                ax.loglog(self.freqs, self.state_fft[graph,:])
                for x, y in zip(self.freqs[peaks2[graph]],self.state_fft[graph, peaks2[graph]]):
                    ax.scatter(x, y, marker='x', color='r')
                    ax.text(x,y, '%.5f' % x, transform=trans_offset)
                ax.set_title(labels[graph])
                graph+=1

        # axarr[0, 1].plot(self.freqs, self.state_fft[1,:])
        # for x, y in zip(self.freqs[peaks2[1]],self.state_fft[1, peaks2[1]]):
        #     axarr[0, 1].scatter(x, y, marker='x', color='r')
        #     axarr[0, 1].text(x,y, '%.5f' % x, transform=trans_offset)
        # axarr[0, 1].set_title('Y')
        #
        # axarr[0, 2].plot(self.freqs, self.state_fft[2,:])
        # for x, y in zip(self.freqs[peaks2[2]],self.state_fft[2, peaks2[2]]):
        #     axarr[0, 2].scatter(x, y, marker='x', color='r')
        #     axarr[0, 2].text(x,y, '%.5f' % x, transform=trans_offset)
        # axarr[0, 2].set_title('Z')
        #
        # axarr[1, 0].plot(self.freqs, self.state_fft[3,:])
        # for x, y in zip(self.freqs[peaks2[3]],self.state_fft[3, peaks2[3]]):
        #     axarr[1, 0].scatter(x, y, marker='x', color='r')
        #     axarr[1, 0].text(x,y, '%.5f' % x, transform=trans_offset)
        # axarr[1, 0].set_title('VX')
        #
        # axarr[1, 1].plot(self.freqs, self.state_fft[4,:])
        # for x, y in zip(self.freqs[peaks2[4]],self.state_fft[4, peaks2[4]]):
        #     axarr[1, 1].scatter(x, y, marker='x', color='r')
        #     axarr[1, 1].text(x,y, '%.5f' % x, transform=trans_offset)
        # axarr[1, 1].set_title('VY')
        #
        # axarr[1, 2].plot(self.freqs, self.state_fft[5,:])
        # for x, y in zip(self.freqs[peaks2[5]],self.state_fft[5, peaks2[5]]):
        #     axarr[1, 2].scatter(x, y, marker='x', color='r')
        #     axarr[1, 2].text(x,y, '%.5f' % x, transform=trans_offset)
        # axarr[1, 2].set_title('VZ')

    def make_graphs(self):
        self.find_means()
        self.find_stds()
        self.make_graph_pos()
        self.make_graph_vel()
        self.make_graph_true_compar_pos()
        self.make_graph_true_compar_vel()


