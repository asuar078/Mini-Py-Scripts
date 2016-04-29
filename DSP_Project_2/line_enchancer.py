'''
Line enchancer class
'''

from math import floor
import matplotlib.pyplot as plt
import numpy as np
# import random
from numpy import pi, sin
# from numpy.linalg import inv
from scipy import signal

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


class LineEnchancer:
    def __init__(self, size, frequency):
        self.size = size
        self.frequency = frequency

    def plt_graph(self, sgnl_arr, title='Title', x_axis='x-axis',
                  y_axis='y-axis'):
        '''
        Plot a sgnl
        '''
        plt.style.use('ggplot')

        rang = self.generate_range(len(sgnl_arr))

        fig = plt.figure(figsize=(22, 10))
        graph = fig.add_subplot(1, 1, 1)

        graph.plot(rang, sgnl_arr)
        graph.set_title(title, fontsize=35)
        graph.set_xlabel(x_axis, fontsize=25)
        graph.set_ylabel(y_axis, fontsize=25)
        # plt.show()

    def plt_graph_overlay(self, sgnl1, sgnl2, title='Title',
                          x_axis='x-axis', y_axis='y-axis'):
        '''
        Plot a sgnl
        '''
        plt.style.use('ggplot')

        rang = self.generate_range(len(sgnl1))

        fig = plt.figure(figsize=(22, 10))
        graph = fig.add_subplot(1, 1, 1)

        graph.plot(rang, sgnl1)
        graph.plot(rang, sgnl2, color='b')

        graph.set_title(title, fontsize=35)
        graph.set_xlabel(x_axis, fontsize=25)
        graph.set_ylabel(y_axis, fontsize=25)
        # plt.show()

    def generate_range(self, size):
        '''
        Generate a range of values
        '''
        steps = 1
        # steps = 0.1
        return np.arange(0, size*steps, steps)

    def generate_noise(self, low, high):
        '''
        generates random float values inbetween low high
        values.
        return
            array of random float
        '''
        return np.random.uniform(low, high, self.size)

    def generate_sine_wave(self):
        '''
        generates a sine wave
        return
            array of sine wave values
        '''
        samples = self.generate_range(self.size)

        return sin(2*pi*samples*self.frequency)

    def upper_bound_u(self, sgnl, L):
        '''
        calculate the upper bound of the u value
        sgnl = input sgnl
        L = number of weights
        '''
        mtrx = np.matrix([sgnl])
        pwr = (mtrx * (mtrx.transpose() / mtrx.size))
        print('Power = ' + str(float(pwr)))

        upper = 1 / ((L+1)*float(pwr))
        print('Upper bound u = ' + str(upper))

        return pwr, upper

    def delay(self, sgnl, dly):
        '''
        Shift a sgnl by the value in dly
        return
        shifted list
        '''
        lst = np.zeros(len(sgnl))

        for idx in range(dly, len(sgnl)):
            lst[idx] = sgnl[idx-dly]
        return lst

    def lms(self, delayed_sgnl, sgnl, L, mu):
        '''
        Implementaion of the lms algorithm
        sgnl = input sgnl
        delayed_sgnl = the input sgnl delayed
        L = number of weights
        mu = mu value
        '''
        # convert list to transposed matrix
        delayed_mtrx = np.matrix([delayed_sgnl]).transpose()
        sgnl_mtrx = np.matrix([sgnl]).transpose()

        # create column matrix the size of sgnl
        weights = np.matrix([np.zeros(L)]).transpose()

        error = np.matrix([np.zeros(len(delayed_sgnl))]).transpose()
        output = np.matrix([np.zeros(len(delayed_sgnl))]).transpose()

        length = len(delayed_sgnl)+1

        for indx in range(L+1, length):

            sample_sig = delayed_mtrx[(indx-L):indx]

            output_value = weights.transpose()*sample_sig

            error_value = sgnl_mtrx[indx-1] - output_value

            weights = weights + (2*mu*float(error_value)*sample_sig)

            output[indx-1] = output_value
            error[indx-1] = error_value

        return output, error, weights

    def generate_fft(self, sgnl):
        N = len(sgnl)

        new_lst = np.zeros(len(sgnl))
        for values in range(0, N):
            new_lst[values] = sgnl[values]
        # print(new_lst)

        # sgnl_fft = np.fft.fft(sgnl)
        sgnl_fft = np.fft.fft(new_lst)

        # return 2.0/N * np.abs(sgnl_fft[0:N/2])
        mtrx_fft = 2.0/N * np.abs(sgnl_fft[0:N/2])

        return mtrx_fft

    def generate_psd(self, sgnl):
        N = len(sgnl)

        new_lst = np.zeros(len(sgnl))
        for values in range(0, N):
            new_lst[values] = sgnl[values]

        F, P_den = signal.welch(new_lst, nfft=128, nperseg=128)

        return P_den

    def gated_sine(self, P0=0.1):
        freq = (1/8)
        pulses = [128, 256, 512, 640, 832, 1088, 1216, 1472]
        pulse_length = 64

        # regular sine wave
        samples = self.generate_range(self.size)
        full_wave = P0 * sin(2*pi*samples*freq)
        # sgnl of all zeros
        gated = np.zeros(self.size)

        for pulse in pulses:
            pulse_end = pulse + pulse_length
            gated[pulse:pulse_end] = full_wave[pulse:pulse_end]

        return gated

    def mse(self, sgnl, swdw):
        length = len(sgnl)
        MSE = np.zeros(length)
        doffset = floor(swdw/2)

        for lead in range(swdw, length):
            trail = lead - swdw + 1
            # take the mean of the squares
            thismse = np.mean(list(map(lambda x: x**2, sgnl[trail:lead])))

            drop = trail + doffset
            MSE[drop] = thismse

        return MSE


