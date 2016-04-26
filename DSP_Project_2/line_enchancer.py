'''
Line enchancer class
'''

from math import floor
import matplotlib.pyplot as plt
import numpy as np
import random
from numpy import pi, sin, cos
from numpy import linalg as LA
from numpy.linalg import inv
from scipy.fftpack import fft
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


class LineEnchancer:
    def __init__(self, size, frequency):
        self.size = size
        self.frequency = frequency

    def plt_graph(self, signal_arr, title='Title', x_axis='x-axis',
                  y_axis='y-axis'):
        '''
        Plot a signal
        '''
        plt.style.use('ggplot')

        rang = self.generate_range(len(signal_arr))

        fig = plt.figure(figsize=(22, 10))
        graph = fig.add_subplot(1, 1, 1)

        graph.plot(rang, signal_arr)
        graph.set_title(title, fontsize=35)
        graph.set_xlabel(x_axis, fontsize=25)
        graph.set_ylabel(y_axis, fontsize=25)
        # plt.show()

    def plt_graph_overlay(self, signal1, signal2, title='Title',
                          x_axis='x-axis', y_axis='y-axis'):
        '''
        Plot a signal
        '''
        plt.style.use('ggplot')

        rang = self.generate_range(len(signal1))

        fig = plt.figure(figsize=(22, 10))
        graph = fig.add_subplot(1, 1, 1)

        graph.plot(rang, signal1)
        graph.plot(rang, signal2, color='b')

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

    def upper_bound_u(self, signal, L):
        '''
        calculate the upper bound of the u value
        signal = input signal
        L = number of weights
        '''
        mtrx = np.matrix([signal])
        pwr = (mtrx * (mtrx.transpose() / mtrx.size))
        print('Power = ' + str(float(pwr)))

        upper = 1 / ((L+1)*float(pwr))
        print('Upper bound u = ' + str(upper))

        return pwr, upper

    def delay(self, signal, dly):
        '''
        Shift a signal by the value in dly
        return
        shifted list
        '''
        lst = np.zeros(len(signal))

        for idx in range(dly, len(signal)):
            lst[idx] = signal[idx-dly]
        return lst

    def lms(self, delayed_signal, signal, L, mu):
        '''
        Implementaion of the lms algorithm
        signal = input signal
        delayed_signal = the input signal delayed
        L = number of weights
        mu = mu value
        '''
        # convert list to transposed matrix
        delayed_mtrx = np.matrix([delayed_signal]).transpose()
        signal_mtrx = np.matrix([signal]).transpose()
        # print(delayed_mtrx)
        # print(signal_mtrx)

        # create column matrix the size of signal
        weights = np.matrix([np.zeros(L)]).transpose()
        # print(weights)

        error = np.matrix([np.zeros(len(delayed_signal))]).transpose()
        output = np.matrix([np.zeros(len(delayed_signal))]).transpose()

        length = len(delayed_signal)+1
        # print('length: ' + str(length))

        for indx in range(L+1, length):
            #print('index: ' + str(indx))

            sample_sig = delayed_mtrx[(indx-L):indx]
            #print(sample_sig)

            output_value = weights.transpose()*sample_sig
            #print('output: ' + str(output_value))

            # output[indx-1] = output_value
            # error_value = signal[indx-1] - output_value
            error_value = signal_mtrx[indx-1] - output_value
            #print('error: ' + str(error_value))

            weights = weights + (2*mu*float(error_value)*sample_sig)
            #print('new weights: ' + str(weights))
            #print()

            output[indx-1] = output_value
            error[indx-1] = error_value

        return output, error, weights

    def generate_fft(self, signal):
        # signal_fft = fft(signal)
        signal_fft = np.fft.rfft(signal, norm='ortho')

        plt.plot(signal_fft)
        plt.show()

        # P2 = np.abs(signal_fft/self.size)
        # P1 = P2[0:self.size/2]

        # return np.abs(signal_fft[0:self.size/2])
        # return signal_fft[0:self.size/2]
        # return P1
        return np.fft.rfft(signal)

    def gated_sine(self, P0=0.1):
        freq = (1/8)
        pulses = [128, 256, 512, 640, 832, 1088, 1216, 1472]
        pulse_length = 64

        # regular sine wave
        samples = self.generate_range(self.size)
        full_wave = P0 * sin(2*pi*samples*freq)
        # signal of all zeros
        gated = np.zeros(self.size)

        for pulse in pulses:
            pulse_end = pulse + pulse_length
            gated[pulse:pulse_end] = full_wave[pulse:pulse_end]

        return gated

    def mse(self, signal, swdw):
        length = len(signal)
        MSE = np.zeros(length)
        doffset = floor(swdw/2)

        for lead in range(swdw, length):
            trail = lead - swdw + 1
            # take the mean of the squares
            thismse = np.mean(list(map(lambda x: x**2, signal[trail:lead])))

            drop = trail + doffset
            MSE[drop] = thismse

        return MSE


