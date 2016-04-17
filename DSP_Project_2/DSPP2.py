'''
EEL-6505: Digital Signal Processing
Individual Project 2
'''

import matplotlib.pyplot as plt
import numpy as np
import random
from numpy import pi, sin, cos
from numpy import linalg as LA
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

def plt_graph(signal_arr, title='Title', x_axis='x-axis', y_axis='y-axis'):
    '''
    Plot a signal
    '''
    plt.style.use('ggplot')

    rang = generate_range(len(signal_arr))

    fig = plt.figure(figsize=(22, 10))
    graph = fig.add_subplot(1, 1, 1)

    graph.plot(rang, signal_arr)
    graph.set_title(title, fontsize=35)
    graph.set_xlabel(x_axis, fontsize=25)
    graph.set_ylabel(y_axis, fontsize=25)
    plt.show()

def generate_range(size):
    return np.arange(0, size*0.1, 0.1)

def generate_noise(low, high, size):
    '''
    generates random float values inbetween low high
    values.
    return
        array of random float
    '''
    return np.random.uniform(low, high, size)

def generate_sine_wave(size, frequency):
    '''
    generates a sine wave
    return
        array of sine wave values
    '''
#    samples = np.arange(0, size, 0.01)
    samples = generate_range(size)

    return sin(2*pi*samples*frequency)

def upper_bound_u(noisy, L):
    mtrx = np.matrix([noisy])
    pwr = (mtrx * (mtrx.transpose() / mtrx.size))
    print('Power = '+ str(float(pwr)))

    upper = 1 / ((L+1)*float(pwr))
    print('Upper bound u = ' + str(upper))

def delay(signal, dly):
    '''
    Shift a signal by the value in dly
    return
    shifted list
    '''
    lst = np.zeros(len(signal))

    for idx in range(dly, len(signal)):
        lst[idx] = signal[idx-dly]
    return lst

if __name__ == '__main__':
    SIZE = 200
    FREQUENCY = (1.0/16)

    noise = generate_noise(low=-0.3, high=0.3, size=SIZE)
    sine_wave = generate_sine_wave(SIZE, FREQUENCY)
    noisy_wave = noise + sine_wave

    upper_bound_u(noisy_wave, 50)

    delayed = delay(noisy_wave, 40)

    plt_graph(noisy_wave)
    plt_graph(delayed)




