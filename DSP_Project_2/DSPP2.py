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

def generate_noise(low, high, size):
    return np.random.uniform(low, high, size)

def random_noise_plt(low, high, size):
    x1 = np.arange(0, size, 1)
    ran = np.random.uniform(low, high, size)

    fig = plt.figure(figsize=(22, 10))
    noise = fig.add_subplot(1, 1, 1)

    noise.plot(x1, ran)

    plt.show()

def sine_wave(size, frequency, const):
    sampling_freq = 6*frequency

    samples = np.arange(0, size, sampling_freq)

    return sin(2*pi*samples*frequency*const)

def plot_sine_wave(size, frequency, const):
    sampling_freq = 6*frequency

    samples = np.arange(0, size, 0.01)

#    wave = sin(2*pi*samples*frequency*const)
    wave = sin(2*pi*samples)

    fig = plt.figure(figsize=(22, 10))
    sine = fig.add_subplot(1, 1, 1)

    sine.plot(samples, wave)

    plt.show()

#random_noise_plt(-3, 3, 200)
plot_sine_wave(200, 1000, 1)

#fs = 44100
#t = np.arange(0, 1, 1.0/fs)
#f0 = 100
#phi = np.pi/2
#A = .8
#x = A * np.sin(2 * np.pi * f0 * t + phi)
#
#plt.plot(t, x)
##plt.axis([0, 0.01, -.8, .8])
#plt.xlabel('time')
#plt.ylabel('amplitude')
#plt.show()
