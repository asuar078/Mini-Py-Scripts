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
    # plt.show()

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

    return pwr, upper

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

def lms(delayed_signal, signal, num_of_weights, mu):
    # convert list to transposed matrix
    delayed_mtrx = np.matrix([delayed_signal]).transpose()
    signal_mtrx = np.matrix([signal]).transpose()
    # print(delayed_mtrx)
    # print(signal_mtrx)

    # create column matrix the size of signal
    weights = np.matrix([np.zeros(num_of_weights)]).transpose()
    # print(weights)

    error = np.matrix([np.zeros(len(delayed_signal))]).transpose()
    output = np.matrix([np.zeros(len(delayed_signal))]).transpose()

    length = len(delayed_signal)+1
    # print('length: ' + str(length))

    for indx in range(num_of_weights+1, length):
        #print('index: ' + str(indx))

        sample_sig = delayed_mtrx[(indx-num_of_weights):indx]
        #print(sample_sig)

        output_value = weights.transpose()*sample_sig
        #print('output: ' + str(output_value))

        # output[indx-1] = output_value
        error_value = signal[indx-1] - output_value
        #print('error: ' + str(error_value))

        weights = weights + (2*mu*float(error_value)*sample_sig)
        #print('new weights: ' + str(weights))
        #print()

        output[indx-1] = output_value
        error[indx-1] = error_value


    return output, error, weights

if __name__ == '__main__':
    SIZE = 2048
    FREQUENCY = (1.0/16)

    noise = generate_noise(low=-0.3, high=0.3, size=SIZE)
    sine_wave = generate_sine_wave(SIZE, FREQUENCY)
    noisy_wave = noise + sine_wave
    # noisy_wave = [values for values in range(0,10)]

    # upper_bound_u(noisy_wave, 50)

    delayed = delay(noisy_wave, 3)

    out, err, w = lms(delayed, noisy_wave, 60, 0.001)

    # test_lst = [values for values in range(5)]
    # # print(test_lst)
    # mtrx = np.matrix([test_lst]).transpose()

    plt_graph(out, title='After LMS')
    # plt_graph(err, title='error')

    plt_graph(noisy_wave, title='Before LMS')
    # plt_graph(delayed)


    plt.show()
