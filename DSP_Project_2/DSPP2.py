'''
EEL-6505: Digital Signal Processing
Individual Project 2
'''

import matplotlib.pyplot as plt
from line_enchancer import LineEnchancer

if __name__ == '__main__':

    # x1 = LineEnchancer(size=2048, frequency=(1/16))
    x2 = LineEnchancer(size=2048, frequency=(6/32))
    x1 = LineEnchancer(size=2048, frequency=(6/32))

    x1_sine = x1.generate_sine_wave()
    x1_noise = x1.generate_noise(low=-0.3, high=0.3)
    A = x1_sine + x1_noise

    x2_sine = x2.generate_sine_wave()
    x2_noise = x2.generate_noise(low=-0.3, high=0.3)
    B = x2_sine + x2_noise

    # delay signal by 3
    A_delay = x1.delay(A, 3)

    # run lm
    A_yk, A_ek, A_w = x1.lms(A_delay, A, 50, 0.001)

    # generate fft for signanl, output and error
    sine_fft = x1.generate_fft(x1_sine)
    A_fft = x1.generate_fft(A)
    A_yk_fft = x1.generate_fft(A_yk)
    A_ek_fft = x1.generate_fft(A_ek)

    # x1.plt_graph(A, title='A')
    # x1.plt_graph(A_yk, title='A yk')
    # x1.plt_graph(A_ek, title='A ek')

    # x1.plt_graph_overlay(x1_sine, A_yk)

    x1.plt_graph(sine_fft, title='sine fft')
    x1.plt_graph(A_fft, title='A fft')
    x1.plt_graph(A_yk_fft, title='A yk fft')
    # x1.plt_graph(A_ek_fft, title='A ek fft')

    # part 2
    gated = x1.gated_sine()
    # x1.plt_graph(gated)

    # weak_carrier = gated + x2_noise/10
    weak_carrier = gated + x2_noise/10 + x2_sine
    # x2.plt_graph(weak_carrier)

    # mse calculation
    MSE = x2.mse(weak_carrier, 20)
    # x2.plt_graph(MSE)

    plt.show()
