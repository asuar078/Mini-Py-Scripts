'''
EEL-6505: Digital Signal Processing
Individual Project 2
'''

import matplotlib.pyplot as plt
from scipy import signal
from line_enchancer import LineEnchancer

if __name__ == '__main__':
    # A1 = True
    A1 = False

    # A2 = True
    A2 = False

    # A3 = True
    A3 = False

    # A3_fft = True
    A3_fft = False

    A3_psd = True
    # A3_psd = False

    # B1 = True
    B1 = False

    ################################################
    # PART A)
    ################################################
    # A.1
    x1 = LineEnchancer(size=2048, frequency=(1/16))
    x2 = LineEnchancer(size=2048, frequency=(6/32))
    x3 = LineEnchancer(size=2048, frequency=(1/4))

    x1_sine = x1.generate_sine_wave()
    x2_sine = x2.generate_sine_wave()
    x3_sine = x3.generate_sine_wave()

    if A1:
        x1.plt_graph(x1_sine[:200], title='x1')
        x2.plt_graph(x2_sine[:200], title='x2')
        x3.plt_graph(x3_sine[:200], title='x3')

    # A.2
    x1_noise = x1.generate_noise(low=-0.3, high=0.3)
    A = x1_sine + x1_noise

    x2_noise = x2.generate_noise(low=-0.3, high=0.3)
    B = x2_sine + x2_noise

    x3_noise = x3.generate_noise(low=-0.3, high=0.3)
    C = x3_sine + x3_noise

    if A2:
        x1.plt_graph(A, title='A')
        x2.plt_graph(B, title='B')
        x3.plt_graph(C, title='C')

    # delay signal
    A_delay = x1.delay(A, 3)
    B_delay = x2.delay(B, 3)
    C_delay = x3.delay(C, 3)

    # run lms
    A_yk, A_ek, A_w = x1.lms(A_delay, A, 50, 0.001)
    B_yk, B_ek, B_w = x2.lms(B_delay, B, 50, 0.001)
    C_yk, C_ek, C_w = x3.lms(C_delay, C, 50, 0.001)

    # A.3
    if A3:
        # x1.plt_graph(A, title='A')
        # x1.plt_graph(A_yk[1024:], title='A yk')
        # x1.plt_graph(A_ek, title='A ek')

        # x2.plt_graph(B, title='B')
        # x2.plt_graph(B_yk[1024:], title='B yk')
        # x2.plt_graph(B_ek, title='B ek')

        x3.plt_graph(C, title='C')
        # x3.plt_graph(C_yk[1024:], title='C yk')
        x3.plt_graph(C_yk, title='C yk')
        x3.plt_graph(C_ek, title='C ek')

    # generate fft for signanl, output and error
    # A
    A_sine_fft = x1.generate_fft(x1_sine)
    A_fft = x1.generate_fft(A)
    A_yk_fft = x1.generate_fft(A_yk)
    A_ek_fft = x1.generate_fft(A_ek)

    # B
    B_sine_fft = x2.generate_fft(x2_sine)
    B_fft = x2.generate_fft(B)
    B_yk_fft = x2.generate_fft(B_yk)
    B_ek_fft = x2.generate_fft(B_ek)

    # C
    C_sine_fft = x3.generate_fft(x3_sine)
    C_fft = x3.generate_fft(C)
    C_yk_fft = x3.generate_fft(C_yk)
    C_ek_fft = x3.generate_fft(C_ek)

    # print('C wave')
    # print(C_fft)
    # print(type(C_fft))
    # print(len(C_fft))

    # print('C fft')
    # print(C_yk_fft)
    # print(type(C_yk_fft))
    # print(len(C_yk_fft))

    # x1.plt_graph_overlay(x1_sine, A_yk)

    if A3_fft:
        # x1.plt_graph(A_sine_fft, title='A sine fft')
        # x1.plt_graph(A_fft, title='A fft')
        # x1.plt_graph(A_yk_fft, title='A yk fft')
        # x1.plt_graph(A_ek_fft, title='A ek fft')
        # x1.plt_graph_overlay(x1_sine, A_yk)

        # x2.plt_graph(B_sine_fft, title='B sine fft')
        # x2.plt_graph(B_fft, title='B fft')
        # x2.plt_graph(B_yk_fft, title='B yk fft')
        # x2.plt_graph(B_ek_fft, title='B ek fft')

        x3.plt_graph(C_sine_fft, title='C sine fft')
        x3.plt_graph(C_fft, title='C fft')
        x3.plt_graph(C_yk_fft, title='C yk fft')
        x3.plt_graph(C_ek_fft, title='C ek fft')

    # Power spectral density
    # A
    # Faa, Paa_den = signal.welch(x1_sine)
    Paa_den = x1.generate_psd(A)
    Pab_den = x1.generate_psd(A_ek)
    Pac_den = x1.generate_psd(A_yk)

    # B
    Pba_den = x2.generate_psd(B)
    Pbb_den = x2.generate_psd(B_ek)
    Pbc_den = x2.generate_psd(B_yk)

    # C
    Pca_den = x3.generate_psd(C)
    Pcb_den = x3.generate_psd(C_ek)
    Pcc_den = x3.generate_psd(C_yk)

    if A3_psd:
        x1.plt_graph(Paa_den, title='A PSD')
        x1.plt_graph(Pab_den, title='A_ek PSD')
        x1.plt_graph(Pac_den, title='A_yk PSD')
        # plt.semilogy(Fac, Pac_den)

    ################################################
    # PART B)
    ################################################
    L = 40
    mu = 0.005
    P0 = 0.1

    gated = x1.gated_sine(P0)
    # x1.plt_graph(gated, title='gated')

    # P = gated + x2_noise/10
    P = gated + x2_noise/10 + x2_sine
    # x2.plt_graph(P, title='P')
    # print(x2.upper_bound_u(P, L))

    # delay signal by 3
    P_delay = x2.delay(P, 3)

    # run lms
    P_yk, P_ek, P_w = x2.lms(P_delay, P, L, mu)

    # x2.plt_graph(P_yk, title='P yk')

    # mse calculation
    MSE = x2.mse(P_ek, 20)

    if B1:
        x2.plt_graph_overlay(gated, MSE, title='carrier')

    plt.show()
