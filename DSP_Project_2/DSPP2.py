'''
EEL-6505: Digital Signal Processing
Individual Project 2
'''

import matplotlib.pyplot as plt
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

    # A3_psd = True
    A3_psd = False

    B1 = True
    # B1 = False

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

    # A.2
    x1_noise = x1.generate_noise(low=-0.3, high=0.3)
    A = x1_sine + x1_noise

    x2_noise = x2.generate_noise(low=-0.3, high=0.3)
    B = x2_sine + x2_noise

    x3_noise = x3.generate_noise(low=-0.3, high=0.3)
    C = x3_sine + x3_noise

    if A1:
        x1.plt_graph(x1_sine[:200], title='x1(k) f1 = fs/16',
                     y_axis='Amplitude', x_axis='k')
        x2.plt_graph(x2_sine[:200], title='x2(k) f2 = fs * (6/32)',
                     y_axis='Amplitude', x_axis='k')
        x3.plt_graph(x3_sine[:200], title='x3(k) f3 = fs/4',
                     y_axis='Amplitude', x_axis='k')

        x1.plt_graph(x1_noise[:200], title='r(k)',
                     y_axis='Amplitude', x_axis='k')

    if A2:
        x1.plt_graph(A[1792:], title='r(k) + x1(k)', x_axis='Amplitude',
                     y_axis='k')
        x2.plt_graph(B[1792:], title='r(k) + x2(k)', x_axis='Amplitude',
                     y_axis='k')
        x3.plt_graph(C[1792:], title='r(k) + x3(k)', x_axis='Amplitude',
                     y_axis='k')

    # number of taps
    A_L = 60
    B_L = 50
    C_L = 70

    Apwr, Amu = x1.upper_bound_u(A, A_L)
    print('A for L={0}, max mu={1:0.4f}'.format(A_L, Amu))

    Bpwr, Bmu = x2.upper_bound_u(B, B_L)
    print('B for L={0}, max mu={1:0.4f}'.format(B_L, Bmu))

    Cpwr, Cmu = x3.upper_bound_u(C, C_L)
    print('C for L={0}, max mu={1:0.4f}'.format(C_L, Cmu))

    # delay values
    delay_A = 3
    delay_B = 3
    delay_C = 3

    # delay signal
    A_delay = x1.delay(A, delay_A)
    B_delay = x2.delay(B, delay_B)
    C_delay = x3.delay(C, delay_C)

    # mu value
    A_mu = 0.001
    B_mu = 0.002
    C_mu = 0.001

    # run lms
    A_yk, A_ek, A_w = x1.lms(A_delay, A, A_L, A_mu)
    B_yk, B_ek, B_w = x2.lms(B_delay, B, B_L, B_mu)
    C_yk, C_ek, C_w = x3.lms(C_delay, C, C_L, C_mu)

    # A.3
    if A3:
        # x1
        x1.plt_3_graph(A[1792:], A_yk[1792:], A_ek[1792:], title1='Primary Input',
                       title3='Error', title2='Filter Output',
                       fTitle='r(k) + x1(k)', delay=delay_A, L=A_L,
                       mu=A_mu)

        x1.plt_graph_overlay(x1_sine[1792:], A_yk[1792:], title='X1 and Filter Output',
                             x_axis='K', y_axis='Amplitude')

        # # x2
        x2.plt_3_graph(B[1792:], B_yk[1792:], B_ek[1792:], title1='Primary Input',
                       title3='Error', title2='Filter Output',
                       fTitle='r(k) + x2(k)', delay=delay_B, L=B_L,
                       mu=B_mu)

        x2.plt_graph_overlay(x2_sine[1792:], B_yk[1792:], title='X2 and Filter Output',
                             x_axis='K', y_axis='Amplitude')

        # # x3
        x3.plt_3_graph(C[1792:], C_yk[1792:], C_ek[1792:], title1='Primary Input',
                       title3='Error', title2='Filter Output',
                       fTitle='r(k) + x3(k)', delay=delay_C, L=C_L,
                       mu=C_mu)

        x3.plt_graph_overlay(x3_sine[1792:], C_yk[1792:], title='X3 and Filter Output',
                             x_axis='K', y_axis='Amplitude')

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

    if A3_fft:
        x1.plt_3_graph(A_fft, A_yk_fft, A_ek_fft,
                       title1='Primary Input fft',
                       title3='Error fft', title2='Filter Output fft',
                       fTitle='r(k) + x1(k) fft', leg=False)

        x2.plt_3_graph(B_fft, B_yk_fft, B_ek_fft,
                       title1='Primary Input fft',
                       title3='Error fft', title2='Filter Output fft',
                       fTitle='r(k) + x2(k) fft', leg=False)

        x3.plt_3_graph(C_fft, C_yk_fft, C_ek_fft,
                       title1='Primary Input fft',
                       title3='Error fft', title2='Filter Output fft',
                       fTitle='r(k) + x3(k) fft', leg=False)

    # Power spectral density
    # A
    # Faa, Paa_den = signal.welch(x1_sine)
    A_psd = x1.generate_psd(A)
    A_yk_psd = x1.generate_psd(A_ek)
    A_ek_psd = x1.generate_psd(A_yk)

    # B
    B_psd = x2.generate_psd(B)
    B_yk_psd = x2.generate_psd(B_ek)
    B_ek_psd = x2.generate_psd(B_yk)

    # C
    C_psd = x3.generate_psd(C)
    C_yk_psd = x3.generate_psd(C_ek)
    C_ek_psd = x3.generate_psd(C_yk)

    if A3_psd:
        x1.plt_3_graph(A_psd, A_yk_psd, A_ek_psd,
                       title1='Primary Input psd',
                       title3='Error psd', title2='Filter Output psd',
                       fTitle='r(k) + x1(k) psd', leg=False)

        x2.plt_3_graph(B_psd, B_yk_psd, B_ek_psd,
                       title1='Primary Input psd',
                       title3='Error psd', title2='Filter Output psd',
                       fTitle='r(k) + x2(k) psd', leg=False)

        x3.plt_3_graph(C_psd, C_yk_psd, C_ek_psd,
                       title1='Primary Input psd',
                       title3='Error psd', title2='Filter Output psd',
                       fTitle='r(k) + x3(k) psd', leg=False)

    ################################################
    # PART B)
    ################################################
    L = 60
    mu = 0.008
    P0 = 0.08

    gated = x1.gated_sine(P0)
    x1.plt_graph(gated, title='gated')

    P = gated + x2_noise/10 + x2_sine
    x2.plt_graph(P, title='P')
    # print(x2.upper_bound_u(P, L))

    # delay signal by 3
    P_delay = x2.delay(P, 3)

    # run lms
    P_yk, P_ek, P_w = x2.lms(P_delay, P, L, mu)

    x2.plt_graph(P_yk, title='P yk')

    # mse calculation
    MSE = x2.mse(P_ek, 20)

    if B1:
        x2.plt_graph_overlay(gated, MSE,
                             title='Detection of Weak Carrier Pulse',
                             y_axis='Amplitude',
                             x_axis='k', P0=P0, L=L, mu=mu)

    plt.show()
