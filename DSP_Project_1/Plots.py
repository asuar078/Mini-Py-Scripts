"""
Examples illustrating the use of plt.subplots().

This function creates a figure and a grid of subplots with a single call, while
providing reasonable control over how the individual plots are created.  For
very refined tuning of subplot creation, you can still use add_subplot()
directly on a new figure.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, sin, cos
from numpy import linalg as LA
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

print("   ****** Digital Signal Processing Project 1 ******")
plt.style.use('ggplot')

def plotSineWaves (N=10, START=1, STOP=200, DELTA=0.01):
    '''
    Part 1: Plot the signal that you have simulated
    '''
    # create an array of values from 1 to 200
    x1 = np.arange(START, STOP, DELTA)
    y1 = sin(2*pi*x1/N)

    # create an array of values from 1 to 200
    dx1 = np.arange(START, STOP, DELTA)
    dy1 = 2 * cos(2*pi*x1/N)

    plt.close('all')

    fig = plt.figure(figsize=(22, 10))
    xk1 = fig.add_subplot(2, 1, 1)
    dk1 = fig.add_subplot(2, 1, 2)

    xk1.plot(x1, y1)
    xk1.set_title('Input Signal, (xk)=sin(2*pi*k/'+ str(N) + ')')

    dk1.plot(dx1, dy1)
    dk1.set_title('Desired Signal, (dk)=2*cos(2*pi*k/' + str(N) + ')')

    plt.xlabel('Time Series ' + str(START) + ' <= k <= ' + str(STOP) , fontsize=20)
    plt.show()

def findR (N=10):
    '''
    Create R matrix
    '''
    # R matrix
    r11 = 0.5
    r12 = 0.5 * cos(2*pi/N)
    r21 = 0.5 * cos(2*pi/N)
    r22 = 0.5
    R = np.matrix([[r11,r12], [r21,r22]])
    print('\n-- R --')
    print(R)

    return R

def findP (N=10):
    '''
    Create P matrix
    '''
    # P matrix
    p11 = 0
    p21 = -sin(2*pi/N)
    P = np.matrix([[p11], [p21]])
    print('\n-- P --')
    print(P)

    return P

def find_optimal_w (R, P):
    '''
    Calculate the optimal weight matrix
    using the weiner solution, requires
    R and P matrix
    '''
    # optimal weights
    w_optimal = inv(R) * P
    print('\n-- w* --')
    print(w_optimal)

    return w_optimal

def min_sqr_error (P, w_optimal, edk2=2):
    '''
    Calculate minimum square error
    based of the matrix P, optimal weights and
    expected error square
    '''
    # minimum mean-square error
    e_min = edk2 - P.transpose()*w_optimal
    print('\n-- minimum mean-square error --')
    print(e_min)

    return e_min

def eigenValues (R):
    '''
    Calculate eigenvalues and normalized eigenvectors
    for the matrix given
    '''
    # Eigenvalues and Eigenvectors
    w, v = LA.eig(R)
    print('\n-- Eigenvalues lamda 1 and lamda 2 --')
    print(w)
    print('-- Normalized Eigenvector --')
    print(v)

    return w, v

def mean_square_error (W, ALPHA=0.5, BETA=0.5, GAMMA=0.81, DELTA=1.176,
                        EPSILON=0, ZETA=2):
    '''
    Calculate mean square error for a two weight system
    equation is broken down into variables
    '''
    W0 = W.item(0,0)
    W1 = W.item(1,0)
    MSE = ALPHA*W0*W0 + BETA*W1*W1 + GAMMA*W0*W1 + DELTA*W1 + EPSILON*W0 + ZETA

    return MSE

def meanSquareError(edk2, W, R, P):
    '''
    Calculate mean square error for a two weight system
    needs W, R, and P matrix
    '''
    MSE = edk2 + W.transpose()*R*W - 2*P.transpose()*W

    return float(MSE)

def gradient(R, P, W):
    '''
    Calculate gradient of performance surface
    Requires R,P and Weight matrix
    '''
    grad = 2*R*W - 2*P
    print('\n-- Gradient --')
    print(grad)

    return grad

def newton_method_next_weight(w_optimal, w_current, mu):
    '''
    Calculate next weight using newtons method
    '''
    w_next = (1-2*mu)*w_current + 2*mu*w_optimal

    return w_next

def newton_method_learning_curve(w_optimal, edk2, w_start, R, P, mu, min_err=0, start=0,
    iteration=15, graph=False, concat=False, w_optimal2=None, edk22=None, R2=None, P2=None,
    iteration2=15):
    '''
    Plot the learning curve and weight track for an interval using the
    newtons method
    graph: if true will display learning curve and weight track plot
    concat: can concatinate two intervals, need to supply second parameters

    '''
    # create empty list
    # to be filled through each iteration
    count = []
    MSE = []

    w0 = []
    w1 = []

    w0_opt = []
    w1_opt = []
    min_error = []

    new_weight = w_start

    end=start+iteration

    for indx in range(start, end):
        mse = meanSquareError(edk2, new_weight, R, P)

        w0.append(new_weight.item(0,0))
        w1.append(new_weight.item(1,0))

        count.append(indx)
        MSE.append(mse)

        w0_opt.append(w_optimal.item(0,0))
        w1_opt.append(w_optimal.item(1,0))
        min_error.append(min_err)

        # calculate new weights using newton method
        new_weight = newton_method_next_weight(w_optimal, new_weight, mu)

    # one more time to catch last weights
    mse = meanSquareError(edk2, new_weight, R, P)

    w0.append(new_weight.item(0,0))
    w1.append(new_weight.item(1,0))

    count.append(indx)
    MSE.append(mse)

    w0_opt.append(w_optimal.item(0,0))
    w1_opt.append(w_optimal.item(1,0))
    min_error.append(min_err)

    # if concat enabled continue with second parameters
    if (concat):
        cont=end
        end=end+iteration2

        for indx in range(cont, end):
            mse = meanSquareError(edk22, new_weight, R2, P2)

            w0.append(new_weight.item(0,0))
            w1.append(new_weight.item(1,0))

            count.append(indx)
            MSE.append(mse)

            w0_opt.append(w_optimal2.item(0,0))
            w1_opt.append(w_optimal2.item(1,0))
            min_error.append(min_err)

            new_weight = newton_method_next_weight(w_optimal2, new_weight, mu)

        mse = meanSquareError(edk22, new_weight, R2, P2)

        w0.append(new_weight.item(0,0))
        w1.append(new_weight.item(1,0))

        count.append(indx)
        MSE.append(mse)

        w0_opt.append(w_optimal.item(0,0))
        w1_opt.append(w_optimal.item(1,0))
        min_error.append(min_err)

    # if graph enabled graph values from list 
    if (graph):
        fig = plt.figure(figsize=(22, 10))

        wt0 = fig.add_subplot(3, 1, 1)
        wt1 = fig.add_subplot(3, 1, 2)

        lc = fig.add_subplot(3, 1, 3)

        wt0.plot(count, w0, linestyle='--')
        wt0.scatter(count, w0)
        wt0.plot(count, w0_opt)
        wt0.set_title('Weight Track for Weight 0')
        wt0.set_ylabel('W0', fontsize=20)
        if (concat):
            wt0.axvline(start+iteration, color='k',linestyle='--')

        wt1.plot(count, w1, linestyle='--')
        wt1.scatter(count, w1)
        wt1.plot(count, w1_opt)
        wt1.set_title('Weight Track for Weight 1')
        wt1.set_ylabel('W1', fontsize=20)
        if (concat):
            wt1.axvline(start+iteration, color='k',linestyle='--')

        lc.plot(count, MSE, linestyle='--')
        lc.scatter(count, MSE)
        lc.plot(count, min_error)
        lc.set_title('Learning Curve')
        lc.set_ylabel('MSE', fontsize=20)
        if (concat):
            lc.axvline(start+iteration, color='k',linestyle='--')

        fig.canvas.set_window_title('Newton Method')
        plt.suptitle("Newton Method u=" + str(mu), size=24)
        plt.xlabel('Interation Number, K', fontsize=20)
        plt.show()

    return new_weight

def steepest_descent_next_weight(w_optimal, w_current, R, mu):
    '''
    Calculate next weight using steepest descent method
    '''
    I = np.identity(2)
    w_next = (I-2*mu*R)*w_current + 2*mu*R*w_optimal

    return w_next

def steepest_descent_learning_curve(w_optimal, edk2, w_start, R, P, mu, min_err=0, start=0,
    iteration=15, graph=False, concat=False, w_optimal2=None, edk22=None, R2=None, P2=None,
    iteration2=15):
    '''
    Plot the learning curve and weight track for an interval using the
    steepest decent method
    graph: if true will display learning curve and weight track plot
    concat: can concatinate two intervals, need to supply second parameters
    '''
    # create empty list
    # to be filled through each iteration
    count = []
    MSE = []

    w0 = []
    w1 = []

    w0_opt = []
    w1_opt = []
    min_error = []

    new_weight = w_start

    end=start+iteration

    for indx in range(start, end):

        # calculate mean square error 
        mse = meanSquareError(edk2, new_weight, R, P)

        # add to list 
        w0.append(new_weight.item(0,0))
        w1.append(new_weight.item(1,0))

        count.append(indx)
        MSE.append(mse)

        w0_opt.append(w_optimal.item(0,0))
        w1_opt.append(w_optimal.item(1,0))
        min_error.append(min_err)

        # calculate new weights using steepest descent method
        new_weight = steepest_descent_next_weight(w_optimal, new_weight, R, mu)

    # one more time to catch last weights
    mse = meanSquareError(edk2, new_weight, R, P)

    w0.append(new_weight.item(0,0))
    w1.append(new_weight.item(1,0))

    count.append(indx)
    MSE.append(mse)

    w0_opt.append(w_optimal.item(0,0))
    w1_opt.append(w_optimal.item(1,0))
    min_error.append(min_err)

    # if concat enabled continue with second parameters
    if (concat):
        cont=end
        end=end+iteration2

        for indx in range(cont, end):
            mse = meanSquareError(edk22, new_weight, R2, P2)

            w0.append(new_weight.item(0,0))
            w1.append(new_weight.item(1,0))

            count.append(indx)
            MSE.append(mse)

            w0_opt.append(w_optimal2.item(0,0))
            w1_opt.append(w_optimal2.item(1,0))
            min_error.append(min_err)

            new_weight = steepest_descent_next_weight(w_optimal2, new_weight, R2, mu)
#            new_weight = newton_method_next_weight(w_optimal2, new_weight, mu)

        mse = meanSquareError(edk22, new_weight, R2, P2)

        w0.append(new_weight.item(0,0))
        w1.append(new_weight.item(1,0))

        count.append(indx)
        MSE.append(mse)

        w0_opt.append(w_optimal.item(0,0))
        w1_opt.append(w_optimal.item(1,0))
        min_error.append(min_err)

    # if graph enabled graph values from list 
    if (graph):
        fig = plt.figure(figsize=(22, 10))

        wt0 = fig.add_subplot(3, 1, 1)
        wt1 = fig.add_subplot(3, 1, 2)

        lc = fig.add_subplot(3, 1, 3)

        wt0.plot(count, w0, linestyle='--')
        wt0.scatter(count, w0)
        wt0.plot(count, w0_opt)
        wt0.set_title('Weight Track for Weight 0')
        wt0.set_ylabel('W0', fontsize=20)
        if (concat):
            wt0.axvline(start+iteration, color='k',linestyle='--')

        wt1.plot(count, w1, linestyle='--')
        wt1.scatter(count, w1)
        wt1.plot(count, w1_opt)
        wt1.set_title('Weight Track for Weight 1')
        wt1.set_ylabel('W1', fontsize=20)
        if (concat):
            wt1.axvline(start+iteration, color='k',linestyle='--')

        lc.plot(count, MSE, linestyle='--')
        lc.scatter(count, MSE)
        lc.plot(count, min_error)
        lc.set_title('Learning Curve')
        lc.set_ylabel('MSE', fontsize=20)
        if (concat):
            lc.axvline(start+iteration, color='k',linestyle='--')

        plt.xlabel('Interation Number, K', fontsize=20)
        fig.canvas.set_window_title('Steepest Descent')
        plt.suptitle("Steepest Descent u=" + str(mu), size=24)
        plt.show()

    return new_weight

def contour2d(w_optimal, START=-5, STOP=5, INCRAMENT=0.2, ALPHA=0.5, BETA=0.5, GAMMA=0.81,
              DELTA=1.176, EPSILON=0, ZETA=2, show_principle=False):
    '''
    2D contour plot for performance surface
    '''
    w_opt0 = w_optimal.item(0,0)
    w_opt1 = w_optimal.item(1,0)

    # weights variables
    w0 = np.arange(START, STOP, DELTA)
    w1 = np.arange(START, STOP, DELTA)

    x = np.arange(START, STOP, 1)
    y = np.arange(START, STOP, 1)

    x1=x+w_opt0
    y1=y+w_opt1

    y2=-1*y+w_opt1

    W0, W1 = np.meshgrid(w0, w1, sparse=False)

    # MSE equation for a two weight system
    MSE = ALPHA*W0*W0 + BETA*W1*W1 + GAMMA*W0*W1 + DELTA*W1 + EPSILON*W0 + ZETA

    # Plot figure
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(22, 10))
    con = fig.add_subplot(1, 1, 1)
    plt.xlabel('Weight-0', fontsize=20)
    plt.ylabel('Weight-1', fontsize=20)

    # add annotation to show optimal weight
    con.annotate('W*', xy=(w_opt0, w_opt1), xytext=(w_opt0+0.75, w_opt1+0.75), size=30,
                 arrowprops=dict(facecolor='black', shrink=1),
                )
    con.set_title('Performance Surface Contour Plot', fontsize=30)

    con.contour(W0, W1, MSE, 15)
    plt.plot(x1, y2)
    plt.plot(x1, y1)
    plt.show()

def contour3d(START=-5, STOP=5, INCRAMENT=0.2, ALPHA=0.5, BETA=0.5, GAMMA=0.81,
              DELTA=1.176, EPSILON=0, ZETA=2):
    '''
    3D contour plot for performance surface
    '''
    # weights variables
    w0 = np.arange(START, STOP, DELTA)
    w1 = np.arange(START, STOP, DELTA)

    W0, W1 = np.meshgrid(w0, w1, sparse=True)

    # MSE equation for a two weight system
    MSE = ALPHA*W0*W0 + BETA*W1*W1 + GAMMA*W0*W1 + DELTA*W1 + EPSILON*W0 + ZETA

    plt.style.use('ggplot')

    fig = plt.figure()
    fig = plt.figure(figsize=(22, 10))
    con3d = fig.add_subplot(111, projection='3d')
    con3d.set_title('3D Performance Surface Contour Plot', fontsize=30)
    con3d.plot_surface(W0, W1, MSE, rstride=4, cstride=4, color='b')

    con3d.set_xlabel('Weight 0', fontsize=15)
    con3d.set_ylabel('Weight 1', fontsize=15)
    con3d.set_zlabel('MSE', fontsize=15)

    plt.show()

if __name__ == '__main__':


    part1=False
    part2_c=True
    part3_c=True
    part3_d=False
    part3_e=False

    # calculate values for first interval
    print('\n 1 <= k <= 400 \n')

    r = findR(10)
    p = findP(10)

    w_opt = find_optimal_w(R=r, P=p)

    eigenValues(r)

    min_sqr_error(p, w_opt)

    # calculate values for second interval
    print('\n 401 <= k <= 800 \n')
    r2 = findR(20)
    p2 = findP(20)

    w_opt2 = find_optimal_w(R=r2, P=p2)

    eigenValues(r2)

    min_sqr_error(p2, w_opt2)

    if (part1):
        '''
        Part 1. Plot signal
        '''
        plotSineWaves()
        plotSineWaves(20, 401, 600)


    if (part2_c):
        '''
        Part 2. c) Plot contour level map and mesh
        for the two intervals
        '''
        #plot first interval
        contour2d(START=-5,STOP=5,INCRAMENT=0.5, w_optimal=w_opt)
        contour3d()

        # plot second interval
        contour2d(START=-10,STOP=10,INCRAMENT=0.5, w_optimal=w_opt2, GAMMA=0.952, EPSILON=0.628)
        contour3d(GAMMA=0.952, EPSILON=0.628)

    MU=0.5
    # starting weights
    w = np.matrix([[-4], [-1]])

    if (part3_c):
        '''
        Part 3. c) Plot weight track and learning curve
        for first interval
        '''
        # use newtons method show graph
        newton_method_learning_curve(w_optimal=w_opt, edk2=2, w_start=w,
        R=r, P=p, mu=MU,iteration=400, graph=True)

        # use steepest descent method show graph
        steepest_descent_learning_curve(w_optimal=w_opt, edk2=2, w_start=w,
        R=r, P=p, mu=MU,iteration=400, graph=True)

    if (part3_d):
        '''
        Part 3. d) Plot weight track and learning curve
        for second interval starting from the last value
        of the first interval
        '''
        # use newtons method save last value, no graph
        newton_last_points = newton_method_learning_curve(w_optimal=w_opt, edk2=2, w_start=w,
        R=r, P=p, mu=MU,iteration=400, graph=False)

        # use steepest descent method save last value, no graph
        steepest_last_points = steepest_descent_learning_curve(w_optimal=w_opt, edk2=2, w_start=w,
        R=r, P=p, mu=MU,iteration=400, graph=False)

        # use newtons method starting where first interval ended, show graph
        newton_method_learning_curve(w_optimal=w_opt2, edk2=2, w_start=newton_last_points, R=r2, P=p2,
        mu=MU,iteration=400, graph=True)

        # use steepest descent method starting where first interval ended, show graph
        steepest_descent_learning_curve(w_optimal=w_opt2, edk2=2, w_start=steepest_last_points, R=r2,
        P=p2, mu=MU,iteration=400, graph=True)

    if (part3_e):
        '''
        Part 3. e) concatenate both intervals and
        plot joint weight track and learning curve
        '''
        newton_method_learning_curve(w_optimal=w_opt, edk2=2, w_start=w, R=r, P=p, mu=MU,iteration=400,
        graph=True, concat=True, w_optimal2=w_opt2, edk22=2, R2=r2, P2=p2, iteration2=400)

        steepest_descent_learning_curve(w_optimal=w_opt, edk2=2, w_start=w, R=r, P=p, mu=MU,iteration=400,
        graph=True, concat=True, w_optimal2=w_opt2, edk22=2, R2=r2, P2=p2, iteration2=400)







