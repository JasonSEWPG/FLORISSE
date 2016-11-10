import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


if __name__=="__main__":
    nRows = 5
    rotor_diameter = 126.4
    nTurbs = nRows**2
    spacing = 5   # turbine grid spacing in diameters
    points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
    xpoints, ypoints = np.meshgrid(points, points)
    turbineXstart = np.ndarray.flatten(xpoints)
    turbineYstart = np.ndarray.flatten(ypoints)
    nTurbs = len(turbineXstart)

    """set up 3D aspects of wind farm"""
    H1_H2 = np.array([])
    for i in range(nTurbs/2):
        H1_H2 = np.append(H1_H2, 0)
        H1_H2 = np.append(H1_H2, 1)
    if len(H1_H2) < nTurbs:
        H1_H2 = np.append(H1_H2, 0)

    opt_filename = "practiceCOE.txt"
    # start_filename = "start.txt"

    opt = open(opt_filename)
    optimized = np.loadtxt(opt)
    turbineXopt = optimized[:,0]
    turbineYopt = optimized[:,1]
    turbineZopt = optimized[:,2]


    print turbineXopt

    # st = open(start_filename)
    # start = np.loadtxt(st)
    # turbineXstart = start[:,0]
    # turbineYstart = start[:,1]

    nTurbs = len(turbineXopt)

    # initialize axis, important: set the aspect ratio to equal
    fig, ax = plt.subplots(1, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    for i in range(nTurbs):
        print H1_H2[i]
        if H1_H2[i] == 0:
            ax.add_artist(Circle(xy=(turbineXopt[i],turbineYopt[i]),
                      radius=rotor_diameter/2., fill=False, edgecolor='blue'))
        else:
            ax.add_artist(Circle(xy=(turbineXopt[i],turbineYopt[i]),
                      radius=rotor_diameter/2., fill=False, edgecolor='red'))
    #
    # plt.plot(turbineXopt[0], turbineYopt[0], 'bo', label="H1")
    # plt.plot(turbineXopt[1], turbineYopt[1], 'b^', label="H2")
    #
    # for i in range(nTurbs):
    #     if H1_H2[i] == 0:
    #         plt.plot(turbineXstart[i], turbineYstart[i], 'ro')
    #         plt.plot(turbineXopt[i], turbineYopt[i], 'bo')
    #         plt.plot([turbineXstart[i], turbineXopt[i]], [turbineYstart[i], turbineYopt[i]], 'k--')
    #     if H1_H2[i] == 1:
    #         plt.plot(turbineXstart[i], turbineYstart[i], 'r^')
    #         plt.plot(turbineXopt[i], turbineYopt[i], 'b^')
    #         plt.plot([turbineXstart[i], turbineXopt[i]], [turbineYstart[i], turbineYopt[i]], 'k--')
    print 'turbineZ: ', turbineZopt
    ax.axis([min(turbineXopt)-200,max(turbineXopt)+200,min(turbineYopt)-200,max(turbineYopt)+200])
    # plt.axes().set_aspect('equal')
    # plt.legend()
    plt.show()
