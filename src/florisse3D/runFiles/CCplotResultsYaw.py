import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import ConvexHull
import math



if __name__=="__main__":

    nRows = 5
    rotor_diameter = 126.4
    nTurbs = nRows**2

    """set up 3D aspects of wind farm"""
    H1_H2 = np.array([])
    for i in range(nTurbs/2):
        H1_H2 = np.append(H1_H2, 0)
        H1_H2 = np.append(H1_H2, 1)
    if len(H1_H2) < nTurbs:
        H1_H2 = np.append(H1_H2, 0)

    space = np.array([2.0,2.5,3.0,3.5,4.0,5.0])

    # initialize axis, important: set the aspect ratio to equal
    # fig, ax = plt.subplots(1, len(space))
    fig = plt.gcf()
    ax = fig.gca()

    warmshear = np.array([0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3])
    warmshear = np.array([0.09,0.1,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3])

    for i in range(1):
        fig = plt.gcf()
        ax = fig.gca()
        # opt_filenameXYZ = 'XYZ_%s_'%space[i]+numRows+'_'+shear_Exp+'_'+optimizer+'_'+minSpacing+'_'+maxD+'.txt'
        # opt_filenameXYZ = 'XYZ_'+spacing+'_'+numRows+'_%s_' %shear_exSNOPT[i]+optimizer+'_'+minSpacing+'_'+maxD+'.txt'
        # opt_filenameXYZ = 'XYZ_%s_5_0.1_SNOPT_amaliaWind.txt'%space[i]
        opt_filenameXYZ = 'XYZ_3.1_5_0.1_SP_2_6.3.txt'
        yaw = np.array([-25.77386411 ,  0.36249738,  20.52707388,  14.52279668 ,  0.  ,        10.60043967,
    0.31658236,  21.36967701,   0.  ,         0.   ,       12.03989514,
   12.33125514,  22.30858498,   0. ,          0.  ,         7.96646751,
   -5.63804421  , 0.  ,        -7.32507205 ,  0.,           0.   ,        0.,
    0.    ,       0.   ,        0. ,       ])

        initial = -45.
        yaw += initial
        for j in range(nTurbs):
            yaw[j] = math.radians(yaw[j])
        print yaw
        # print opt_filenameXYZ

        optXYZ = open(opt_filenameXYZ)
        optimizedXYZ = np.loadtxt(optXYZ)
        turbineX = optimizedXYZ[:,0]
        turbineY = optimizedXYZ[:,1]
        Z = optimizedXYZ[:,2]
        print Z


        print turbineX

        # st = open(start_filename)
        # start = np.loadtxt(st)
        # turbineXstart = start[:,0]
        # turbineYstart = start[:,1]

        # spacingGrid = shear[i]   # turbine grid spacing in diameters
        spacingGrid = warmshear[i]
        spacingGrid = 3.1
        points = np.linspace(start=spacingGrid*rotor_diameter, stop=nRows*spacingGrid*rotor_diameter, num=nRows)
        xpoints, ypoints = np.meshgrid(points, points)
        turbineXstart = np.ndarray.flatten(xpoints)
        turbineYstart = np.ndarray.flatten(ypoints)
        points = np.zeros((nTurbs,2))
        for j in range(nTurbs):
            points[j] = (turbineXstart[j],turbineYstart[j])
        hull = ConvexHull(points)

        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # ax[i].text(0, -100, '%s'%space[i], fontsize=20)




        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], 'k--')

        d = rotor_diameter/2.


        for j in range(nTurbs):
            if H1_H2[j] == 0:
                #blue
                x = turbineX[j]
                y = turbineY[j]
                theta = yaw[j]
                xt = x+d*np.sin(theta)
                xb = x-d*np.sin(theta)
                yt = y+d*np.cos(theta)
                yb = y-d*np.cos(theta)
                plt.plot([xt,xb],[yt,yb],'b',linewidth=2)
            else:
                #red
                x = turbineX[j]
                y = turbineY[j]
                theta = yaw[j]
                xt = x+d*np.sin(theta)
                xb = x-d*np.sin(theta)
                yt = y+d*np.cos(theta)
                yb = y-d*np.cos(theta)
                plt.plot([xt,xb],[yt,yb],'r',linewidth=2)

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
        # print 'turbineZ: ', turbineZopt
        # plt.axes().set_aspect('equal', 'datalim')
        ax.axis([min(turbineX)-400,max(turbineX)+200,min(turbineY)-200,max(turbineY)+200])
        # plt.axes().set_aspect('equal')
        # plt.legend()

        # plt.title('Optimized Turbine Layout')
        # plt.savefig('optimizedLayout.pdf', transparent=True)
        plt.axis('off')
        ax = plt.axes()
        ax.arrow(-100, 600, 300, 300, head_width=50, head_length=100, fc='k', ec='k')
        ax.text(2, 600, '    wind\n direction', fontsize=10)
        plt.show()
