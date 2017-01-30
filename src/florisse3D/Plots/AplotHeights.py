import matplotlib.pyplot as plt
import numpy as np

if __name__=='__main__':

    opt_filenameXYZ = 'Z_XYZ_dt_3.0.txt'
    opt_filename_dt = 'Z_dt_dt_3.0.txt'

    optXYZ = open(opt_filenameXYZ)
    optimizedXYZ = np.loadtxt(optXYZ)
    turbineX = optimizedXYZ[:,0]
    turbineY = optimizedXYZ[:,1]
    turbineZ = optimizedXYZ[:,2]
    turbineH1 = turbineZ[0]
    turbineH2 = turbineZ[1]

    opt_dt = open(opt_filename_dt)
    optimized_dt = np.loadtxt(opt_dt)
    d_paramH1 = optimized_dt[:,0]
    d_paramH2 = optimized_dt[:,1]
    t_paramH1 = optimized_dt[:,2]
    t_paramH2 = optimized_dt[:,3]

    diameter = 126.4
    diameter2 = 90.
    # diameter = 60.
    radius = diameter/2.
    # r2 = diameter2/2.
    r2 = radius
    spacing = 1.3

    H1 = turbineH1
    H2 = turbineH2
    print H1
    print H2
    d1 = d_paramH1
    d2 = d_paramH2
    x1 = 1.5*radius
    x2 = x1+diameter*spacing

    color = (0,0.6,0.8)

    circle1 = plt.Circle((x1,H1), radius, color=color, fill=False)
    circle2 = plt.Circle((x2, H2), r2, color=color, fill=False)


    # (or if you have an existing figure)
    fig = plt.gcf()
    ax = fig.gca()

    ax.add_artist(circle1)
    ax.add_artist(circle2)

    px1 = np.array([x1-d1[0]/2,x1-d1[1]/2,x1-d1[2]/2,x1+d1[2]/2,x1+d1[1]/2,x1+d1[0]/2,x1-d1[0]/2])
    py1 = np.array([0,H1/2,H1,H1,H1/2,0,0])
    px2 = np.array([x2-d1[0]/2,x2-d1[1]/2,x2-d1[2]/2,x2+d1[2]/2,x2+d1[1]/2,x2+d1[0]/2,x2-d1[0]/2])
    py2 = np.array([0,H2/2,H2,H2,H2/2,0,0])
    ax.plot(px1,py1,color=color)
    ax.plot(px2,py2,color=color)

    plt.axis([0,x2+1.5*radius,-10,max(np.array([H1,H2]))+1.5*radius])
    plt.axes().set_aspect('equal')
    # plt.title('Spacing: %s'%space)
    ax.get_xaxis().set_visible(False)
    plt.ylabel('Height (m)', fontsize=15)
    # plt.title('Optimized Heights', fontsize=15)
    # fig.savefig('optimizedHeights.pdf', transparent=True)
    plt.show()
