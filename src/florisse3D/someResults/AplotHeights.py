import matplotlib.pyplot as plt
import numpy as np

if __name__=='__main__':

    diameter = 126.4
    # diameter = 60.
    radius = diameter/2.
    spacing = 1.3

    H1 = 73.2
    H2 = 133.2
    d1 = np.array([7,4,3])
    d2 = np.array([10,9,5])
    x1 = 1.5*radius
    x2 = x1+diameter*spacing

    circle1 = plt.Circle((x1,H1), radius, color='blue', fill=False)
    circle2 = plt.Circle((x2, H2), radius, color='blue', fill=False)


    # (or if you have an existing figure)
    fig = plt.gcf()
    ax = fig.gca()

    ax.add_artist(circle1)
    ax.add_artist(circle2)

    px1 = np.array([x1-d1[0]/2,x1-d1[1]/2,x1-d1[2]/2,x1+d1[2]/2,x1+d1[1]/2,x1+d1[0]/2,x1-d1[0]/2])
    py1 = np.array([0,H1/2,H1,H1,H1/2,0,0])
    px2 = np.array([x2-d2[0]/2,x2-d2[1]/2,x2-d2[2]/2,x2+d2[2]/2,x2+d2[1]/2,x2+d2[0]/2,x2-d2[0]/2])
    py2 = np.array([0,H2/2,H2,H2,H2/2,0,0])
    ax.plot(px1,py1,color='blue')
    ax.plot(px2,py2,color='blue')

    plt.axis([0,x2+1.5*radius,-10,max(np.array([H1,H2]))+1.5*radius])
    plt.axes().set_aspect('equal')
    plt.show()
