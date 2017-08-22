import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from math import sin, cos, radians

if __name__=='__main__':

    H1 = 1.061449994084667026e+02
    H2 = 73.2



    d1 = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d2 = np.array([4.580021876109398704e+00, 4.494543899393605102e+00, 3.870000000000000107e+00])



    diameter = 126.4
    radius = diameter/2.

    spacing = 1.1

    r = 126.4/2.

    x1 = 1.25*r
    x2 = x1+126.4*spacing

    color = (0,0.6,0.8)

    circle1 = plt.Circle((x1,H1), radius, color='blue', fill=False, linestyle = '--', linewidth=3)
    circle2 = plt.Circle((x2, H2), radius, color='red', fill=False, linestyle = '--', linewidth=3)


    # (or if you have an existing figure)
    fig = plt.gcf()
    ax = fig.gca()

    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

    matplotlib.rc('font', **font)

    ax.add_artist(circle1)
    ax.add_artist(circle2)

    px1 = np.array([x1-d1[0]/2,x1-d1[1]/2,x1-d1[2]/2,x1+d1[2]/2,x1+d1[1]/2,x1+d1[0]/2,x1-d1[0]/2])
    py1 = np.array([0,H1/2,H1-5.,H1-5.,H1/2,0,0])
    px2 = np.array([x2-d2[0]/2,x2-d2[1]/2,x2-d2[2]/2,x2+d2[2]/2,x2+d2[1]/2,x2+d2[0]/2,x2-d2[0]/2])
    py2 = np.array([0,H2/2,H2-5.,H2-5.,H2/2,0,0])
    ax.plot(px1,py1,color='blue', linewidth=3)
    ax.plot(px2,py2,color='red', linewidth=3)


    #add blades
    hub1 = plt.Circle((x1,H1), 5, color='blue', fill=False, linewidth=2)
    hub2 = plt.Circle((x2,H2), 5, color='red', fill=False, linewidth=2)
    ax.add_artist(hub1)
    ax.add_artist(hub2)
    bladeX = np.array([5.*70./126.4,7.,10.,15.,20.,25.,30.,35.,35.,30.,25.,20.,15.,10.,5.*70./126.4,5.*70./126.4,5.*70./126.4])*(126.4/70.)
    bladeY = np.array([0.,0.,0.8,1.5,1.55,1.6,1.7,1.75,2.9,2.9,2.9,2.9,2.9,2.9,2.9,2.9,0.])*(126.4/70.)-2.5

    angle1 = -75.

    blade1X = bladeX*cos(radians(angle1))-bladeY*sin(radians(angle1))
    blade1Y = bladeX*sin(radians(angle1))+bladeY*cos(radians(angle1))

    blade2X = bladeX*cos(radians(angle1+120.))-bladeY*sin(radians(angle1+120.))
    blade2Y = bladeX*sin(radians(angle1+120.))+bladeY*cos(radians(angle1+120.))

    blade3X = bladeX*cos(radians(angle1+240.))-bladeY*sin(radians(angle1+240.))
    blade3Y = bladeX*sin(radians(angle1+240.))+bladeY*cos(radians(angle1+240.))

    ax.plot(blade1X+x1, blade1Y+H1, linewidth=2, color='blue')
    ax.plot(blade2X+x1, blade2Y+H1, linewidth=2, color='blue')
    ax.plot(blade3X+x1, blade3Y+H1, linewidth=2, color='blue')


    angle2 = -10.

    blade1X = bladeX*cos(radians(angle2))-bladeY*sin(radians(angle2))
    blade1Y = bladeX*sin(radians(angle2))+bladeY*cos(radians(angle2))

    blade2X = bladeX*cos(radians(angle2+120.))-bladeY*sin(radians(angle2+120.))
    blade2Y = bladeX*sin(radians(angle2+120.))+bladeY*cos(radians(angle2+120.))

    blade3X = bladeX*cos(radians(angle2+240.))-bladeY*sin(radians(angle2+240.))
    blade3Y = bladeX*sin(radians(angle2+240.))+bladeY*cos(radians(angle2+240.))

    ax.plot(blade1X+x2, blade1Y+H2, linewidth=2, color='red')
    ax.plot(blade2X+x2, blade2Y+H2, linewidth=2, color='red')
    ax.plot(blade3X+x2, blade3Y+H2, linewidth=2, color='red')





    ax.set_xlim([0,x2+1.5*r])
    ax.set_ylim([0,190])
    plt.axes().set_aspect('equal')

    ax.set_ylabel('Optimized Hub Height (m)')
    ax.set_xticks([x1,x2])
    ax.set_yticks([25.,50.,75.,100.,125.,150.,175.])
    ax.set_xticklabels(['group 1', 'group 2'])

    # fig.savefig('optimizedHeights.pdf', transparent=True)
    plt.title('Large Rotor Farm')
    plt.tight_layout()
    plt.show()
