import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Circle
from scipy.spatial import ConvexHull



if __name__=="__main__":

    nRows = 9
    rotor_diameter = 126.4
    nTurbs = nRows**2

    """set up 3D aspects of wind farm"""
    H1_H2 = np.array([])
    for i in range(nTurbs/2):
        H1_H2 = np.append(H1_H2, 0)
        H1_H2 = np.append(H1_H2, 1)
    if len(H1_H2) < nTurbs:
        H1_H2 = np.append(H1_H2, 0)

    spacing = 4.0

    points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
    xpoints, ypoints = np.meshgrid(points, points)
    turbineX = np.ndarray.flatten(xpoints)
    turbineY = np.ndarray.flatten(ypoints)

    fig = plt.gcf()
    ax = fig.gca()

    s = 8.0
    fig = plt.gcf()
    ax = fig.gca()

    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

    matplotlib.rc('font', **font)

    spacingGrid = 4
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

    r1 = 70.
    r2 = 70.

    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], 'k--')

    color = (0,0.6,0.8)
    ax.add_artist(Circle(xy=(turbineX[0],turbineY[j]),
              radius=r1, fill=False, edgecolor='red'))
    ax.add_artist(Circle(xy=(turbineX[1],turbineY[j]),
              radius=r2, fill=False, edgecolor='blue'))
    for j in range(nTurbs):
        if H1_H2[j] == 0:
            ax.add_artist(Circle(xy=(turbineX[j],turbineY[j]),
                      radius=r1, fill=False, edgecolor='red'))
        else:
            ax.add_artist(Circle(xy=(turbineX[j],turbineY[j]),
                      radius=r2, fill=False, edgecolor='blue'))

    ax.axis([min(turbineX)-200,max(turbineX)+200,min(turbineY)-200,max(turbineY)+200])
    ax.legend()
    plt.title('Medium Wind Farm Layout with Two Height Groups')

    plt.axis('off')
    plt.show()
