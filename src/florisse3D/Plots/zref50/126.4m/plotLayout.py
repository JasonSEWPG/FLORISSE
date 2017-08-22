import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import ConvexHull



if __name__=="__main__":
    turbineX = np.array([7.415772326069553628e+02,
        2.243984601989541716e+03,
        2.375325682497517846e+03,
        3.463545355260799170e+03,
        3.249972357227155044e+03,
        7.415772326069553628e+02,
        3.009463701079065686e+03,
        7.415772326069553628e+02,
        1.541240484550012525e+03,
        2.321153003752757286e+03,
        7.415772326069553628e+02,
        3.478776390203879600e+03,
        1.487673149877896321e+03,
        2.228204179253073562e+03,
        2.262179955130288818e+03,
        7.415772326069553628e+02,
        1.088872546662104924e+03,
        2.185800773784705598e+03,
        2.615341850682047152e+03,
        3.488675152396151134e+03,
        3.488675152396151134e+03,
        3.084754674974436057e+03,
        3.488675152396151134e+03,
        1.450806679205157707e+03,
        1.332747524001721104e+03]) 

    turbineY = np.array([ 3.492264538024667218e+03,
        3.492264538024667218e+03,
        2.468576562131552237e+03,
        1.697525558597415056e+03,
        8.090118302075393331e+02,
        1.751403828964515242e+03,
        8.868806564036241298e+02,
        3.009419225825119156e+03,
        3.492264538024667218e+03,
        1.970882875788450974e+03,
        2.341955804212307157e+03,
        2.365491843055920981e+03,
        3.001781439098754618e+03,
        1.131972418465213650e+03,
        1.531161462006901502e+03,
        7.301059335118819718e+02,
        7.257720519004402604e+02,
        7.257720519004402604e+02,
        3.492264538024667218e+03,
        3.475870340291984121e+03,
        7.257720519004402604e+02,
        3.492264538024667218e+03,
        2.854967117922467878e+03,
        2.393107801391699013e+03,
        1.769402170755027555e+03])

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

    fig = plt.gcf()
    ax = fig.gca()

    spacingGrid = 5.537974684
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


    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], 'k--')

    for j in range(nTurbs):
        if H1_H2[j] == 0:
            ax.add_artist(Circle(xy=(turbineX[j],turbineY[j]),
                      radius=rotor_diameter/2., fill=False, edgecolor='blue'))
        else:
            ax.add_artist(Circle(xy=(turbineX[j],turbineY[j]),
                      radius=rotor_diameter/2., fill=False, edgecolor='red'))

    # ax.axis([min(turbineXopt)-200,max(turbineXopt)+200,min(turbineYopt)-200,max(turbineYopt)+200])
    # plt.axes().set_aspect('equal')
    # plt.legend()

    # plt.title('Optimized Turbine Layout')
    # plt.savefig('optimizedLayout.pdf', transparent=True)
    plt.axis('off')
    # plt.title('Optimized Turbine Layout\nShear Exponent = 0.1\nFarm Size = 144 rotor diameters squared')
    plt.show()
