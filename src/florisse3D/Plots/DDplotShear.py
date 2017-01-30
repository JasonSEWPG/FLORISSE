from FLORISSE3D.GeneralWindFarmComponents import PowWind
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from openmdao.api import Problem, Group
if __name__=="__main__":
    n = 1000
    z = np.linspace(0.,150.,n)
    Uref = np.array([10.])
    shearExp = .08
    nDirections = 1
    nTurbs = n

    """set up the problem"""
    prob = Problem()
    root = prob.root = Group()

    root.add('PowWind', PowWind(nDirections,nTurbs), promotes=['*'])

    prob.setup()

    prob['shearExp'] = shearExp
    prob['Uref'] = Uref
    prob['turbineZ'] = z


    prob.run()

    speeds = prob['windSpeeds']

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    plt.plot(speeds,z,'k',linewidth=4)
    plt.xlabel('Wind Speed')
    plt.ylabel('Height')
    plt.title(r'$\alpha=%s$'%shearExp)

    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

    matplotlib.rc('font', **font)

    plt.show()
