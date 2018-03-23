import numpy as np
from openmdao.api import Group, Problem, IndepVarComp, pyOptSparseDriver
from FLORISSE3D.setupOptimization import *
from FLORISSE3D.simpleTower import Tower
from FLORISSE3D.GeneralWindFarmComponents import calculate_boundary, SpacingComp,\
            BoundaryComp, get_z, getTurbineZ, AEPobj, DeMUX, hGroups, randomStart,\
            getRotorDiameter, getRatedPower, DeMUX, Myy_estimate, bladeLengthComp, minHeight,\
            SpacingConstraint
from FLORISSE3D.COE import COEGroup
from FLORISSE3D.floris import AEPGroup
from FLORISSE3D.rotorComponents import getRating, freqConstraintGroup, optCOE
from FLORISSE3D.SimpleRotorSE import SimpleRotorSE, create_rotor_functions
from rotorse.rotor import RotorSE
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow

from time import time

if __name__ == '__main__':
    """setup the turbine locations"""
    nGroups = 2

    rotor_diameter = 126.4

    turbineX, turbineY = amaliaLayout()
    mult = 1.5
    turbineX = turbineX*mult
    turbineY = turbineY*mult
    # turbineX = np.array([0.,0.,150.,200.])
    # turbineY = np.array([0.,100.,0.,800.])

    nTurbs = len(turbineX)

    nPoints = 3
    nFull = 15

    d_param = np.array([6.3,4.7515362,3.87])
    t_param = np.array([0.0200858,0.01623167,0.00975147])

    shearExp = 0.08
    ratedPower = np.array([5000.,5000.])
    rotorDiameter = np.array([126.4,70.])
    turbineZ = np.array([90.,90.])

    nDirs = 10
    nSpeeds = 2
    dirs, freqs, speeds = amaliaRose(nDirs)
    windDirections, windFrequencies, windSpeeds = setup_weibull(dirs,freqs,speeds,nSpeeds)

    print 'sampled'
    nDirections = len(windDirections)
    yaw = np.ones((nDirections, nTurbs))*0.

    """OpenMDAO"""

    start_setup = time()
    prob = Problem()
    root = prob.root = Group()

    # interp_spline_ratedQ, interp_spline_blade_mass, interp_spline_Vrated, interp_spline_I1, interp_spline_I2, interp_spline_I3, interp_spline_ratedT, interp_spline_extremeT = create_rotor_functions()

    # Design Variables
    for i in range(nGroups):
        root.add('ratedPower%s'%i, IndepVarComp('ratedPower%s'%i, float(ratedPower[i]), units='kW'), promotes=['*'])
        root.add('d_param%s'%i, IndepVarComp('d_param%s'%i, d_param), promotes=['*'])
        root.add('t_param%s'%i, IndepVarComp('t_param%s'%i, t_param), promotes=['*'])
        root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(turbineZ[i])), promotes=['*'])
        root.add('rotorDiameter%s'%i, IndepVarComp('rotorDiameter%s'%i, float(rotorDiameter[i])), promotes=['*'])

    root.add('optCOE', optCOE(nGroups,nPoints,nFull,nTurbs,nDirections),promotes=['*'])

    root.add('spacing_comp', SpacingComp(nTurbines=nTurbs), promotes=['*'])

    root.add('SpacingConstraint', SpacingConstraint(nTurbs), promotes=['*'])



    print 'added'

    for i in range(nGroups):
        root.connect('rotorDiameter%s'%i, 'Rotor%s.rotorDiameter'%i)
        root.connect('ratedPower%s'%i, 'Rotor%s.turbineRating'%i)

        root.connect('rotorDiameter%s'%i, 'Myy_estimate%s.rotor_diameter'%i)
        root.connect('rotorDiameter%s'%i,'bladeLengthComp%s.rotor_diameter'%i)
        root.connect('rotorDiameter%s'%i,'freqConstraintGroup%s.diameter'%i)

        root.connect('turbineH%s'%i, 'minHeight%s.height'%i)
        root.connect('rotorDiameter%s'%i, 'minHeight%s.diameter'%i)

    for i in range(nGroups):
        root.connect('d_param%s'%i, 'Tower%s_max_thrust.d_param'%i)
        root.connect('t_param%s'%i, 'Tower%s_max_thrust.t_param'%i)
        root.connect('d_param%s'%i, 'Tower%s_max_speed.d_param'%i)
        root.connect('t_param%s'%i, 'Tower%s_max_speed.t_param'%i)
        root.connect('d_param%s'%i, 'TowerDiscretization%s.d_param'%i)
        root.connect('t_param%s'%i, 'TowerDiscretization%s.t_param'%i)
    print 'connected'

    prob.root.ln_solver.options['single_voi_relevance_reduction'] = True
    prob.setup(check=True)
    print 'setup'

    end_setup = time()

    start_assign = time()

    setupTower(nFull, prob)
    simpleSetup(nTurbs, prob)
    prob['Uref'] = windSpeeds
    prob['windDirections'] = windDirections
    prob['windFrequencies'] = windFrequencies

    for i in range(nDirections):
        prob['yaw%s'%i] = yaw[i]
    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    prob['shearExp'] = shearExp

    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY

    for i in range(nGroups):
        prob['Tower%s_max_speed.Vel'%i] = 60.
    print 'running'
    prob.run()

    print 'COE: ', prob['COE']
    print 'AEP: ', prob['AEP']
    print 'rotorDiameter: ', prob['rotorDiameter']
    print 'wtSeparationSquared: ', prob['wtSeparationSquared']
    print 'spacing_con: ', prob['spacing_con']



    H1_H2 = prob['hGroup']

    fig = plt.gcf()
    ax = fig.gca()

    points = np.zeros((nTurbs,2))
    for j in range(nTurbs):
        points[j] = (turbineX[j],turbineY[j])
    hull = ConvexHull(points)

    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    color = (0.5, 0.1, 0.6)

    color_arrow = (0.4, 0.7, 0.9)

    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], 'k--')

    ax.add_artist(Circle(xy=(turbineX[0],turbineY[0]),
              radius=rotor_diameter/2., fill=False, edgecolor=color, lw=2.))#,label='Group 1')

    ax.add_artist(Circle(xy=(turbineX[1],turbineY[1]),
              radius=rotor_diameter/2., fill=False, edgecolor=color_arrow, lw=2.))#,label='Group 2')

    for j in range(nTurbs):
        if H1_H2[j] == 0:
            ax.add_artist(Circle(xy=(turbineX[j],turbineY[j]),
                      radius=rotor_diameter/2., fill=False, edgecolor=color, lw=2.))
        else:
            ax.add_artist(Circle(xy=(turbineX[j],turbineY[j]),
                      radius=rotor_diameter/2., fill=False, edgecolor=color_arrow, lw=2.))


    # plt.axis([min(turbineX)-1000.,max(turbineX)+1000.,min(turbineY)-1000.,max(turbineY)+1000.])
    plt.axis([min(turbineX)-100.,max(turbineX)+100.,min(turbineY)-100.,max(turbineY)+100.])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # plt.legend()
    # plt.savefig('optimizedLayout_legend.pdf', transparent=True)
    plt.axis('off')
    plt.show()
