import numpy as np
from openmdao.api import Group, Problem, IndepVarComp, pyOptSparseDriver, view_tree, profile
from FLORISSE3D.setupOptimization import *
from FLORISSE3D.simpleTower import Tower
from FLORISSE3D.GeneralWindFarmComponents import calculate_boundary, SpacingComp,\
            BoundaryComp, get_z, getTurbineZ, AEPobj, DeMUX, hGroups, randomStart,\
            getRotorDiameter, getRatedPower, DeMUX, Myy_estimate, bladeLengthComp, minHeight
from FLORISSE3D.COE import COEGroup
from FLORISSE3D.floris import AEPGroup
from FLORISSE3D.rotorComponents import getRating
import cPickle as pickle
from sys import argv
from rotorse.rotor import RotorSE
import os

from time import time

if __name__ == '__main__':
    nDirections = amaliaWind({})

    """setup the turbine locations"""
    nRows = 5
    nTurbs = nRows**2
    nGroups = 2
    spacing = 3.


    rotor_diameter = 126.4
    turbineX, turbineY = setupGrid(nRows, rotor_diameter, spacing)

    # turbineX = np.array([0.,   0.,   0.,   0.,    0.,    0.,    0.,    0.,    0.,    0.])
    # turbineY = np.array([0., 300., 600., 900., 1200., 1500., 1800., 2100., 2400., 2700.])

    turbineX = np.array([0.,   0.,   0.,   0.,    0.,    0.])
    turbineY = np.array([0., 300., 600., 900., 1200., 1500.])
    nTurbs = len(turbineX)

    # turbineX = np.array([0.])
    # turbineY = np.array([0.])
    # nTurbs = 1
    # nGroups = 1
    nDirections = 1
    # nVertices, boundaryVertices, boundaryNormals = setupBoundaryConstraints(turbineX, turbineY)



    # density = 0.1354922143
    # spacing = nRows/(2.*nRows-2.)*np.sqrt(3.1415926535/density)
    # print 'SPACING: ', spacing


    minSpacing = 2.0

    """initial yaw values"""
    yaw = np.zeros((nDirections, nTurbs))


    nPoints = 3
    nFull = 15

    d_param = np.array([6.3,5.3,4.3])
    t_param = np.array([0.02,0.015,0.01])

    shearExp = 0.15
    rotorDiameter = np.array([126.4, 70.,150.,155.,141.])
    turbineZ = np.array([120., 70., 100., 120., 30.])
    ratedPower = np.array([100.,200.,2000.,3000.,3004.])

    """OpenMDAO"""

    start_setup = time()
    prob = Problem()
    root = prob.root = Group()

    for i in range(nGroups):
        root.add('ratedPower%s'%i, IndepVarComp('ratedPower%s'%i, float(ratedPower[i])), promotes=['*'])
        root.add('d_param%s'%i, IndepVarComp('d_param%s'%i, d_param), promotes=['*'])
        root.add('t_param%s'%i, IndepVarComp('t_param%s'%i, t_param), promotes=['*'])
        root.add('get_z_param%s'%i, get_z(nPoints))
        root.add('get_z_full%s'%i, get_z(nFull))
        root.add('Tower%s_max_thrust'%i, Tower(nPoints, nFull), promotes=['L_reinforced','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
        root.add('Tower%s_max_speed'%i, Tower(nPoints, nFull), promotes=['L_reinforced','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
        root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(turbineZ[i])), promotes=['*'])
        root.add('rotorDiameter%s'%i, IndepVarComp('rotorDiameter%s'%i, float(rotorDiameter[i])), promotes=['*'])
        root.add('bladeLengthComp%s'%i, bladeLengthComp())
        root.add('minHeight%s'%i, minHeight())


    root.add('Zs', DeMUX(nTurbs))
    root.add('hGroups', hGroups(nTurbs, nGroups), promotes=['*'])
    root.add('getRotorDiameter', getRotorDiameter(nTurbs, nGroups), promotes=['*'])
    root.add('getRatedPower', getRatedPower(nTurbs, nGroups), promotes=['*'])

    for i in range(nGroups):
        topName = 'Rotor%s.'%i
        root.add('Rotor%s'%i, RotorSE(topName=topName, naero=17, nstr=38, npower=20))
        root.add('split_I%s'%i, DeMUX(6))
        root.add('Myy_estimate%s'%i, Myy_estimate())

    root.add('COEGroup', COEGroup(nTurbs, nGroups, nDirections, nPoints, nFull), promotes=['*'])

    # root.add('spacing_comp', SpacingComp(nTurbines=nTurbs), promotes=['*'])

    # add constraint definitions
    # root.add('spacing_con', ExecComp('sc = wtSeparationSquared-(minSpacing*rotorDiameter[0])**2',
    #                              minSpacing=minSpacing, rotorDiameter=np.zeros(nTurbs),
    #                              sc=np.zeros(((nTurbs-1)*nTurbs/2)),
    #                              wtSeparationSquared=np.zeros(((nTurbs-1)*nTurbs/2))),
    #                              promotes=['*'])

    # if nVertices > 0:
    #     # add component that enforces a convex hull wind farm boundary
    #     root.add('boundary_con', BoundaryComp(nVertices=nVertices, nTurbines=nTurbs), promotes=['*'])

    root.connect('turbineZ', 'Zs.Array')
    # root.connect('air_density', 'rhoAir')

    for i in range(nGroups):
        root.connect('Rotor%s.mass.blade_mass'%i, 'rotor_nacelle_costs%s.blade_mass'%i)
        root.connect('Rotor%s.setuppc.Vrated'%i,'Tower%s_max_thrust.Vel'%i)
        root.connect('Rotor%s.turbineclass.V_extreme'%i, 'Tower%s_max_speed.Vel'%i)
        root.connect('Rotor%s.I_all_blades'%i, 'split_I%s.Array'%i)
        root.connect('split_I%s.output%s'%(i,2),'Tower%s_max_speed.It'%i)
        root.connect('Tower%s_max_speed.It'%i,'Tower%s_max_thrust.It'%i)
        root.connect('Rotor%s.ratedConditions:T'%i,'Tower%s_max_thrust.Fx'%i)
        root.connect('Rotor%s.T_extreme'%i,'Tower%s_max_speed.Fx'%i)
        root.connect('Rotor%s.ratedConditions:Q'%i,'Tower%s_max_thrust.Mxx'%i)
        root.connect('Rotor%s.Q_extreme'%i,'Tower%s_max_speed.Mxx'%i)
        root.connect('rotorDiameter%s'%i, 'Myy_estimate%s.rotor_diameter'%i)
        root.connect('rotorDiameter%s'%i, 'Myy_estimate%s.rotor_diameter'%i)
        root.connect('Myy_estimate%s.Myy'%i,'Tower%s_max_thrust.Myy'%i)
        root.connect('Myy_estimate%s.Myy'%i,'Tower%s_max_speed.Myy'%i)

        root.connect('rotorDiameter%s'%i,'bladeLengthComp%s.rotor_diameter'%i)
        # root.connect('Rotor%s.hubFraction'%i,'bladeLengthComp%s.hubFraction'%i) #breaks when I connect this,
                                                    # I just defined a default value of 0.025 in the component
        root.connect('bladeLengthComp%s.blade_length'%i,'Rotor%s.bladeLength'%i)

        root.connect('turbineH%s'%i, 'minHeight%s.height'%i)
        root.connect('rotorDiameter%s'%i, 'minHeight%s.diameter'%i)

    for i in range(nGroups):
        root.connect('rotor_nacelle_costs%s.rotor_mass'%i, 'Tower%s_max_speed.rotor_mass'%i)
        root.connect('rotor_nacelle_costs%s.nacelle_mass'%i, 'Tower%s_max_speed.nacelle_mass'%i)

        root.connect('Tower%s_max_speed.rotor_mass'%i, 'Tower%s_max_thrust.rotor_mass'%i)
        root.connect('Tower%s_max_speed.nacelle_mass'%i, 'Tower%s_max_thrust.nacelle_mass'%i)

    for j in range(nGroups):
        root.connect('rotor_diameters%s'%j,'rotor_nacelle_costs%s.rotor_diameter'%j)
        root.connect('rated_powers%s'%j,'rotor_nacelle_costs%s.machine_rating'%j)

    for i in range(nGroups):
        root.connect('get_z_param%s.z_param'%i, 'Tower%s_max_thrust.z_param'%i)
        root.connect('get_z_full%s.z_param'%i, 'Tower%s_max_thrust.z_full'%i)
        root.connect('get_z_param%s.z_param'%i, 'Tower%s_max_speed.z_param'%i)
        root.connect('get_z_full%s.z_param'%i, 'Tower%s_max_speed.z_full'%i)

        root.connect('Zs.output%s'%i, 'get_z_param%s.turbineZ'%i)
        root.connect('Zs.output%s'%i, 'get_z_full%s.turbineZ'%i)
        root.connect('Zs.output%s'%i, 'Tower%s_max_thrust.L'%i)
        root.connect('Zs.output%s'%i, 'Tower%s_max_speed.L'%i)

        root.connect('get_z_param%s.z_param'%i, 'Tower%s_max_thrust.z_param'%i)
        root.connect('get_z_full%s.z_param'%i, 'Tower%s_max_thrust.z_full'%i)

        root.connect('d_param%s'%i, 'Tower%s_max_thrust.d_param'%i)
        root.connect('t_param%s'%i, 'Tower%s_max_thrust.t_param'%i)
        root.connect('d_param%s'%i, 'Tower%s_max_speed.d_param'%i)
        root.connect('t_param%s'%i, 'Tower%s_max_speed.t_param'%i)

        root.connect('get_z_param%s.z_param'%i, 'TowerDiscretization%s.z_param'%i)
        root.connect('get_z_full%s.z_param'%i, 'TowerDiscretization%s.z_full'%i)
        root.connect('d_param%s'%i, 'TowerDiscretization%s.d_param'%i)
        root.connect('t_param%s'%i, 'TowerDiscretization%s.t_param'%i)
        root.connect('rho', 'calcMass%s.rho'%i)

        # root.connect('Loads%s.Fx1'%i, 'Tower%s_max_thrust.Fx'%i)
        # root.connect('Loads%s.Fy1'%i, 'Tower%s_max_thrust.Fy'%i)
        # root.connect('Loads%s.Fz1'%i, 'Tower%s_max_thrust.Fz'%i)
        # root.connect('Loads%s.Mxx1'%i, 'Tower%s_max_thrust.Mxx'%i)
        # root.connect('Loads%s.Myy1'%i, 'Tower%s_max_thrust.Myy'%i)
        #
        # root.connect('Loads%s.Fx2'%i, 'Tower%s_max_speed.Fx'%i)
        # root.connect('Loads%s.Fy2'%i, 'Tower%s_max_speed.Fy'%i)
        # root.connect('Loads%s.Fz2'%i, 'Tower%s_max_speed.Fz'%i)
        # root.connect('Loads%s.Mxx2'%i, 'Tower%s_max_speed.Mxx'%i)
        # root.connect('Loads%s.Myy2'%i, 'Tower%s_max_speed.Myy'%i)
        #
        # root.connect('Loads%s.m'%i, 'Tower%s_max_speed.Mt'%i)
        # root.connect('Loads%s.mIzz'%i, 'Tower%s_max_speed.It'%i)

    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings['Major iterations limit'] = 1000
    prob.driver.opt_settings['Major optimality tolerance'] = 1.1E-4
    prob.driver.opt_settings['Major feasibility tolerance'] = 1.1E-4

    prob.driver.opt_settings['Summary file'] = 'SNOPT.out'

    root.deriv_options['type'] = 'fd'
    root.deriv_options['form'] = 'central'
    root.deriv_options['step_size'] = 1.E-4
    root.deriv_options['step_calc'] = 'relative'

    prob.driver.add_objective('COE', scaler=0.1)

    for i in range(nGroups):
        # prob.driver.add_desvar('d_param%s'%i, lower=3.87, upper=6.3, scaler=1.)
        # prob.driver.add_desvar('t_param%s'%i, lower=0.001, upper=None, scaler=0.1)
        # prob.driver.add_desvar('turbineH%s'%i, lower=10., scaler=100.)
        # prob.driver.add_desvar('rotorDiameter%s'%i, lower=10., upper=None, scaler=1.)
        prob.driver.add_desvar('ratedPower%s'%i, lower=0., upper=10000., scaler=1.)
        #TODO add design variables for blade design

    for i in range(nGroups):
        prob.driver.add_constraint('Tower%s_max_thrust.shell_buckling'%i, upper=np.ones(nFull))
        prob.driver.add_constraint('Tower%s_max_speed.shell_buckling'%i, upper=np.ones(nFull))
        freq1p = 0.2  # 1P freq in Hz TODO how to do this?
        #TODO ADD frequency constraint
        # prob.driver.add_constraint('Tower%s_max_thrust.freq'%i, lower=1.1*freq1p)
        # prob.driver.add_constraint('Tower%s_max_speed.freq'%i, lower=1.1*freq1p)
        prob.driver.add_constraint('minHeight%s.minHeight'%i, lower=0.0)

    prob.root.ln_solver.options['single_voi_relevance_reduction'] = True
    profile.setup(prob)
    profile.start()
    prob.setup(check=True)
    # view_tree(prob, offline=False)
    end_setup = time()
    start_assign = time()

    # amaliaWind(prob)
    setupTower(nFull, prob)
    simpleSetup(nTurbs, prob)
    setupRotor(nGroups, prob)
    prob['Uref'] = np.array([10.])
    prob['windDirections'] = np.array([0.])
    prob['windFrequencies'] = np.array([1.])

    for i in range(nDirections):
        prob['yaw%s'%i] = yaw[i]
    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    prob['shearExp'] = shearExp

    # prob['boundaryVertices'] = boundaryVertices
    # prob['boundaryNormals'] = boundaryNormals

    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    end_assign = time()


    prob.run()

    print 'AEP: ', prob['AEP']
    print 'COE: ', prob['COE']


    for i in range(nGroups):

        print 'Height: ', prob['turbineH%s'%i]
        print 'Diameter: ', prob['rotorDiameter%s'%i]
        print 'Rating: ', prob['ratedPower%s'%i]
        print 'Diameters: ', prob['d_param%s'%i]
        print 'Thicknesses: ', prob['t_param%s'%i]

        print 'MIN HEIGHT: ', prob['minHeight%s.minHeight'%i]

        print 'SHELL BUCKLING'
        print 'TowerSE thrust: ', prob['Tower%s_max_thrust.shell_buckling'%i]
        print 'TowerSE speed: ', prob['Tower%s_max_speed.shell_buckling'%i]
