import numpy as np
from openmdao.api import Group, Problem, IndepVarComp, pyOptSparseDriver, view_tree, profile
from FLORISSE3D.setupOptimization import *
from FLORISSE3D.simpleTower import Tower
from FLORISSE3D.GeneralWindFarmComponents import calculate_boundary, SpacingComp,\
            BoundaryComp, get_z, getTurbineZ, AEPobj, DeMUX, hGroups, randomStart,\
            getRotorDiameter, getRatedPower, DeMUX, Myy_estimate, bladeLengthComp, minHeight
from FLORISSE3D.COE import COEGroup
from FLORISSE3D.floris import AEPGroup
from FLORISSE3D.rotorComponents import getRating, freqConstraintGroup
import cPickle as pickle
from sys import argv
from rotorse.rotor import RotorSE
import os

from time import time

if __name__ == '__main__':
    nDirections = amaliaWind({})

    """setup the turbine locations"""
    nRows = 2
    nTurbs = nRows**2
    nGroups = 1
    spacing = 3.


    rotor_diameter = 126.4
    turbineX, turbineY = setupGrid(nRows, rotor_diameter, spacing)

    # turbineX = np.array([0.,   0.,   0.,   0.,    0.,    0.,    0.,    0.,    0.,    0.])
    # turbineY = np.array([0., 300., 600., 900., 1200., 1500., 1800., 2100., 2400., 2700.])

    turbineX = np.array([0.,   0.,   0.,   0.,    0.,    0.])
    turbineY = np.array([0., 300., 600., 900., 1200., 1500.])
    nTurbs = len(turbineX)

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

    shearExp = 0.08
    rotorDiameter = np.array([126.4, 70.,150.,155.,141.])
    turbineZ = np.array([120., 70., 100., 120., 30.])
    ratedPower = np.array([5000.,2000.,2000.,3000.,3004.])

    """OpenMDAO"""

    start_setup = time()
    prob = Problem()
    root = prob.root = Group()

    #Design Variables
    for i in range(nGroups):
        root.add('ratedPower%s'%i, IndepVarComp('ratedPower%s'%i, float(ratedPower[i]), units='kW'), promotes=['*'])
        root.add('d_param%s'%i, IndepVarComp('d_param%s'%i, d_param), promotes=['*'])
        root.add('t_param%s'%i, IndepVarComp('t_param%s'%i, t_param), promotes=['*'])
        root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(turbineZ[i])), promotes=['*'])
        root.add('rotorDiameter%s'%i, IndepVarComp('rotorDiameter%s'%i, float(rotorDiameter[i])), promotes=['*'])


    for i in range(nGroups):
        root.add('get_z_param%s'%i, get_z(nPoints)) #have derivatives
        root.add('get_z_full%s'%i, get_z(nFull)) #have derivatives
        root.add('Tower%s_max_thrust'%i, Tower(nPoints, nFull), promotes=['L_reinforced','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
        root.add('Tower%s_max_speed'%i, Tower(nPoints, nFull), promotes=['L_reinforced','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
        root.add('bladeLengthComp%s'%i, bladeLengthComp()) #have derivatives
        root.add('minHeight%s'%i, minHeight()) #have derivatives
        root.add('freqConstraintGroup%s'%i, freqConstraintGroup())

        topName = 'Rotor%s.'%i
        root.add('Rotor%s'%i, RotorSE(topName=topName, naero=17, nstr=38, npower=20)) #TODO check derivatives?
        root.add('split_I%s'%i, DeMUX(6)) #have derivatives
        root.add('Myy_estimate%s'%i, Myy_estimate()) #have derivatives

    root.add('Zs', DeMUX(nTurbs)) #have derivatives
    root.add('hGroups', hGroups(nTurbs, nGroups), promotes=['*']) #have derivatives
    root.add('getRotorDiameter', getRotorDiameter(nTurbs, nGroups), promotes=['*']) #have derivatives
    root.add('getRatedPower', getRatedPower(nTurbs, nGroups), promotes=['*'])    #have derivatives

    root.add('COEGroup', COEGroup(nTurbs, nGroups, nDirections, nPoints, nFull), promotes=['*']) #TODO check derivatives?

    root.connect('turbineZ', 'Zs.Array')
    # root.connect('air_density', 'rhoAir')

    for i in range(nGroups):
        root.connect('Rotor%s.ratedConditions:Q'%i, 'rotor_nacelle_costs%s.rotor_torque'%i)
        root.connect('ratedPower%s'%i, 'Rotor%s.control:ratedPower'%i) #TODO commented out line in RotorSE

        root.connect('Rotor%s.mass.blade_mass'%i, 'rotor_nacelle_costs%s.blade_mass'%i)
        root.connect('Rotor%s.setuppc.Vrated'%i,'Tower%s_max_thrust.Vel'%i)
        root.connect('Rotor%s.turbineclass.V_extreme'%i, 'Tower%s_max_speed.Vel'%i)
        root.connect('Rotor%s.I_all_blades'%i, 'split_I%s.Array'%i)
        root.connect('split_I%s.output%s'%(i,2),'Tower%s_max_speed.It'%i)
        root.connect('Tower%s_max_speed.It'%i,'Tower%s_max_thrust.It'%i)
        root.connect('Rotor%s.ratedConditions:T'%i,'Tower%s_max_thrust.Fx'%i)
        root.connect('Rotor%s.T_extreme'%i,'Tower%s_max_speed.Fx'%i)

        root.connect('rotorDiameter%s'%i, 'Myy_estimate%s.rotor_diameter'%i)
        root.connect('Myy_estimate%s.Myy'%i,'Tower%s_max_thrust.Myy'%i)
        root.connect('Myy_estimate%s.Myy'%i,'Tower%s_max_speed.Myy'%i)

        root.connect('rotorDiameter%s'%i,'bladeLengthComp%s.rotor_diameter'%i)
        root.connect('rotorDiameter%s'%i,'freqConstraintGroup%s.diameter'%i)
        root.connect('Tower%s_max_thrust.freq'%i,'freqConstraintGroup%s.freq'%i)
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

    prob.root.ln_solver.options['single_voi_relevance_reduction'] = True
    prob.setup(check=True)

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

    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY

    num = 100
    ratedPower = np.linspace(500.,10000.,num)
    rotorDiameter = np.linspace(25.,160.,num)
    # ratedQ = np.zeros((num, num))
    # ratedP = np.zeros((num, num))
    # blade_mass = np.zeros((num, num))
    # Vrated = np.zeros((num, num))
    # Vextreme = np.zeros((num, num))
    I1 = np.zeros((num, num))
    I2 = np.zeros((num, num))
    I3 = np.zeros((num, num))
    # ratedT = np.zeros((num, num))
    # extremeT = np.zeros((num, num))

    for i in range(num):
        for j in range(num):
            prob['ratedPower0'] = ratedPower[i]
            prob['rotorDiameter0'] = rotorDiameter[j]
            prob.run()
            # ratedQ[i][j] = float(prob['Rotor0.ratedConditions:Q'])
            # ratedP[i][j] = float(prob['Rotor0.control:ratedPower'])
            # blade_mass[i][j] = float(prob['Rotor0.mass.blade_mass'])
            # Vrated[i][j] = float(prob['Rotor0.setuppc.Vrated'])
            # Vextreme[i][j] = float(prob['Rotor0.turbineclass.V_extreme'])
            I1[i][j] = prob['Rotor0.I_all_blades'][0]
            I2[i][j] = prob['Rotor0.I_all_blades'][1]
            I3[i][j] = prob['Rotor0.I_all_blades'][2]
            # ratedT[i][j] = float(prob['Rotor0.ratedConditions:T'])
            # extremeT[i][j] = float(prob['Rotor0.T_extreme'])
            print i

    # np.savetxt('ratedQ50.txt', np.c_[ratedQ], header="ratedQ")
    # np.savetxt('ratedP50.txt', np.c_[ratedP], header="ratedP")
    # np.savetxt('blade_mass50.txt', np.c_[blade_mass], header="blade_mass")
    # np.savetxt('Vrated50.txt', np.c_[Vrated], header="Vrated")
    # np.savetxt('Vextreme50.txt', np.c_[Vextreme], header="Vextreme")
    np.savetxt('I1_100.txt', np.c_[I1], header="I_all_blades")
    np.savetxt('I2_100.txt', np.c_[I2], header="I_all_blades")
    np.savetxt('I3_100.txt', np.c_[I3], header="I_all_blades")
    # np.savetxt('ratedT50.txt', np.c_[ratedT], header="ratedT")
    # np.savetxt('extremeT50.txt', np.c_[extremeT], header="extremeT")

    # print 'ratedQ: ', 'np.',repr(ratedQ)
    # print 'ratedP: ', 'np.',repr(ratedP)
    # print 'blade_mass: ', 'np.',repr(blade_mass)
    # print 'Vrated: ', 'np.',repr(Vrated)
    # print 'Vextreme: ', 'np.',repr(Vextreme)
    print 'I1: ', 'np.',repr(I1)
    print 'I2: ', 'np.',repr(I2)
    print 'I3: ', 'np.',repr(I3)
    # print 'ratedT: ', 'np.',repr(ratedT)
    # print 'extremeT: ', 'np.',repr(extremeT)


    print 'AEP: ', prob['AEP']
    print 'COE: ', prob['COE']


    # for i in range(nGroups):
    #     print 'Q: ', prob['Rotor%s.ratedConditions:Q'%i]
    #
    #     print 'Height: ', prob['turbineH%s'%i]
    #     print 'Diameter: ', prob['rotorDiameter%s'%i]
    #     print 'Rating: ', prob['ratedPower%s'%i]
    #     print 'Diameters: ', prob['d_param%s'%i]
    #     print 'Thicknesses: ', prob['t_param%s'%i]
    #
    #     print 'MIN HEIGHT: ', prob['minHeight%s.minHeight'%i]
    #
    #     print 'SHELL BUCKLING'
    #     print 'TowerSE thrust: ', prob['Tower%s_max_thrust.shell_buckling'%i]
    #     print 'TowerSE speed: ', prob['Tower%s_max_speed.shell_buckling'%i]
