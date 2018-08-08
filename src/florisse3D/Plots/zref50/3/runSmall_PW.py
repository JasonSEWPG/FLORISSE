import numpy as np
from openmdao.api import Group, Problem, IndepVarComp, pyOptSparseDriver
from FLORISSE3D.setupOptimization import *
from FLORISSE3D.simpleTower import Tower
from FLORISSE3D.GeneralWindFarmComponents import calculate_boundary, SpacingComp,\
            BoundaryComp, get_z, getTurbineZ, AEPobj, DeMUX, hGroups, randomStart
from FLORISSE3D.COE import COEGroup
from FLORISSE3D.floris import AEPGroup
import cPickle as pickle
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Circle
from sys import argv
import os
import random


if __name__ == '__main__':
    nGroups = 1
    shearExp = 0.08
    use_rotor_components = False

    datasize = 0
    rotor_diameter = 70.0

    nRows = 9
    nTurbs = nRows**2

    turbineX = np.array([2.100000000233957849e+02,    4.200000000467915697e+02,    6.300000000701872978e+02,
    8.400000000935831395e+02, 1.050000000116978981e+03, 1.260000000140374823e+03, 1.470000000163770437e+03,
    1.680000000187166279e+03, 1.890000000210562121e+03, 2.100000000233957849e+02, 4.200000000467915697e+02,
    6.300000000701872978e+02, 8.400000000935831395e+02, 1.050000000116978981e+03, 1.260000000140374823e+03,
    1.470000000163770437e+03, 1.680000000187166279e+03, 1.890000000210562121e+03, 2.100000000233957849e+02,
    4.200000000467915697e+02, 6.300000000701872978e+02, 8.400000000935831395e+02, 1.050000000116978981e+03,
    1.260000000140374823e+03, 1.470000000163770437e+03, 1.680000000187166279e+03, 1.890000000210562121e+03,
    2.100000000233957849e+02, 4.200000000467915697e+02, 6.300000000701872978e+02, 8.400000000935831395e+02,
    1.050000000116978981e+03, 1.260000000140374823e+03, 1.470000000163770437e+03, 1.680000000187166279e+03,
    1.890000000210562121e+03, 2.100000000233957849e+02, 4.200000000467915697e+02, 6.300000000701872978e+02,
    8.400000000935831395e+02, 1.050000000116978981e+03, 1.260000000140374823e+03, 1.470000000163770437e+03,
    1.680000000187166279e+03, 1.890000000210562121e+03, 2.100000000233957849e+02, 4.200000000467915697e+02,
    6.300000000701872978e+02, 8.400000000935831395e+02, 1.050000000116978981e+03, 1.260000000140374823e+03,
    1.470000000163770437e+03, 1.680000000187166279e+03, 1.890000000210562121e+03, 2.100000000233957849e+02,
    4.200000000467915697e+02, 6.300000000701872978e+02, 8.400000000935831395e+02, 1.050000000116978981e+03,
    1.260000000140374823e+03, 1.470000000163770437e+03, 1.680000000187166279e+03, 1.890000000210562121e+03,
    2.100000000233957849e+02, 4.200000000467915697e+02, 6.300000000701872978e+02, 8.400000000935831395e+02,
    1.050000000116978981e+03, 1.260000000140374823e+03, 1.470000000163770437e+03, 1.680000000187166279e+03,
    1.890000000210562121e+03, 2.100000000233957849e+02, 4.200000000467915697e+02, 6.300000000701872978e+02,
    8.400000000935831395e+02, 1.050000000116978981e+03, 1.260000000140374823e+03, 1.470000000163770437e+03,
    1.680000000187166279e+03, 1.890000000210562121e+03])

    turbineY = np.array([2.100000000233957849e+02,    2.100000000233957849e+02,    2.100000000233957849e+02,
    2.100000000233957849e+02, 2.100000000233957849e+02, 2.100000000233957849e+02, 2.100000000233957849e+02,
    2.100000000233957849e+02, 2.100000000233957849e+02, 4.200000000467915697e+02, 4.200000000467915697e+02,
    4.200000000467915697e+02, 4.200000000467915697e+02, 4.200000000467915697e+02, 4.200000000467915697e+02,
    4.200000000467915697e+02, 4.200000000467915697e+02, 4.200000000467915697e+02, 6.300000000701872978e+02,
    6.300000000701872978e+02, 6.300000000701872978e+02, 6.300000000701872978e+02, 6.300000000701872978e+02,
    6.300000000701872978e+02, 6.300000000701872978e+02, 6.300000000701872978e+02, 6.300000000701872978e+02,
    8.400000000935831395e+02, 8.400000000935831395e+02, 8.400000000935831395e+02, 8.400000000935831395e+02,
    8.400000000935831395e+02, 8.400000000935831395e+02, 8.400000000935831395e+02, 8.400000000935831395e+02,
    8.400000000935831395e+02, 1.050000000116978981e+03, 1.050000000116978981e+03, 1.050000000116978981e+03,
    1.050000000116978981e+03, 1.050000000116978981e+03, 1.050000000116978981e+03, 1.050000000116978981e+03,
    1.050000000116978981e+03, 1.050000000116978981e+03, 1.260000000140374823e+03, 1.260000000140374823e+03,
    1.260000000140374823e+03, 1.260000000140374823e+03, 1.260000000140374823e+03, 1.260000000140374823e+03,
    1.260000000140374823e+03, 1.260000000140374823e+03, 1.260000000140374823e+03, 1.470000000163770437e+03,
    1.470000000163770437e+03, 1.470000000163770437e+03, 1.470000000163770437e+03, 1.470000000163770437e+03,
    1.470000000163770437e+03, 1.470000000163770437e+03, 1.470000000163770437e+03, 1.470000000163770437e+03,
    1.680000000187166279e+03, 1.680000000187166279e+03, 1.680000000187166279e+03, 1.680000000187166279e+03,
    1.680000000187166279e+03, 1.680000000187166279e+03, 1.680000000187166279e+03, 1.680000000187166279e+03,
    1.680000000187166279e+03, 1.890000000210562121e+03, 1.890000000210562121e+03, 1.890000000210562121e+03,
    1.890000000210562121e+03, 1.890000000210562121e+03, 1.890000000210562121e+03, 1.890000000210562121e+03,
    1.890000000210562121e+03, 1.890000000210562121e+03])


    """Define wind flow"""
    air_density = 1.1716    # kg/m^3

    x = {}
    nDirections = amaliaWind(x)

    rotorDiameter = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generatorEfficiency = np.zeros(nTurbs)
    yaw = np.zeros((nDirections, nTurbs))

    # define initial values
    for turbI in range(0, nTurbs):
        rotorDiameter[turbI] = rotor_diameter
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        Cp[turbI] = 0.7737 * 4.0 * 1.0/3.0 * np.power((1. - 1.0/3.0), 2)
        generatorEfficiency[turbI] = 1.0



    """Define tower structural properties"""
    # --- geometry ---
    n = 15

    """same as for NREL 5MW"""
    L_reinforced = 30.0*np.ones(n)  # [m] buckling length
    Toweryaw = 0.0

    # --- material props ---
    E = 210.e9*np.ones(n)
    G = 80.8e9*np.ones(n)
    rho = 8500.0*np.ones(n)
    sigma_y = 450.0e6*np.ones(n)

    # --- spring reaction data.  Use float('inf') for rigid constraints. ---
    kidx = np.array([0], dtype=int)  # applied at base
    kx = np.array([float('inf')])
    ky = np.array([float('inf')])
    kz = np.array([float('inf')])
    ktx = np.array([float('inf')])
    kty = np.array([float('inf')])
    ktz = np.array([float('inf')])
    nK = len(kidx)

    """scale with rotor diameter"""
    # --- extra mass ----
    midx = np.array([n-1], dtype=int)  # RNA mass at top
    # m = np.array([285598.8])*(rotor_diameter/126.4)**3
    m = np.array([78055.827])
    mIxx = np.array([3.5622774377E+006])
    mIyy = np.array([1.9539222007E+006])
    mIzz = np.array([1.821096074E+006])
    mIxy = np.array([0.00000000e+00])
    mIxz = np.array([1.1141296293E+004])
    mIyz = np.array([0.00000000e+00])
    # mrhox = np.array([-1.13197635]) # Does not change with rotor_diameter
    mrhox = np.array([-0.1449])
    mrhoy = np.array([0.])
    mrhoz = np.array([1.389])
    nMass = len(midx)
    addGravityLoadForExtraMass = True
    # -----------

    # --- wind ---
    wind_zref = 50.0
    wind_z0 = 0.0
    # ---------------

    # # --- loading case 1: max Thrust ---
    wind_Uref1 = 11.73732
    plidx1 = np.array([n-1], dtype=int)  # at  top
    Fx1 = np.array([283000.])
    Fy1 = np.array([0.])
    Fz1 = np.array([-765727.66])
    Mxx1 = np.array([1513000.])
    Myy1 = np.array([-1360000.])
    Mzz1 = np.array([-127400.])
    nPL = len(plidx1)
    # # ---------------

    # # --- loading case 2: max wind speed ---
    wind_Uref2 = 70.0
    plidx2 = np.array([n-1], dtype=int)  # at  top
    Fx2 = np.array([204901.5477])
    Fy2 = np.array([0.])
    Fz2 = np.array([-832427.12368949])
    Mxx2 = np.array([-642674.9329])
    Myy2 = np.array([-1507872])
    Mzz2 = np.array([54115.])
    # # ---------------

    # --- safety factors ---
    gamma_f = 1.35
    gamma_m = 1.3
    gamma_n = 1.0
    gamma_b = 1.1
    # ---------------

    # --- constraints ---
    min_d_to_t = 120.0
    min_taper = 0.4
    # ---------------

    nPoints = 3
    nFull = n
    rhoAir = air_density

    d_param = np.array([4.031876248202890700e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    t_param = np.array([1.373755823634198632e-02, 8.961345551353992744e-03, 6.069695887100739172e-03])

    turbineZ = np.array([6.556977525499463866e+01])
    turbineZ = np.array([90.])

    """OpenMDAO"""

    prob = Problem()
    root = prob.root = Group()

    for i in range(nGroups):
        root.add('d_param%s'%i, IndepVarComp('d_param%s'%i, d_param), promotes=['*'])
        root.add('t_param%s'%i, IndepVarComp('t_param%s'%i, t_param), promotes=['*'])
        root.add('get_z_param%s'%i, get_z(nPoints))
        root.add('get_z_full%s'%i, get_z(nFull))
        root.add('Tower%s_max_thrust'%i, Tower(nPoints, nFull), promotes=['L_reinforced','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
        root.add('Tower%s_max_speed'%i, Tower(nPoints, nFull), promotes=['L_reinforced','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])

        root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(turbineZ[i])), promotes=['*'])

    root.add('Zs', DeMUX(nTurbs))
    root.add('hGroups', hGroups(nTurbs,nGroups), promotes=['*'])
    root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
                use_rotor_components=use_rotor_components, datasize=datasize, differentiable=True,
                optimizingLayout=False, nSamples=0), promotes=['*'])
    root.add('COEGroup', COEGroup(nTurbs, nGroups), promotes=['*'])

    root.connect('turbineZ', 'Zs.Array')
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

        root.connect('Tower%s_max_thrust.mass'%i, 'mass%s'%i)

        root.connect('d_param%s'%i, 'Tower%s_max_thrust.d_param'%i)
        root.connect('t_param%s'%i, 'Tower%s_max_thrust.t_param'%i)
        root.connect('d_param%s'%i, 'Tower%s_max_speed.d_param'%i)
        root.connect('t_param%s'%i, 'Tower%s_max_speed.t_param'%i)

        root.connect('Tower%s_max_speed.Mt'%i, 'Tower%s_max_speed.Mt'%i)
        root.connect('Tower%s_max_speed.It'%i, 'Tower%s_max_speed.It'%i)

        # root.connect('Tower%s_max_speed.m'%i, 'Tower%s_max_thrust.m'%i)


    # ----------------------

    prob.root.ln_solver.options['single_voi_relevance_reduction'] = True

    prob.setup(check=True)

    for i in range(nDirections):
        prob['yaw%s'%i] = yaw[i]
    # prob['nGroups'] = nGroups
    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY

    prob['ratedPower'] = np.ones(nTurbs)*1543.209877 # in kw

    # assign values to constant inputs (not design variables)
    prob['rotorDiameter'] = rotorDiameter
    # prob['rotor_diameter'] = rotor_diameter
    prob['axialInduction'] = axialInduction
    prob['generatorEfficiency'] = generatorEfficiency
    prob['air_density'] = air_density
    nDirs = amaliaWind(prob)

    prob['Ct_in'] = Ct
    prob['Cp_in'] = Cp
    prob['floris_params:cos_spread'] = 1E12 # turns off cosine spread (just needs to be very large)

    prob['L_reinforced'] = L_reinforced
    prob['rho'] = rho
    prob['shearExp'] = shearExp
    prob['E'] = E
    prob['gamma_f'] = gamma_f
    prob['gamma_b'] = gamma_b
    prob['sigma_y'] = sigma_y
    # prob['m'] = m
    prob['mrhox'] = mrhox
    prob['zref'] = 50.
    prob['z0'] = 0.

    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY

    for i in range(nGroups):
        prob['Tower%s_max_thrust.Fy'%i] = Fy1
        prob['Tower%s_max_thrust.Fx'%i] = Fx1
        prob['Tower%s_max_thrust.Fz'%i] = Fz1
        prob['Tower%s_max_thrust.Mxx'%i] = Mxx1
        prob['Tower%s_max_thrust.Myy'%i] = Myy1
        prob['Tower%s_max_thrust.Vel'%i] = wind_Uref1
        # prob['Tower%s_max_thrust.Mt'%i] = m[0]
        prob['Tower%s_max_thrust.It'%i] = mIzz[0]

        prob['Tower%s_max_speed.Fy'%i] = Fy2
        prob['Tower%s_max_speed.Fx'%i] = Fx2
        prob['Tower%s_max_speed.Fz'%i] = Fz2
        prob['Tower%s_max_speed.Mxx'%i] = Mxx2
        prob['Tower%s_max_speed.Myy'%i] = Myy2
        prob['Tower%s_max_speed.Vel'%i] = wind_Uref2


    prob.run()

    print 'AEP: ', prob['AEP']
    print 'COE: ', prob['COE']
    print 'cost: ', prob['cost']
    print 'd: ', prob['rotorDiameter']
    # 7.973416978975259894e+01 2.293186460008625686e+08
    # 8.504107990423464969e+01 2.473283094171800613e+08
