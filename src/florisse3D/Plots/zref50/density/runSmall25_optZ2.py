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
    nRuns = 10

    shearExp = 0.25
    nGroups = 2

    datasize = 0
    rotor_diameter = 70.0
    nRows = 9
    nTurbs = nRows**2

    use_rotor_components = False

    d1 = np.zeros((nRuns,3))
    t1 = np.zeros((nRuns,3))
    z1 = np.zeros((nRuns,1))
    d2 = np.zeros((nRuns,3))
    t2 = np.zeros((nRuns,3))
    z2 = np.zeros((nRuns,1))

    d1[0] =  np.array([6.299999999999999822e+00, 6.184676384567724838e+00, 3.870000000000000107e+00])
    d1[1] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 5.735726424046208471e+00])
    d1[2] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])
    d1[3] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])
    d1[4] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])
    d1[5] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])
    d1[6] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])
    d1[7] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])
    d1[8] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])
    d1[9] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])

    d2[0] =  np.array([6.299999999999999822e+00, 5.040180970026144180e+00, 3.870000000000000107e+00])
    d2[1] =  np.array([4.847833296269847381e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[2] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[3] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[4] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[5] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[6] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[7] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[8] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[9] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])

    t1[0] =  np.array([1.617772365430871634e-02, 1.085506524200060934e-02, 6.268547474231726986e-03])
    t1[1] =  np.array([1.802626446222068898e-02, 1.173071353831037396e-02, 6.957054087611673834e-03])
    t1[2] =  np.array([1.862613436104007536e-02, 1.200781584719997144e-02, 7.218221175999375155e-03])
    t1[3] =  np.array([1.862613436104113701e-02, 1.200781584720518949e-02, 7.218221176022852903e-03])
    t1[4] =  np.array([1.862558534103880489e-02, 1.200769278387329650e-02, 7.212418528052758591e-03])
    t1[5] =  np.array([1.862613436104091497e-02, 1.200781584720510969e-02, 7.218221176022844229e-03])
    t1[6] =  np.array([1.862613436103977699e-02, 1.200781584720469856e-02, 7.218221176022799994e-03])
    t1[7] =  np.array([1.862613436104228193e-02, 1.200781584720560582e-02, 7.218221176022896271e-03])
    t1[8] =  np.array([1.862613436103256401e-02, 1.200781584718610753e-02, 7.218221175944954278e-03])
    t1[9] =  np.array([1.862613389247878837e-02, 1.200781500554697068e-02, 7.218217712083098078e-03])

    t2[0] =  np.array([1.482210960685052467e-02, 1.047923795730063348e-02, 6.233974614147450542e-03])
    t2[1] =  np.array([1.317898633531686473e-02, 9.424776805027092891e-03, 6.151631535288739835e-03])
    t2[2] =  np.array([1.336386316079053900e-02, 8.755160380721639657e-03, 6.118772863207429767e-03])
    t2[3] =  np.array([1.237845506845377044e-02, 8.388807945748661538e-03, 6.090448663593419840e-03])
    t2[4] =  np.array([1.187216008075166543e-02, 8.191193692618826266e-03, 6.075351253316227905e-03])
    t2[5] =  np.array([1.156180114792406191e-02, 8.068242703767417381e-03, 6.070871366302692425e-03])
    t2[6] =  np.array([1.134355013852701016e-02, 7.977514425271135801e-03, 6.061026880403687193e-03])
    t2[7] =  np.array([1.117993210468441745e-02, 7.909437488714018971e-03, 6.055971573530156797e-03])
    t2[8] =  np.array([1.106119253054421193e-02, 7.859521542196648211e-03, 6.051791325280306104e-03])
    t2[9] =  np.array([1.098121332743051681e-02, 7.825570735539253212e-03, 6.049788572945393773e-03])

    z1[0] =  np.array([9.627943335842522288e+01])
    z1[1] =  np.array([1.020146455076379510e+02])
    z1[2] =  np.array([1.038037920187946526e+02])
    z1[3] =  np.array([1.038037920187995269e+02])
    z1[4] =  np.array([1.038013134938301647e+02])
    z1[5] =  np.array([1.038037920187985037e+02])
    z1[6] =  np.array([1.038037920187932315e+02])
    z1[7] =  np.array([1.038037920188047849e+02])
    z1[8] =  np.array([1.038037920187602481e+02])
    z1[9] =  np.array([1.038037898750872046e+02])

    z2[0] =  np.array([8.823673440864403972e+01])
    z2[1] =  np.array([7.042217855309942820e+01])
    z2[2] =  np.array([6.314081867080955845e+01])
    z2[3] =  np.array([5.841887399550864757e+01])
    z2[4] =  np.array([5.584847700857219621e+01])
    z2[5] =  np.array([5.421993804839747355e+01])
    z2[6] =  np.array([5.304302573340196147e+01])
    z2[7] =  np.array([5.214759045762459522e+01])
    z2[8] =  np.array([5.148924570556527414e+01])
    z2[9] =  np.array([5.104163365117688755e+01])

    """Define wind flow"""
    air_density = 1.1716    # kg/m^3

    windData = "Amalia"

    """Amalia Wind Arrays"""
    if windData == "Amalia":
        windSpeeds, windFrequencies, windDirections, nDirections = amaliaWind()

    rotorDiameter = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generatorEfficiency = np.zeros(nTurbs)
    yaw = np.zeros((nDirections, nTurbs))

    # define initial values
    for turbI in range(0, nTurbs):
        rotorDiameter[turbI] = rotor_diameter            # m
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        Cp[turbI] = 0.7737 * 4.0 * 1.0/3.0 * np.power((1. - 1.0/3.0), 2)
        generatorEfficiency[turbI] = 1.0#0.944

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

    # if addGravityLoadForExtraMass=True be sure not to double count by adding those force here also
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

    """OpenMDAO"""

    COE = np.zeros(nRuns)
    AEP = np.zeros(nRuns)
    idealAEP = np.zeros(nRuns)
    cost = np.zeros(nRuns)
    tower_cost = np.zeros(nRuns)

    prob = Problem()
    root = prob.root = Group()

    root.add('d_param0', IndepVarComp('d_param0', d1[0]), promotes=['*'])
    root.add('t_param0', IndepVarComp('t_param0', t1[0]), promotes=['*'])
    root.add('turbineH0', IndepVarComp('turbineH0', float(z1[0])), promotes=['*'])
    root.add('d_param1', IndepVarComp('d_param1', d2[0]), promotes=['*'])
    root.add('t_param1', IndepVarComp('t_param1', t2[0]), promotes=['*'])
    root.add('turbineH1', IndepVarComp('turbineH1', float(z2[0])), promotes=['*'])

    for i in range(nGroups):
        root.add('get_z_param%s'%i, get_z(nPoints))
        root.add('get_z_full%s'%i, get_z(nFull))
        root.add('Tower%s_max_thrust'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
        root.add('Tower%s_max_speed'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])

    root.add('Zs', DeMUX(nTurbs))
    root.add('hGroups', hGroups(nTurbs), promotes=['*'])
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

    # ----------------------

    prob.root.ln_solver.options['single_voi_relevance_reduction'] = True

    prob.setup(check=True)

    for i in range(nDirections):
        prob['yaw%s'%i] = yaw[i]
    prob['nGroups'] = nGroups

    prob['ratedPower'] = np.ones(nTurbs)*1543.209877 # in kw

    # assign values to constant inputs (not design variables)
    prob['rotorDiameter'] = rotorDiameter
    prob['rotor_diameter'] = rotor_diameter
    prob['axialInduction'] = axialInduction
    prob['generatorEfficiency'] = generatorEfficiency
    prob['air_density'] = air_density
    prob['windDirections'] = windDirections
    prob['windFrequencies'] = windFrequencies
    prob['Uref'] = windSpeeds

    prob['Ct_in'] = Ct
    prob['Cp_in'] = Cp
    prob['floris_params:cos_spread'] = 1E12 # turns off cosine spread (just needs to be very large)

    prob['L_reinforced'] = L_reinforced
    prob['rho'] = rho
    prob['E'] = E
    prob['gamma_f'] = gamma_f
    prob['gamma_b'] = gamma_b
    prob['sigma_y'] = sigma_y
    prob['m'] = m
    prob['mrhox'] = mrhox
    prob['zref'] = 50.
    prob['z0'] = 0.

    for i in range(nGroups):
        prob['Tower%s_max_thrust.Fy'%i] = Fy1
        prob['Tower%s_max_thrust.Fx'%i] = Fx1
        prob['Tower%s_max_thrust.Fz'%i] = Fz1
        prob['Tower%s_max_thrust.Mxx'%i] = Mxx1
        prob['Tower%s_max_thrust.Myy'%i] = Myy1
        prob['Tower%s_max_thrust.Vel'%i] = wind_Uref1
        prob['Tower%s_max_thrust.Mt'%i] = m[0]
        prob['Tower%s_max_thrust.It'%i] = mIzz[0]

        prob['Tower%s_max_speed.Fy'%i] = Fy2
        prob['Tower%s_max_speed.Fx'%i] = Fx2
        prob['Tower%s_max_speed.Fz'%i] = Fz2
        prob['Tower%s_max_speed.Mxx'%i] = Mxx2
        prob['Tower%s_max_speed.Myy'%i] = Myy2
        prob['Tower%s_max_speed.Vel'%i] = wind_Uref2

    prob['shearExp'] = shearExp
    density = np.array([0.0248420225,0.049684045,0.0745260675,0.09936809,0.1242101126,\
                0.1490521351,0.1738941576,0.1987361801,0.2235782026,0.2484202251])
    for k in range(nRuns):

        spacing = nRows/(2.*nRows-2.)*np.sqrt(3.1415926535/density[k])
        points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
        xpoints, ypoints = np.meshgrid(points, points)
        turbineX = np.ndarray.flatten(xpoints)
        turbineY = np.ndarray.flatten(ypoints)

        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY

        prob['d_param0'] = d1[k]
        prob['t_param0'] = t1[k]
        prob['turbineH0'] = z1[k]
        prob['d_param1'] = d2[k]
        prob['t_param1'] = t2[k]
        prob['turbineH1'] = z2[k]

        prob.run()
        COE[k] = prob['COE']
        AEP[k] = prob['AEP']
        cost[k] = prob['farm_cost']
        tower_cost[k] = prob['tower_cost']



    nGroups = 1
    prob = Problem()
    root = prob.root = Group()

    for i in range(nGroups):
        root.add('d_param%s'%i, IndepVarComp('d_param%s'%i, d1[0]), promotes=['*'])
        root.add('t_param%s'%i, IndepVarComp('t_param%s'%i, t1[0]), promotes=['*'])
        root.add('get_z_param%s'%i, get_z(nPoints))
        root.add('get_z_full%s'%i, get_z(nFull))
        root.add('Tower%s_max_thrust'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
        root.add('Tower%s_max_speed'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])

        root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(z1[0][i])), promotes=['*'])

    root.add('Zs', DeMUX(1))
    root.add('hGroups', hGroups(1), promotes=['*'])
    root.add('AEPGroup', AEPGroup(1, nDirections=nDirections,
                use_rotor_components=use_rotor_components, datasize=datasize, differentiable=True,
                optimizingLayout=False, nSamples=0), promotes=['*'])
    root.add('COEGroup', COEGroup(1, nGroups), promotes=['*'])

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

    # ----------------------

    prob.root.ln_solver.options['single_voi_relevance_reduction'] = True

    prob.setup(check=True)

    for i in range(nDirections):
        prob['yaw%s'%i] = np.array([0.])
    prob['nGroups'] = nGroups
    prob['turbineX'] = np.array([0.])
    prob['turbineY'] = np.array([0.])

    prob['ratedPower'] = np.ones(1)*1543.209877 # in kw

    # assign values to constant inputs (not design variables)
    prob['rotorDiameter'] = np.ones(1)*rotor_diameter
    prob['rotor_diameter'] = rotor_diameter
    prob['axialInduction'] = axialInduction[0]
    prob['generatorEfficiency'] = generatorEfficiency[0]
    prob['air_density'] = air_density
    prob['windDirections'] = windDirections
    prob['windFrequencies'] = windFrequencies
    prob['Uref'] = windSpeeds

    prob['Ct_in'] = Ct[0]
    prob['Cp_in'] = Cp[0]
    prob['floris_params:cos_spread'] = 1E12 # turns off cosine spread (just needs to be very large)

    prob['L_reinforced'] = L_reinforced
    prob['rho'] = rho
    prob['E'] = E
    prob['gamma_f'] = gamma_f
    prob['gamma_b'] = gamma_b
    prob['sigma_y'] = sigma_y
    prob['m'] = m
    prob['mrhox'] = mrhox
    prob['zref'] = 50.
    prob['z0'] = 0.

    for i in range(nGroups):
        prob['Tower%s_max_thrust.Fy'%i] = Fy1
        prob['Tower%s_max_thrust.Fx'%i] = Fx1
        prob['Tower%s_max_thrust.Fz'%i] = Fz1
        prob['Tower%s_max_thrust.Mxx'%i] = Mxx1
        prob['Tower%s_max_thrust.Myy'%i] = Myy1
        prob['Tower%s_max_thrust.Vel'%i] = wind_Uref1
        prob['Tower%s_max_thrust.Mt'%i] = m[0]
        prob['Tower%s_max_thrust.It'%i] = mIzz[0]

        prob['Tower%s_max_speed.Fy'%i] = Fy2
        prob['Tower%s_max_speed.Fx'%i] = Fx2
        prob['Tower%s_max_speed.Fz'%i] = Fz2
        prob['Tower%s_max_speed.Mxx'%i] = Mxx2
        prob['Tower%s_max_speed.Myy'%i] = Myy2
        prob['Tower%s_max_speed.Vel'%i] = wind_Uref2


    prob['shearExp'] = shearExp

    for k in range(nRuns):

        prob['d_param0'] = d1[k]
        prob['t_param0'] = t1[k]
        prob['turbineH0'] = z1[k]
        prob.run()
        AEP1 = prob['AEP']

        prob['d_param0'] = d2[k]
        prob['t_param0'] = t2[k]
        prob['turbineH0'] = z2[k]
        prob.run()
        AEP2 = prob['AEP']

        idealAEP[k] = 41.*AEP1 + 40.*AEP2

    print 'ideal AEP: ', repr(idealAEP)
    print 'AEP: ', repr(AEP)
    print 'COE: ', repr(COE)
    print 'cost: ', repr(cost)
    print 'tower cost: ', repr(tower_cost)

    print 'wake loss: ', repr((idealAEP-AEP)/idealAEP*100.)

    # ideal AEP:  array([  5.55331246e+08,   5.25638440e+08,   5.10473138e+08,
    #          4.98598099e+08,   4.92029733e+08,   4.87835279e+08,
    #          4.84781681e+08,   4.82447047e+08,   4.80724174e+08,
    #          4.79549636e+08])
    # AEP:  array([  4.90914779e+08,   4.31845149e+08,   3.99681294e+08,
    #          3.72641498e+08,   3.51516572e+08,   3.33748691e+08,
    #          3.18229103e+08,   3.04499233e+08,   2.92249416e+08,
    #          2.81231958e+08])
    # COE:  array([ 46.79303038,  51.64926438,  55.19686882,  58.3757087 ,
    #         61.30484639,  64.10940694,  66.83637278,  69.49517347,
    #         72.09183695,  74.63459138])
    # cost:  array([  2.29713902e+10,   2.23044843e+10,   2.20611560e+10,
    #          2.17532116e+10,   2.15496694e+10,   2.13964306e+10,
    #          2.12692789e+10,   2.11612271e+10,   2.10687972e+10,
    #          2.09896323e+10])
    # tower cost:  array([ 36701004.52910343,  34990570.76757029,  34846126.35261185,
    #         33968516.21880025,  33514435.38904266,  33244880.65202157,
    #         33049852.14075491,  32904629.53458659,  32799182.26219534,
    #         32728267.69264033])
    # wake loss:  array([ 11.59964742,  17.84368936,  21.70375597,  25.26215023,
    #         28.55785986,  31.58578207,  34.35620301,  36.88442376,
    #         39.20642402,  41.35498448])
