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
    nGroups = 1

    datasize = 0
    rotor_diameter = 70.0
    nRows = 9
    nTurbs = nRows**2

    use_rotor_components = False

    d = np.zeros((nRuns,3))
    t = np.zeros((nRuns,3))
    z = np.zeros((nRuns,1))

    d[0] =  np.array([6.299999999999999822e+00, 5.295217809318518043e+00, 3.870000000000000107e+00])
    d[1] =  np.array([6.299999999999999822e+00, 5.295089173463020771e+00, 3.870000000000000107e+00])
    d[2] =  np.array([6.299999999999999822e+00, 5.295169554367131326e+00, 3.870000000000000107e+00])
    d[3] =  np.array([6.299999999999999822e+00, 5.293767385245545576e+00, 3.870000000000000107e+00])
    d[4] =  np.array([6.299999999999999822e+00, 5.294586900787667361e+00, 3.870000000000000107e+00])
    d[5] =  np.array([6.299999999999999822e+00, 5.295487108420721789e+00, 3.870000000000000107e+00])
    d[6] =  np.array([6.299999999999999822e+00, 5.293704107885899290e+00, 3.870000000000000107e+00])
    d[7] =  np.array([6.299999999999999822e+00, 5.295121553837032202e+00, 3.870000000000000107e+00])
    d[8] =  np.array([6.299999999999999822e+00, 5.295484174612728800e+00, 3.870000000000000107e+00])
    d[9] =  np.array([6.299999999999999822e+00, 5.295239561611933432e+00, 3.870000000000000107e+00])

    t[0] =  np.array([1.511644074895303509e-02, 1.055548900646388275e-02, 6.241386808434917777e-03])
    t[1] =  np.array([1.511860356127574072e-02, 1.055495450855838298e-02, 6.241391470362112753e-03])
    t[2] =  np.array([1.511856909937251553e-02, 1.055504914675800369e-02, 6.239225932605108743e-03])
    t[3] =  np.array([1.511147702494722474e-02, 1.055864427017495950e-02, 6.236773659658829440e-03])
    t[4] =  np.array([1.512253527408145663e-02, 1.055202380347107505e-02, 6.241390625675483951e-03])
    t[5] =  np.array([1.511526866100026262e-02, 1.055543984638916688e-02, 6.238738565690116894e-03])
    t[6] =  np.array([1.513911661542032341e-02, 1.054635876324270041e-02, 6.241323957323380300e-03])
    t[7] =  np.array([1.512446548026971996e-02, 1.054826772349531407e-02, 6.237104525633948952e-03])
    t[8] =  np.array([1.511321265660389018e-02, 1.055770078153510372e-02, 6.238926282004964192e-03])
    t[9] =  np.array([1.511621297828936138e-02, 1.055671362596084448e-02, 6.239655524673419495e-03])

    z[0] =  np.array([9.000000000000000000e+01])
    z[1] =  np.array([9.000000000000000000e+01])
    z[2] =  np.array([9.000000000000000000e+01])
    z[3] =  np.array([9.000000000000000000e+01])
    z[4] =  np.array([9.000000000000000000e+01])
    z[5] =  np.array([9.000000000000000000e+01])
    z[6] =  np.array([9.000000000000000000e+01])
    z[7] =  np.array([9.000000000000000000e+01])
    z[8] =  np.array([9.000000000000000000e+01])
    z[9] =  np.array([9.000000000000000000e+01])

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

    for i in range(nGroups):
        root.add('d_param%s'%i, IndepVarComp('d_param%s'%i, d[0]), promotes=['*'])
        root.add('t_param%s'%i, IndepVarComp('t_param%s'%i, t[0]), promotes=['*'])
        root.add('get_z_param%s'%i, get_z(nPoints))
        root.add('get_z_full%s'%i, get_z(nFull))
        root.add('Tower%s_max_thrust'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
        root.add('Tower%s_max_speed'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])

        root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(z[0][i])), promotes=['*'])

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

        prob['d_param0'] = d[k]
        prob['t_param0'] = t[k]
        prob['turbineH0'] = z[k]

        prob.run()
        COE[k] = prob['COE']
        AEP[k] = prob['AEP']
        cost[k] = prob['farm_cost']
        tower_cost[k] = prob['tower_cost']



    prob = Problem()
    root = prob.root = Group()

    for i in range(nGroups):
        root.add('d_param%s'%i, IndepVarComp('d_param%s'%i, d[0]), promotes=['*'])
        root.add('t_param%s'%i, IndepVarComp('t_param%s'%i, t[0]), promotes=['*'])
        root.add('get_z_param%s'%i, get_z(nPoints))
        root.add('get_z_full%s'%i, get_z(nFull))
        root.add('Tower%s_max_thrust'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
        root.add('Tower%s_max_speed'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])

        root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(z[0][i])), promotes=['*'])

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

        prob['d_param0'] = d[k]
        prob['t_param0'] = t[k]
        prob['turbineH0'] = z[k]

        prob.run()
        idealAEP[k] = prob['AEP']*81.

    print 'ideal AEP: ', repr(idealAEP)
    print 'AEP: ', repr(AEP)
    print 'COE: ', repr(COE)
    print 'cost: ', repr(cost)
    print 'tower cost: ', repr(tower_cost)

    print 'wake loss: ', repr((idealAEP-AEP)/idealAEP*100.)

    # ideal AEP:  array([  5.46787472e+08,   5.46787472e+08,   5.46787472e+08,
    #          5.46787472e+08,   5.46787472e+08,   5.46787472e+08,
    #          5.46787472e+08,   5.46787472e+08,   5.46787472e+08,
    #          5.46787472e+08])
    # AEP:  array([  4.81620357e+08,   4.25717195e+08,   3.81884581e+08,
    #          3.46005733e+08,   3.16439608e+08,   2.91694854e+08,
    #          2.70586470e+08,   2.52483305e+08,   2.36798079e+08,
    #          2.23015768e+08])
    # COE:  array([ 46.76338996,  52.10202084,  57.38060267,  62.6944035 ,
    #         67.98350782,  73.2322968 ,  78.47004661,  83.65379161,
    #         88.79427065,  93.904643  ])
    # cost:  array([  2.25222006e+10,   2.21807262e+10,   2.19127674e+10,
    #          2.16926230e+10,   2.15126746e+10,   2.13614841e+10,
    #          2.12329329e+10,   2.11211858e+10,   2.10263127e+10,
    #          2.09422160e+10])
    # tower cost:  array([ 34044251.48583033,  34044937.02829164,  34043903.91558909,
    #         34037154.16248824,  34042268.40320115,  34042268.29379238,
    #         34045341.85062581,  34037043.20908744,  34044078.20149913,
    #         34044904.77379616])
    # wake loss:  array([ 11.91817988,  22.14210878,  30.15849848,  36.72025224,
    #         42.12749482,  46.6529742 ,  50.51341085,  53.82423368,
    #         56.69284844,  59.213446  ])
