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
    shearExp = np.linspace(0.08,0.3,23)
    nGroups = 1

    use_rotor_components = False

    datasize = 0
    rotor_diameter = 70.0

    nRows = 9
    nTurbs = nRows**2

    turbineX = np.array([
    2.100000000233957849e+02,    4.200000000467915697e+02,    6.300000000701872978e+02,    8.400000000935831395e+02,
    1.050000000116978981e+03,    1.260000000140374823e+03,    1.470000000163770437e+03,    1.680000000187166279e+03,
    1.890000000210562121e+03,    2.100000000233957849e+02,    4.200000000467915697e+02,    6.300000000701872978e+02,
    8.400000000935831395e+02,    1.050000000116978981e+03,    1.260000000140374823e+03,    1.470000000163770437e+03,
    1.680000000187166279e+03,    1.890000000210562121e+03,    2.100000000233957849e+02,    4.200000000467915697e+02,
    6.300000000701872978e+02,    8.400000000935831395e+02,    1.050000000116978981e+03,    1.260000000140374823e+03,
    1.470000000163770437e+03,    1.680000000187166279e+03,    1.890000000210562121e+03,    2.100000000233957849e+02,
    4.200000000467915697e+02,    6.300000000701872978e+02,    8.400000000935831395e+02,    1.050000000116978981e+03,
    1.260000000140374823e+03,    1.470000000163770437e+03,    1.680000000187166279e+03,    1.890000000210562121e+03,
    2.100000000233957849e+02,    4.200000000467915697e+02,    6.300000000701872978e+02,    8.400000000935831395e+02,
    1.050000000116978981e+03,    1.260000000140374823e+03,    1.470000000163770437e+03,    1.680000000187166279e+03,
    1.890000000210562121e+03,    2.100000000233957849e+02,    4.200000000467915697e+02,    6.300000000701872978e+02,
    8.400000000935831395e+02,    1.050000000116978981e+03,    1.260000000140374823e+03,    1.470000000163770437e+03,
    1.680000000187166279e+03,    1.890000000210562121e+03,    2.100000000233957849e+02,    4.200000000467915697e+02,
    6.300000000701872978e+02,    8.400000000935831395e+02,    1.050000000116978981e+03,    1.260000000140374823e+03,
    1.470000000163770437e+03,    1.680000000187166279e+03,    1.890000000210562121e+03,    2.100000000233957849e+02,
    4.200000000467915697e+02,    6.300000000701872978e+02,    8.400000000935831395e+02,    1.050000000116978981e+03,
    1.260000000140374823e+03,    1.470000000163770437e+03,    1.680000000187166279e+03,    1.890000000210562121e+03,
    2.100000000233957849e+02,    4.200000000467915697e+02,    6.300000000701872978e+02,    8.400000000935831395e+02,
    1.050000000116978981e+03,    1.260000000140374823e+03,    1.470000000163770437e+03,    1.680000000187166279e+03,
    1.890000000210562121e+03    ])

    turbineY = np.array([
    2.100000000233957849e+02,    2.100000000233957849e+02,    2.100000000233957849e+02,    2.100000000233957849e+02,
    2.100000000233957849e+02,    2.100000000233957849e+02,    2.100000000233957849e+02,    2.100000000233957849e+02,
    2.100000000233957849e+02,    4.200000000467915697e+02,    4.200000000467915697e+02,    4.200000000467915697e+02,
    4.200000000467915697e+02,    4.200000000467915697e+02,    4.200000000467915697e+02,    4.200000000467915697e+02,
    4.200000000467915697e+02,    4.200000000467915697e+02,    6.300000000701872978e+02,    6.300000000701872978e+02,
    6.300000000701872978e+02,    6.300000000701872978e+02,    6.300000000701872978e+02,    6.300000000701872978e+02,
    6.300000000701872978e+02,    6.300000000701872978e+02,    6.300000000701872978e+02,    8.400000000935831395e+02,
    8.400000000935831395e+02,    8.400000000935831395e+02,    8.400000000935831395e+02,    8.400000000935831395e+02,
    8.400000000935831395e+02,    8.400000000935831395e+02,    8.400000000935831395e+02,    8.400000000935831395e+02,
    1.050000000116978981e+03,    1.050000000116978981e+03,    1.050000000116978981e+03,    1.050000000116978981e+03,
    1.050000000116978981e+03,    1.050000000116978981e+03,    1.050000000116978981e+03,    1.050000000116978981e+03,
    1.050000000116978981e+03,    1.260000000140374823e+03,    1.260000000140374823e+03,    1.260000000140374823e+03,
    1.260000000140374823e+03,    1.260000000140374823e+03,    1.260000000140374823e+03,    1.260000000140374823e+03,
    1.260000000140374823e+03,    1.260000000140374823e+03,    1.470000000163770437e+03,    1.470000000163770437e+03,
    1.470000000163770437e+03,    1.470000000163770437e+03,    1.470000000163770437e+03,    1.470000000163770437e+03,
    1.470000000163770437e+03,    1.470000000163770437e+03,    1.470000000163770437e+03,    1.680000000187166279e+03,
    1.680000000187166279e+03,    1.680000000187166279e+03,    1.680000000187166279e+03,    1.680000000187166279e+03,
    1.680000000187166279e+03,    1.680000000187166279e+03,    1.680000000187166279e+03,    1.680000000187166279e+03,
    1.890000000210562121e+03,    1.890000000210562121e+03,    1.890000000210562121e+03,    1.890000000210562121e+03,
    1.890000000210562121e+03,    1.890000000210562121e+03,    1.890000000210562121e+03,    1.890000000210562121e+03,
    1.890000000210562121e+03    ])

    d = np.zeros((len(shearExp),3))
    t = np.zeros((len(shearExp),3))

    d[0] = np.array([6.299999999999999822e+00, 5.342121931724036266e+00, 3.870000000000000107e+00])
    t[0] = np.array([1.501388158675442085e-02, 1.031022931106565302e-02, 6.098452367279039513e-03])
    d[1] = np.array([6.299999999999999822e+00, 5.339687614165924856e+00, 3.870000000000000107e+00])
    t[1] = np.array([1.501753430476570998e-02, 1.032406601010004560e-02, 6.106563648114222628e-03])
    d[2] = np.array([6.299999999999999822e+00, 5.337314241495159983e+00, 3.870000000000000107e+00])
    t[2] = np.array([1.502054139993937859e-02, 1.033828369687197699e-02, 6.114321443934339628e-03])
    d[3] = np.array([6.299999999999999822e+00, 5.334649736316916524e+00, 3.870000000000000107e+00])
    t[3] = np.array([1.502598834985854372e-02, 1.035181005873910334e-02, 6.123051568597977744e-03])
    d[4] = np.array([6.299999999999999822e+00, 5.332175060703528757e+00, 3.870000000000000107e+00])
    t[4] = np.array([1.503113847235761080e-02, 1.036543033336890379e-02, 6.131052948565596412e-03])
    d[5] = np.array([6.299999999999999822e+00, 5.329585451424619258e+00, 3.870000000000000107e+00])
    t[5] = np.array([1.503616831364008345e-02, 1.037880579213289189e-02, 6.138782197068153208e-03])
    d[6] = np.array([6.299999999999999822e+00, 5.326949788124615282e+00, 3.870000000000000107e+00])
    t[6] = np.array([1.504183633590758934e-02, 1.039324982796479072e-02, 6.146987214526025874e-03])
    d[7] = np.array([6.299999999999999822e+00, 5.324241257146105966e+00, 3.870000000000000107e+00])
    t[7] = np.array([1.504638612335020172e-02, 1.040833342264003113e-02, 6.154408584208526274e-03])
    d[8] = np.array([6.299999999999999822e+00, 5.321575890554625765e+00, 3.870000000000000107e+00])
    t[8] = np.array([1.505153660470438456e-02, 1.042320427457093558e-02, 6.161763532152475520e-03])
    d[9] = np.array([6.299999999999999822e+00, 5.318903935245247183e+00, 3.870000000000000107e+00])
    t[9] = np.array([1.505680552557287118e-02, 1.043817223693072928e-02, 6.169709901726469319e-03])
    d[10] = np.array([6.299999999999999822e+00, 5.316092938923187461e+00, 3.870000000000000107e+00])
    t[10] = np.array([1.506349883936317427e-02, 1.045261363747443037e-02, 6.177770510382353530e-03])
    d[11] = np.array([6.299999999999999822e+00, 5.313249272794873690e+00, 3.870000000000000107e+00])
    t[11] = np.array([1.507025881854084232e-02, 1.046699852397399433e-02, 6.187844519797544920e-03])
    d[12] = np.array([6.299999999999999822e+00, 5.311111519560084204e+00, 3.870000000000000107e+00])
    t[12] = np.array([1.507052721458951826e-02, 1.047998095948627501e-02, 6.196845333561617000e-03])
    d[13] = np.array([6.299999999999999822e+00, 5.307336839925420158e+00, 3.870000000000000107e+00])
    t[13] = np.array([1.508492502223917481e-02, 1.049615233812953817e-02, 6.205614684343691012e-03])
    d[14] = np.array([6.299999999999999822e+00, 5.304478696321064390e+00, 3.870000000000000107e+00])
    t[14] = np.array([1.509106772155721470e-02, 1.051177542372303507e-02, 6.212735062136916751e-03])
    d[15] = np.array([6.299999999999999822e+00, 5.301369495588491354e+00, 3.870000000000000107e+00])
    t[15] = np.array([1.509980669774867862e-02, 1.052657243944187318e-02, 6.221748480188211278e-03])
    d[16] = np.array([6.299999999999999822e+00, 5.298264194987894804e+00, 3.870000000000000107e+00])
    t[16] = np.array([1.510806468512575045e-02, 1.054152379777182486e-02, 6.231713030094479695e-03])
    d[17] = np.array([6.299999999999999822e+00, 5.295158409322115567e+00, 3.870000000000000107e+00])
    t[17] = np.array([1.511608728312002770e-02, 1.055682953897321003e-02, 6.241391259346217024e-03])
    d[18] = np.array([6.299999999999999822e+00, 5.291564468173239888e+00, 3.870000000000000107e+00])
    t[18] = np.array([1.513242885258400845e-02, 1.056518049963327960e-02, 6.249355694100919810e-03])
    d[19] = np.array([6.299999999999999822e+00, 5.288788068847224189e+00, 3.870000000000000107e+00])
    t[19] = np.array([1.513405236525417093e-02, 1.058725031790321361e-02, 6.259415612092483414e-03])
    d[20] = np.array([6.299999999999999822e+00, 5.285780154134824826e+00, 3.870000000000000107e+00])
    t[20] = np.array([1.514060278297159334e-02, 1.060352261303974634e-02, 6.269358035331918863e-03])
    d[21] = np.array([6.299999999999999822e+00, 5.282332955876336555e+00, 3.870000000000000107e+00])
    t[21] = np.array([1.515221275978153974e-02, 1.061816652686623300e-02, 6.277780186072762399e-03])
    d[22] = np.array([6.299999999999999822e+00, 5.278927103306517488e+00, 3.870000000000000107e+00])
    t[22] = np.array([1.516267226856699880e-02, 1.063368006751895349e-02, 6.288068331214980745e-03])




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

    turbineZ = np.array([90.])

    """OpenMDAO"""

    COE = np.zeros(23)
    AEP = np.zeros(23)
    idealAEP = np.zeros(23)
    cost = np.zeros(23)
    tower_cost = np.zeros(23)

    for k in range(23):
        prob = Problem()
        root = prob.root = Group()

        for i in range(nGroups):
            root.add('d_param%s'%i, IndepVarComp('d_param%s'%i, d[k]), promotes=['*'])
            root.add('t_param%s'%i, IndepVarComp('t_param%s'%i, t[k]), promotes=['*'])
            root.add('get_z_param%s'%i, get_z(nPoints))
            root.add('get_z_full%s'%i, get_z(nFull))
            root.add('Tower%s_max_thrust'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
            root.add('Tower%s_max_speed'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])

            root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(turbineZ[i])), promotes=['*'])

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
        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY

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

        prob['shearExp'] = shearExp[k]
        prob.run()
        COE[k] = prob['COE']
        AEP[k] = prob['AEP']
        cost[k] = prob['farm_cost']
        tower_cost[k] = prob['tower_cost']



        prob = Problem()
        root = prob.root = Group()

        for i in range(nGroups):
            root.add('d_param%s'%i, IndepVarComp('d_param%s'%i, d[k]), promotes=['*'])
            root.add('t_param%s'%i, IndepVarComp('t_param%s'%i, t[k]), promotes=['*'])
            root.add('get_z_param%s'%i, get_z(nPoints))
            root.add('get_z_full%s'%i, get_z(nFull))
            root.add('Tower%s_max_thrust'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
            root.add('Tower%s_max_speed'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])

            root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(turbineZ[i])), promotes=['*'])

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

        prob['shearExp'] = shearExp[k]
        prob.run()
        idealAEP[k] = prob['AEP']*81.

    print 'ideal AEP: ', repr(idealAEP)
    print 'AEP: ', repr(AEP)
    print 'COE: ', repr(COE)
    print 'cost: ', repr(cost)
    print 'tower cost: ', repr(tower_cost)

    print 'wake loss: ', repr((idealAEP-AEP)/idealAEP*100.)

    # ideal AEP:  array([  4.10465416e+08,   4.17767591e+08,   4.25199673e+08,
    #          4.32763970e+08,   4.40462837e+08,   4.48298666e+08,
    #          4.56273895e+08,   4.64391003e+08,   4.72652514e+08,
    #          4.80948195e+08,   4.88928743e+08,   4.97051265e+08,
    #          5.05318286e+08,   5.13678226e+08,   5.21896202e+08,
    #          5.30260376e+08,   5.38570749e+08,   5.46787472e+08,
    #          5.55060792e+08,   5.63261342e+08,   5.71475307e+08,
    #          5.79223386e+08,   5.86466197e+08])
    # AEP:  array([  2.47328309e+08,   2.51728278e+08,   2.56206521e+08,
    #          2.60764432e+08,   2.65403429e+08,   2.70124953e+08,
    #          2.74930473e+08,   2.79821484e+08,   2.84799505e+08,
    #          2.89842411e+08,   2.94877917e+08,   3.00003005e+08,
    #          3.05219268e+08,   3.10511615e+08,   3.15808394e+08,
    #          3.21199404e+08,   3.26643798e+08,   3.32134365e+08,
    #          3.37710839e+08,   3.43302754e+08,   3.48966346e+08,
    #          3.54602247e+08,   3.60197381e+08])
    # COE:  array([ 85.0410799 ,  83.67377623,  82.33013751,  81.01019268,
    #         79.71309569,  78.4379296 ,  77.1860451 ,  75.95545808,
    #         74.74630745,  73.56388118,  72.42358968,  71.30264436,
    #         70.1973027 ,  69.11906154,  68.07330913,  67.04478708,
    #         66.04059963,  65.06119631,  64.09640556,  63.16527345,
    #         62.25029853,  61.36869011,  60.5214502 ])
    # cost:  array([  2.10330665e+10,   2.10630556e+10,   2.10935181e+10,
    #          2.11245769e+10,   2.11561289e+10,   2.11880421e+10,
    #          2.12207959e+10,   2.12539690e+10,   2.12877114e+10,
    #          2.13219327e+10,   2.13561173e+10,   2.13910076e+10,
    #          2.14255694e+10,   2.14622714e+10,   2.14981225e+10,
    #          2.15347456e+10,   2.15717523e+10,   2.16090591e+10,
    #          2.16460509e+10,   2.16848123e+10,   2.17232592e+10,
    #          2.17614754e+10,   2.17996679e+10])
    # tower cost:  array([ 33623682.97330754,  33646352.1523455 ,  33668986.58628966,
    #         33692420.35947319,  33715837.93248764,  33738210.63789596,
    #         33762974.20537763,  33786984.82206847,  33811270.62580765,
    #         33836158.45814328,  33861108.45444167,  33887214.47070372,
    #         33906855.3459703 ,  33938727.26815102,  33964188.70571121,
    #         33991083.7832114 ,  34018397.29465535,  34045842.4089272 ,
    #         34067155.95865495,  34100699.76286625,  34128750.13412264,
    #         34156351.28385241,  34185597.63273675])
    # wake loss:  array([ 39.74442187,  39.74442187,  39.74442187,  39.74442187,
    #         39.74442187,  39.74442187,  39.74442187,  39.74442187,
    #         39.74442187,  39.7352119 ,  39.68897899,  39.64344805,
    #         39.59861015,  39.55133791,  39.488275  ,  39.42609737,
    #         39.34988143,  39.25713706,  39.1578645 ,  39.05089368,
    #         38.93588363,  38.77970824,  38.58173198])
