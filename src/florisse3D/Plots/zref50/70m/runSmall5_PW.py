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
    3.499999999949981770e+02,    6.999999999899963541e+02,    1.049999999984994474e+03,    1.399999999979992708e+03,
    1.749999999974990942e+03,    2.099999999969988949e+03,    2.449999999964987182e+03,    2.799999999959985416e+03,
    3.149999999954983650e+03,    3.499999999949981770e+02,    6.999999999899963541e+02,    1.049999999984994474e+03,
    1.399999999979992708e+03,    1.749999999974990942e+03,    2.099999999969988949e+03,    2.449999999964987182e+03,
    2.799999999959985416e+03,    3.149999999954983650e+03,    3.499999999949981770e+02,    6.999999999899963541e+02,
    1.049999999984994474e+03,    1.399999999979992708e+03,    1.749999999974990942e+03,    2.099999999969988949e+03,
    2.449999999964987182e+03,    2.799999999959985416e+03,    3.149999999954983650e+03,    3.499999999949981770e+02,
    6.999999999899963541e+02,    1.049999999984994474e+03,    1.399999999979992708e+03,    1.749999999974990942e+03,
    2.099999999969988949e+03,    2.449999999964987182e+03,    2.799999999959985416e+03,    3.149999999954983650e+03,
    3.499999999949981770e+02,    6.999999999899963541e+02,    1.049999999984994474e+03,    1.399999999979992708e+03,
    1.749999999974990942e+03,    2.099999999969988949e+03,    2.449999999964987182e+03,    2.799999999959985416e+03,
    3.149999999954983650e+03,    3.499999999949981770e+02,    6.999999999899963541e+02,    1.049999999984994474e+03,
    1.399999999979992708e+03,    1.749999999974990942e+03,    2.099999999969988949e+03,    2.449999999964987182e+03,
    2.799999999959985416e+03,    3.149999999954983650e+03,    3.499999999949981770e+02,    6.999999999899963541e+02,
    1.049999999984994474e+03,    1.399999999979992708e+03,    1.749999999974990942e+03,    2.099999999969988949e+03,
    2.449999999964987182e+03,    2.799999999959985416e+03,    3.149999999954983650e+03,    3.499999999949981770e+02,
    6.999999999899963541e+02,    1.049999999984994474e+03,    1.399999999979992708e+03,    1.749999999974990942e+03,
    2.099999999969988949e+03,    2.449999999964987182e+03,    2.799999999959985416e+03,    3.149999999954983650e+03,
    3.499999999949981770e+02,    6.999999999899963541e+02,    1.049999999984994474e+03,    1.399999999979992708e+03,
    1.749999999974990942e+03,    2.099999999969988949e+03,    2.449999999964987182e+03,    2.799999999959985416e+03,
    3.149999999954983650e+03        ])

    turbineY = np.array([
    3.499999999949981770e+02,    3.499999999949981770e+02,    3.499999999949981770e+02,    3.499999999949981770e+02,
    3.499999999949981770e+02,    3.499999999949981770e+02,    3.499999999949981770e+02,    3.499999999949981770e+02,
    3.499999999949981770e+02,    6.999999999899963541e+02,    6.999999999899963541e+02,    6.999999999899963541e+02,
    6.999999999899963541e+02,    6.999999999899963541e+02,    6.999999999899963541e+02,    6.999999999899963541e+02,
    6.999999999899963541e+02,    6.999999999899963541e+02,    1.049999999984994474e+03,    1.049999999984994474e+03,
    1.049999999984994474e+03,    1.049999999984994474e+03,    1.049999999984994474e+03,    1.049999999984994474e+03,
    1.049999999984994474e+03,    1.049999999984994474e+03,    1.049999999984994474e+03,    1.399999999979992708e+03,
    1.399999999979992708e+03,    1.399999999979992708e+03,    1.399999999979992708e+03,    1.399999999979992708e+03,
    1.399999999979992708e+03,    1.399999999979992708e+03,    1.399999999979992708e+03,    1.399999999979992708e+03,
    1.749999999974990942e+03,    1.749999999974990942e+03,    1.749999999974990942e+03,    1.749999999974990942e+03,
    1.749999999974990942e+03,    1.749999999974990942e+03,    1.749999999974990942e+03,    1.749999999974990942e+03,
    1.749999999974990942e+03,    2.099999999969988949e+03,    2.099999999969988949e+03,    2.099999999969988949e+03,
    2.099999999969988949e+03,    2.099999999969988949e+03,    2.099999999969988949e+03,    2.099999999969988949e+03,
    2.099999999969988949e+03,    2.099999999969988949e+03,    2.449999999964987182e+03,    2.449999999964987182e+03,
    2.449999999964987182e+03,    2.449999999964987182e+03,    2.449999999964987182e+03,    2.449999999964987182e+03,
    2.449999999964987182e+03,    2.449999999964987182e+03,    2.449999999964987182e+03,    2.799999999959985416e+03,
    2.799999999959985416e+03,    2.799999999959985416e+03,    2.799999999959985416e+03,    2.799999999959985416e+03,
    2.799999999959985416e+03,    2.799999999959985416e+03,    2.799999999959985416e+03,    2.799999999959985416e+03,
    3.149999999954983650e+03,    3.149999999954983650e+03,    3.149999999954983650e+03,    3.149999999954983650e+03,
    3.149999999954983650e+03,    3.149999999954983650e+03,    3.149999999954983650e+03,    3.149999999954983650e+03,
    3.149999999954983650e+03    ])

    d = np.zeros((len(shearExp),3))
    t = np.zeros((len(shearExp),3))
    z = np.zeros((len(shearExp),1))

    d[0] =  np.array([6.299999999999999822e+00, 5.341895155562502495e+00, 3.870000000000000107e+00])
    d[1] =  np.array([6.299999999999999822e+00, 5.339787836784722685e+00, 3.870000000000000107e+00])
    d[2] =  np.array([6.299999999999999822e+00, 5.338890754583194287e+00, 3.870000000000000107e+00])
    d[3] =  np.array([6.299999999999999822e+00, 5.334749883267022597e+00, 3.870000000000000107e+00])
    d[4] =  np.array([6.299999999999999822e+00, 5.332195480925346232e+00, 3.870000000000000107e+00])
    d[5] =  np.array([6.299999999999999822e+00, 5.329681382449954263e+00, 3.870000000000000107e+00])
    d[6] =  np.array([6.299999999999999822e+00, 5.326919740601524289e+00, 3.870000000000000107e+00])
    d[7] =  np.array([6.299999999999999822e+00, 5.324285383242640002e+00, 3.870000000000000107e+00])
    d[8] =  np.array([6.299999999999999822e+00, 5.321610193954304791e+00, 3.870000000000000107e+00])
    d[9] =  np.array([6.299999999999999822e+00, 5.318690013164797392e+00, 3.870000000000000107e+00])
    d[10] = np.array([6.299999999999999822e+00, 5.316098080668536063e+00, 3.870000000000000107e+00])
    d[11] = np.array([6.299999999999999822e+00, 5.313164314121850751e+00, 3.870000000000000107e+00])
    d[12] = np.array([6.299999999999999822e+00, 5.310294552519601474e+00, 3.870000000000000107e+00])
    d[13] = np.array([6.299999999999999822e+00, 5.307353136008732264e+00, 3.870000000000000107e+00])
    d[14] = np.array([6.299999999999999822e+00, 5.304668019912939236e+00, 3.870000000000000107e+00])
    d[15] = np.array([6.299999999999999822e+00, 5.301494802050810584e+00, 3.870000000000000107e+00])
    d[16] = np.array([6.299999999999999822e+00, 5.298260674534196468e+00, 3.870000000000000107e+00])
    d[17] = np.array([6.299999999999999822e+00, 5.295149282518880618e+00, 3.870000000000000107e+00])
    d[18] = np.array([6.299999999999999822e+00, 5.291989660374192717e+00, 3.870000000000000107e+00])
    d[19] = np.array([6.299999999999999822e+00, 5.288825590653108755e+00, 3.870000000000000107e+00])
    d[20] = np.array([6.299999999999999822e+00, 5.285550161293312676e+00, 3.870000000000000107e+00])
    d[21] = np.array([6.299999999999999822e+00, 5.282991606178153887e+00, 3.870000000000000107e+00])
    d[22] = np.array([6.299999999999999822e+00, 5.278866474957664146e+00, 3.870000000000000107e+00])

    t[0] =  np.array([1.501613963291067479e-02, 1.030831780081670895e-02, 6.100465433473761453e-03])
    t[1] =  np.array([1.501758741342090059e-02, 1.032402503520939673e-02, 6.105881179704137490e-03])
    t[2] =  np.array([1.499835115286708015e-02, 1.034703212035328543e-02, 6.115481279042334246e-03])
    t[3] =  np.array([1.502548150504170663e-02, 1.035223070963502134e-02, 6.122334878224533136e-03])
    t[4] =  np.array([1.502996083832851271e-02, 1.036637361137542415e-02, 6.130827762811852030e-03])
    t[5] =  np.array([1.503478524375035860e-02, 1.038054070982444115e-02, 6.136638822584727579e-03])
    t[6] =  np.array([1.504303635263222819e-02, 1.039238893523541409e-02, 6.146923779715912813e-03])
    t[7] =  np.array([1.504642367151516613e-02, 1.040826067128083378e-02, 6.155058865317008729e-03])
    t[8] =  np.array([1.505107159839063138e-02, 1.042359197027913770e-02, 6.162768090684235187e-03])
    t[9] =  np.array([1.505868454785724668e-02, 1.043683514644169925e-02, 6.171156953124401143e-03])
    t[10] = np.array([1.506361712727396530e-02, 1.045156479733608951e-02, 6.179881645665887858e-03])
    t[11] = np.array([1.507183415236567313e-02, 1.046549029654140345e-02, 6.188318658041635015e-03])
    t[12] = np.array([1.507712670095041028e-02, 1.048197594382282993e-02, 6.196387601790778916e-03])
    t[13] = np.array([1.508457410479365109e-02, 1.049661207831067283e-02, 6.204884733113051956e-03])
    t[14] = np.array([1.509082914383421542e-02, 1.051187546689388332e-02, 6.211537396902489655e-03])
    t[15] = np.array([1.509791531654431501e-02, 1.052700779286874083e-02, 6.223290419232176129e-03])
    t[16] = np.array([1.510745853101003004e-02, 1.054178851755398737e-02, 6.231824792397454467e-03])
    t[17] = np.array([1.511837587645554343e-02, 1.055519403315930381e-02, 6.238573793096581326e-03])
    t[18] = np.array([1.512506509719355532e-02, 1.057195463071835054e-02, 6.250104800503537839e-03])
    t[19] = np.array([1.513365412579556826e-02, 1.058740111317874565e-02, 6.259582834585252983e-03])
    t[20] = np.array([1.514331837365352068e-02, 1.060263297856673531e-02, 6.268636307778951804e-03])
    t[21] = np.array([1.514766427662307853e-02, 1.061522583837083179e-02, 6.278102593219969489e-03])
    t[22] = np.array([1.516534612296020768e-02, 1.063145436023036489e-02, 6.288557519721663450e-03])

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
    z[10] = np.array([9.000000000000000000e+01])
    z[11] = np.array([9.000000000000000000e+01])
    z[12] = np.array([9.000000000000000000e+01])
    z[13] = np.array([9.000000000000000000e+01])
    z[14] = np.array([9.000000000000000000e+01])
    z[15] = np.array([9.000000000000000000e+01])
    z[16] = np.array([9.000000000000000000e+01])
    z[17] = np.array([9.000000000000000000e+01])
    z[18] = np.array([9.000000000000000000e+01])
    z[19] = np.array([9.000000000000000000e+01])
    z[20] = np.array([9.000000000000000000e+01])
    z[21] = np.array([9.000000000000000000e+01])
    z[22] = np.array([9.000000000000000000e+01])

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

            root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(z[k][i])), promotes=['*'])

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

            root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(z[k][i])), promotes=['*'])

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
    # AEP:  array([  3.32025425e+08,   3.37932154e+08,   3.43943963e+08,
    #          3.50062723e+08,   3.56290335e+08,   3.62628736e+08,
    #          3.69079898e+08,   3.75645825e+08,   3.82328561e+08,
    #          3.89106507e+08,   3.95907921e+08,   4.02830332e+08,
    #          4.09875892e+08,   4.17030079e+08,   4.24221821e+08,
    #          4.31541504e+08,   4.38948883e+08,   4.46369475e+08,
    #          4.53841695e+08,   4.61374767e+08,   4.68974588e+08,
    #          4.76554957e+08,   4.84106713e+08])
    # COE:  array([ 64.90621563,  63.88779711,  62.88681251,  61.90375452,
    #         60.9376487 ,  59.98778241,  59.0549426 ,  58.13859791,
    #         57.23805845,  56.35581862,  55.50126514,  54.66112035,
    #         53.83546625,  53.02518461,  52.23799259,  51.46431951,
    #         50.70733359,  49.97364729,  49.260374  ,  48.56409328,
    #         47.88422793,  47.22591018,  46.59431738])
    # cost:  array([  2.15505138e+10,   2.15897409e+10,   2.16295395e+10,
    #          2.16701969e+10,   2.17114953e+10,   2.17532937e+10,
    #          2.17959922e+10,   2.18395216e+10,   2.18837445e+10,
    #          2.19284158e+10,   2.19733905e+10,   2.20191572e+10,
    #          2.20658597e+10,   2.21130969e+10,   2.21604963e+10,
    #          2.22089898e+10,   2.22579274e+10,   2.23067107e+10,
    #          2.23564116e+10,   2.24062472e+10,   2.24564860e+10,
    #          2.25057416e+10,   2.25566218e+10])
    # tower cost:  array([ 33623302.70392097,  33646203.34831154,  33668589.78934412,
    #         33692474.2789682 ,  33716183.53672943,  33738601.59064662,
    #         33762560.133247  ,  33787465.9042061 ,  33812224.16363604,
    #         33836008.52188704,  33860961.72435628,  33886299.78171607,
    #         33912976.08816385,  33938710.40879235,  33963953.91642834,
    #         33991477.81607558,  34018332.21902361,  34043470.33565913,
    #         34073004.96871205,  34100808.47193441,  34128578.17181974,
    #         34150037.28091785,  34184633.58843114])
    # wake loss:  array([ 19.11001226,  19.11001226,  19.11001226,  19.11001226,
    #         19.11001226,  19.11001226,  19.11001226,  19.11001226,
    #         19.11001226,  19.09596267,  19.0254354 ,  18.95597898,
    #         18.88757977,  18.81491992,  18.71528872,  18.61705614,
    #         18.49745208,  18.36508741,  18.23567765,  18.08868585,
    #         17.93615896,  17.72518725,  17.45360349])
