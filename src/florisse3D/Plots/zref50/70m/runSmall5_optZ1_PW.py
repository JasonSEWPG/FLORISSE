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

    d[0] =  np.array([4.032562579607792408e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[1] =  np.array([4.309886002882426403e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[2] =  np.array([4.592540991504303172e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[3] =  np.array([5.250483952357802409e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[4] =  np.array([5.837388930556595135e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[5] =  np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[6] =  np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[7] =  np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[8] =  np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[9] =  np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[10] = np.array([6.299999999999999822e+00, 4.120416764295404377e+00, 3.870000000000000107e+00])
    d[11] = np.array([6.299999999999999822e+00, 4.418846559467444202e+00, 3.870000000000000107e+00])
    d[12] = np.array([6.299999999999999822e+00, 4.637131940300982080e+00, 3.870000000000000107e+00])
    d[13] = np.array([6.299999999999999822e+00, 4.940389194988579646e+00, 3.870000000000000107e+00])
    d[14] = np.array([6.299999999999999822e+00, 5.144628836024589980e+00, 3.870000000000000107e+00])
    d[15] = np.array([6.299999999999999822e+00, 5.445741169560309736e+00, 3.870000000000000107e+00])
    d[16] = np.array([6.299999999999999822e+00, 5.512424231731905877e+00, 3.870000000000000107e+00])
    d[17] = np.array([6.299999999999999822e+00, 5.721401615470874447e+00, 3.870000000000000107e+00])
    d[18] = np.array([6.299999999999999822e+00, 5.831090290052074465e+00, 3.870000000000000107e+00])
    d[19] = np.array([6.299999999999999822e+00, 5.892721521693336406e+00, 3.870000000000000107e+00])
    d[20] = np.array([6.299999999999999822e+00, 6.069672876124944771e+00, 3.870000000000000107e+00])
    d[21] = np.array([6.299999999999999822e+00, 6.159660249102692831e+00, 3.870000000000000107e+00])
    d[22] = np.array([6.299999999999999822e+00, 6.171960915167980843e+00, 3.870000000000000107e+00])

    t[0] =  np.array([1.373759483222702221e-02, 8.961559947102258616e-03, 6.069367646027652372e-03])
    t[1] =  np.array([1.344472997490329946e-02, 9.068076037614057269e-03, 6.075511065684970936e-03])
    t[2] =  np.array([1.334582049727347716e-02, 9.220005073145421878e-03, 6.081769324883753471e-03])
    t[3] =  np.array([1.326659588334828596e-02, 9.600934840531642755e-03, 6.094281829165017032e-03])
    t[4] =  np.array([1.331020226078902594e-02, 9.930808364549067319e-03, 6.106602660219269917e-03])
    t[5] =  np.array([1.340254936398655762e-02, 1.019003701451945063e-02, 6.118765841979175585e-03])
    t[6] =  np.array([1.339794255754722836e-02, 1.020196246009870064e-02, 6.125759577834109272e-03])
    t[7] =  np.array([1.339609086818089809e-02, 1.021293126107636340e-02, 6.130547748115677126e-03])
    t[8] =  np.array([1.338485962395300974e-02, 1.022784535381122267e-02, 6.139232100981177088e-03])
    t[9] =  np.array([1.351300285261601217e-02, 1.019887959898137242e-02, 6.144731394230086574e-03])
    t[10] = np.array([1.387740828918368127e-02, 1.017935992065219751e-02, 6.157111379463928863e-03])
    t[11] = np.array([1.424258708663110259e-02, 1.020466992922534726e-02, 6.169876265399778896e-03])
    t[12] = np.array([1.435360592207234574e-02, 1.029722283373502197e-02, 6.182028931814967203e-03])
    t[13] = np.array([1.467310630955642910e-02, 1.039250088045878123e-02, 6.196817498534640950e-03])
    t[14] = np.array([1.491162118529222244e-02, 1.046424384326150037e-02, 6.210430578905420419e-03])
    t[15] = np.array([1.526390168347900589e-02, 1.057194712326705449e-02, 6.227197709200346307e-03])
    t[16] = np.array([1.535330824089595723e-02, 1.061111283273744217e-02, 6.239077229078052525e-03])
    t[17] = np.array([1.560827166888438533e-02, 1.069867533387121490e-02, 6.254224604773584578e-03])
    t[18] = np.array([1.575221498071291762e-02, 1.075725321425476284e-02, 6.270499217236109868e-03])
    t[19] = np.array([1.584229308691507618e-02, 1.079805814919358037e-02, 6.282114130230559076e-03])
    t[20] = np.array([1.607220242660221260e-02, 1.088319602781667034e-02, 6.297180860777905863e-03])
    t[21] = np.array([1.623370104312566592e-02, 1.092667877505624184e-02, 6.311101775693325061e-03])
    t[22] = np.array([1.627945952307210259e-02, 1.094915174281595877e-02, 6.323066535891997027e-03])

    z[0] =  np.array([6.568389383801178383e+01])
    z[1] =  np.array([6.715279399904912339e+01])
    z[2] =  np.array([6.884020585317244922e+01])
    z[3] =  np.array([7.304879764063647940e+01])
    z[4] =  np.array([7.704931014526289346e+01])
    z[5] =  np.array([8.034183719537027457e+01])
    z[6] =  np.array([8.034639970102949746e+01])
    z[7] =  np.array([8.035181503163056504e+01])
    z[8] =  np.array([8.035474586507447725e+01])
    z[9] =  np.array([8.045134520690847069e+01])
    z[10] = np.array([8.210047514767694565e+01])
    z[11] = np.array([8.405864477246872468e+01])
    z[12] = np.array([8.543220560491654680e+01])
    z[13] = np.array([8.748010665660861207e+01])
    z[14] = np.array([8.889717125836928346e+01])
    z[15] = np.array([9.100408425780254618e+01])
    z[16] = np.array([9.149230504398737196e+01])
    z[17] = np.array([9.298454845720561934e+01])
    z[18] = np.array([9.378547646257639769e+01])
    z[19] = np.array([9.425056411527843636e+01])
    z[20] = np.array([9.553850090959055308e+01])
    z[21] = np.array([9.622233217762658342e+01])
    z[22] = np.array([9.634448513519758706e+01])

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

    # ideal AEP:  array([  3.80582227e+08,   3.86008237e+08,   3.92349241e+08,
    #          4.03964668e+08,   4.16503562e+08,   4.28884306e+08,
    #          4.35040476e+08,   4.41288621e+08,   4.47621670e+08,
    #          4.54316444e+08,   4.65922871e+08,   4.79303087e+08,
    #          4.90765720e+08,   5.05196607e+08,   5.18085846e+08,
    #          5.33937720e+08,   5.44072087e+08,   5.58221879e+08,
    #          5.70052903e+08,   5.80131653e+08,   5.92579178e+08,
    #          6.03311781e+08,   6.12218000e+08])
    # AEP:  array([  3.07852916e+08,   3.12242016e+08,   3.17371253e+08,
    #          3.26766971e+08,   3.36909680e+08,   3.46924462e+08,
    #          3.51904187e+08,   3.56958312e+08,   3.62081114e+08,
    #          3.67496516e+08,   3.76884954e+08,   3.87708208e+08,
    #          3.97473483e+08,   4.09772191e+08,   4.20887290e+08,
    #          4.34759633e+08,   4.43934311e+08,   4.56747104e+08,
    #          4.67629787e+08,   4.77465787e+08,   4.90537456e+08,
    #          5.01774691e+08,   5.10827959e+08])
    # COE:  array([ 60.95340601,  60.5343135 ,  60.09616157,  59.66659909,
    #         59.1440659 ,  58.52839474,  57.792681  ,  57.06691829,
    #         56.3513999 ,  55.64836071,  54.9861217 ,  54.30639424,
    #         53.60898141,  52.91721117,  52.19790319,  51.49366033,
    #         50.74143745,  50.02961579,  49.302417  ,  48.58545541,
    #         47.90841492,  47.21956642,  46.5448462 ])
    # cost:  array([  1.87646838e+10,   1.89013561e+10,   1.90727941e+10,
    #          1.94970738e+10,   1.99262083e+10,   2.03049319e+10,
    #          2.03374864e+10,   2.03705108e+10,   2.04037777e+10,
    #          2.04505787e+10,   2.07234419e+10,   2.10550348e+10,
    #          2.13081485e+10,   2.16840016e+10,   2.19694340e+10,
    #          2.23873649e+10,   2.25258650e+10,   2.28508821e+10,
    #          2.30552788e+10,   2.31978927e+10,   2.35008720e+10,
    #          2.36935833e+10,   2.37764088e+10])
    # tower cost:  array([ 16148460.62586858,  16846891.25405893,  17750706.02986099,
    #         20132317.94840223,  22530719.40113551,  24616717.76340432,
    #         24631934.5155254 ,  24647202.16436171,  24661351.84414344,
    #         24754650.94066271,  26211618.95978993,  28011547.29311379,
    #         29326188.51107006,  31356199.10710458,  32818499.75960995,
    #         35058453.66280904,  35616249.42262504,  37287338.74469142,
    #         38228504.06060486,  38794644.41251355,  40330388.36312236,
    #         41187801.6434521 ,  41380022.57770525])
    # wake loss:  array([ 19.11001226,  19.11001226,  19.11001226,  19.11001226,
    #         19.11001226,  19.11001226,  19.11001226,  19.11001226,
    #         19.11001226,  19.11001226,  19.11001226,  19.11001226,
    #         19.00952606,  18.88857028,  18.76109083,  18.57484178,
    #         18.40524052,  18.17821533,  17.96730009,  17.69699423,
    #         17.2199303 ,  16.82995317,  16.56110088])
