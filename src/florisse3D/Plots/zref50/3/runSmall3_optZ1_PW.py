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
    z = np.zeros((len(shearExp),1))

    d[0] =  np.array([4.031876248202890700e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[1] =  np.array([4.309950364062105521e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[2] =  np.array([4.575930291280817741e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[3] =  np.array([5.262999895020611163e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[4] =  np.array([5.829949830284644641e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[5] =  np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[6] =  np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[7] =  np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[8] =  np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[9] =  np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[10] = np.array([6.299999999999999822e+00, 4.117445508343767990e+00, 3.870000000000000107e+00])
    d[11] = np.array([6.299999999999999822e+00, 4.362630650244921000e+00, 3.870000000000000107e+00])
    d[12] = np.array([6.299999999999999822e+00, 4.607815792146074507e+00, 3.870000000000000107e+00])
    d[13] = np.array([6.299999999999999822e+00, 4.884025413812286054e+00, 3.870000000000000107e+00])
    d[14] = np.array([6.299999999999999822e+00, 5.037173552384908604e+00, 3.870000000000000107e+00])
    d[15] = np.array([6.299999999999999822e+00, 5.352849700373274011e+00, 3.870000000000000107e+00])
    d[16] = np.array([6.299999999999999822e+00, 5.542655172744654912e+00, 3.870000000000000107e+00])
    d[17] = np.array([6.299999999999999822e+00, 5.700507638700017310e+00, 3.870000000000000107e+00])
    d[18] = np.array([6.299999999999999822e+00, 5.863977668306750601e+00, 3.870000000000000107e+00])
    d[19] = np.array([6.299999999999999822e+00, 5.951546391470803000e+00, 3.870000000000000107e+00])
    d[20] = np.array([6.299999999999999822e+00, 6.039115114634854287e+00, 3.870000000000000107e+00])
    d[21] = np.array([6.299999999999999822e+00, 6.174973086174018633e+00, 3.870000000000000107e+00])
    d[22] = np.array([6.299999999999999822e+00, 6.220673902477386719e+00, 3.870000000000000107e+00])

    t[0] =  np.array([1.373755823634198632e-02, 8.961345551353992744e-03, 6.069695887100739172e-03])
    t[1] =  np.array([1.344462830788646299e-02, 9.068070397818216860e-03, 6.075115894383009349e-03])
    t[2] =  np.array([1.334922534438087033e-02, 9.210554993872695217e-03, 6.082161938898402592e-03])
    t[3] =  np.array([1.326673869376527454e-02, 9.607746558775409965e-03, 6.094035792144982672e-03])
    t[4] =  np.array([1.330902409768397465e-02, 9.927000828824973991e-03, 6.107198894237807381e-03])
    t[5] =  np.array([1.340218317421715960e-02, 1.019028035973838439e-02, 6.119584580832231231e-03])
    t[6] =  np.array([1.339919220159042565e-02, 1.020149313276963628e-02, 6.125018316162765541e-03])
    t[7] =  np.array([1.339394482375047199e-02, 1.021424556292965802e-02, 6.131475945953701601e-03])
    t[8] =  np.array([1.339934670900455073e-02, 1.021818740172799646e-02, 6.136746599788173336e-03])
    t[9] =  np.array([1.347758214119364716e-02, 1.020975582672756059e-02, 6.145152649457741138e-03])
    t[10] = np.array([1.387325291757741247e-02, 1.017939781284339130e-02, 6.156659761503342244e-03])
    t[11] = np.array([1.409011232070112700e-02, 1.023776801219705700e-02, 6.169413599906011000e-03])
    t[12] = np.array([1.430697172382484374e-02, 1.029613821155072120e-02, 6.182167438308678979e-03])
    t[13] = np.array([1.460995167725374563e-02, 1.037728043499951819e-02, 6.195446939296978157e-03])
    t[14] = np.array([1.486450169225921568e-02, 1.040868291072987589e-02, 6.214248692657214912e-03])
    t[15] = np.array([1.515713175717753845e-02, 1.054299706179914406e-02, 6.224958118653852422e-03])
    t[16] = np.array([1.538778332789906764e-02, 1.062030649933114711e-02, 6.239249362734419421e-03])
    t[17] = np.array([1.558416161998380477e-02, 1.069144004125973865e-02, 6.253684975009662222e-03])
    t[18] = np.array([1.579049746858843944e-02, 1.076814993625811387e-02, 6.269493090648209539e-03])
    t[19] = np.array([1.591237102051378000e-02, 1.082000297835048300e-02, 6.282553220095428000e-03])
    t[20] = np.array([1.603424457243911652e-02, 1.087185602044284972e-02, 6.295613349542646814e-03])
    t[21] = np.array([1.625887704748410489e-02, 1.093015698965309895e-02, 6.311787141379021464e-03])
    t[22] = np.array([1.636181498905323398e-02, 1.097055225673399664e-02, 6.325956581145496943e-03])

    z[0] =  np.array([6.567986614829598579e+01])
    z[1] =  np.array([6.715279701518103650e+01])
    z[2] =  np.array([6.873744125319370823e+01])
    z[3] =  np.array([7.313192516994406844e+01])
    z[4] =  np.array([7.699738526959727380e+01])
    z[5] =  np.array([8.034193423599138839e+01])
    z[6] =  np.array([8.034681038001522779e+01])
    z[7] =  np.array([8.035378677496241551e+01])
    z[8] =  np.array([8.035380674634561160e+01])
    z[9] =  np.array([8.042596041137397833e+01])
    z[10] = np.array([8.208064617330525437e+01])
    z[11] = np.array([8.365449275666592000e+01])
    z[12] = np.array([8.522833934002659362e+01])
    z[13] = np.array([8.709655357837702638e+01])
    z[14] = np.array([8.819290303397797004e+01])
    z[15] = np.array([9.035334593358865618e+01])
    z[16] = np.array([9.170418579068424947e+01])
    z[17] = np.array([9.283739108228138548e+01])
    z[18] = np.array([9.401877046003882299e+01])
    z[19] = np.array([9.466926500788196000e+01])
    z[20] = np.array([9.531975955572508497e+01])
    z[21] = np.array([9.633412163664513628e+01])
    z[22] = np.array([9.670899184645917046e+01])

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


    # ideal AEP:  array([  3.80576626e+08,   3.86008242e+08,   3.92173440e+08,
    #          4.04116312e+08,   4.16402492e+08,   4.28884508e+08,
    #          4.35041409e+08,   4.41293494e+08,   4.47619159e+08,
    #          4.54243329e+08,   4.65862102e+08,   4.77988174e+08,
    #          4.90109059e+08,   5.03889445e+08,   5.15644136e+08,
    #          5.31555908e+08,   5.44851134e+08,   5.57670876e+08,
    #          5.70932761e+08,   5.81635575e+08,   5.91783957e+08,
    #          6.03740446e+08,   6.13692251e+08])
    # AEP:  array([  2.29318646e+08,   2.32591498e+08,   2.36306373e+08,
    #          2.43502620e+08,   2.50905729e+08,   2.58426840e+08,
    #          2.62136716e+08,   2.65903946e+08,   2.69715512e+08,
    #          2.73706944e+08,   2.80707903e+08,   2.88014537e+08,
    #          2.95622664e+08,   3.04317709e+08,   3.11778714e+08,
    #          3.22034421e+08,   3.30840468e+08,   3.39490643e+08,
    #          3.48583727e+08,   3.56420394e+08,   3.64333007e+08,
    #          3.73631577e+08,   3.81371102e+08])
    # COE:  array([ 79.73416979,  79.17173698,  78.57945231,  78.01028311,
    #         77.30363856,  76.47919063,  75.49124854,  74.51685625,
    #         73.55581553,  72.61191521,  71.7227976 ,  70.79862555,
    #         69.87662992,  68.9498832 ,  67.98618527,  67.06075598,
    #         66.10037574,  65.12784117,  64.16331788,  63.20786085,
    #         62.28459167,  61.37087483,  60.44327432])
    # cost:  array([  1.82845319e+10,   1.84146729e+10,   1.85688254e+10,
    #          1.89957083e+10,   1.93959258e+10,   1.97642755e+10,
    #          1.97890280e+10,   1.98143261e+10,   1.98391444e+10,
    #          1.98743854e+10,   2.01331561e+10,   2.03910334e+10,
    #          2.06571155e+10,   2.09826705e+10,   2.11966454e+10,
    #          2.15958717e+10,   2.18686792e+10,   2.21102927e+10,
    #          2.23662885e+10,   2.25285707e+10,   2.26923325e+10,
    #          2.29300967e+10,   2.30513181e+10])
    # tower cost:  array([ 16146499.25736432,  16846755.85285272,  17694712.55348704,
    #         20180332.09994264,  22499349.8059318 ,  24617179.1323556 ,
    #         24632057.50928923,  24648212.98341035,  24659378.52771848,
    #         24733524.99454877,  26193714.31553246,  27639445.55458517,
    #         29131409.60355898,  30972794.38871579,  32106897.10679964,
    #         34367728.9668173 ,  35844180.65151103,  37123443.08465368,
    #         38490330.64327099,  39276288.65290488,  40072555.02028491,
    #         41322042.61672229,  41835351.10951237])
    # wake loss:  array([ 39.74442187,  39.74442187,  39.74442187,  39.74442187,
    #         39.74442187,  39.74442187,  39.74442187,  39.74442187,
    #         39.74442187,  39.74442187,  39.74442187,  39.74442187,
    #         39.68226899,  39.6062546 ,  39.53606905,  39.41664165,
    #         39.2787411 ,  39.12347634,  38.94487211,  38.72101206,
    #         38.43479498,  38.11387348,  37.85629502])
