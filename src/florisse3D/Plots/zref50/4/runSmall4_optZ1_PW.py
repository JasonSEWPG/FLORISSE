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
    2.800000000171101533e+02,    5.600000000342204203e+02,    8.400000000513305167e+02,    1.120000000068440841e+03,
    1.400000000085550937e+03,    1.680000000102661261e+03,    1.960000000119771357e+03,    2.240000000136881681e+03,
    2.520000000153992005e+03,    2.800000000171101533e+02,    5.600000000342204203e+02,    8.400000000513305167e+02,
    1.120000000068440841e+03,    1.400000000085550937e+03,    1.680000000102661261e+03,    1.960000000119771357e+03,
    2.240000000136881681e+03,    2.520000000153992005e+03,    2.800000000171101533e+02,    5.600000000342204203e+02,
    8.400000000513305167e+02,    1.120000000068440841e+03,    1.400000000085550937e+03,    1.680000000102661261e+03,
    1.960000000119771357e+03,    2.240000000136881681e+03,    2.520000000153992005e+03,    2.800000000171101533e+02,
    5.600000000342204203e+02,    8.400000000513305167e+02,    1.120000000068440841e+03,    1.400000000085550937e+03,
    1.680000000102661261e+03,    1.960000000119771357e+03,    2.240000000136881681e+03,    2.520000000153992005e+03,
    2.800000000171101533e+02,    5.600000000342204203e+02,    8.400000000513305167e+02,    1.120000000068440841e+03,
    1.400000000085550937e+03,    1.680000000102661261e+03,    1.960000000119771357e+03,    2.240000000136881681e+03,
    2.520000000153992005e+03,    2.800000000171101533e+02,    5.600000000342204203e+02,    8.400000000513305167e+02,
    1.120000000068440841e+03,    1.400000000085550937e+03,    1.680000000102661261e+03,    1.960000000119771357e+03,
    2.240000000136881681e+03,    2.520000000153992005e+03,    2.800000000171101533e+02,    5.600000000342204203e+02,
    8.400000000513305167e+02,    1.120000000068440841e+03,    1.400000000085550937e+03,    1.680000000102661261e+03,
    1.960000000119771357e+03,    2.240000000136881681e+03,    2.520000000153992005e+03,    2.800000000171101533e+02,
    5.600000000342204203e+02,    8.400000000513305167e+02,    1.120000000068440841e+03,    1.400000000085550937e+03,
    1.680000000102661261e+03,    1.960000000119771357e+03,    2.240000000136881681e+03,    2.520000000153992005e+03,
    2.800000000171101533e+02,    5.600000000342204203e+02,    8.400000000513305167e+02,    1.120000000068440841e+03,
    1.400000000085550937e+03,    1.680000000102661261e+03,    1.960000000119771357e+03,    2.240000000136881681e+03,
    2.520000000153992005e+03    ])

    turbineY = np.array([
    2.800000000171101533e+02,    2.800000000171101533e+02,    2.800000000171101533e+02,    2.800000000171101533e+02,
    2.800000000171101533e+02,    2.800000000171101533e+02,    2.800000000171101533e+02,    2.800000000171101533e+02,
    2.800000000171101533e+02,    5.600000000342204203e+02,    5.600000000342204203e+02,    5.600000000342204203e+02,
    5.600000000342204203e+02,    5.600000000342204203e+02,    5.600000000342204203e+02,    5.600000000342204203e+02,
    5.600000000342204203e+02,    5.600000000342204203e+02,    8.400000000513305167e+02,    8.400000000513305167e+02,
    8.400000000513305167e+02,    8.400000000513305167e+02,    8.400000000513305167e+02,    8.400000000513305167e+02,
    8.400000000513305167e+02,    8.400000000513305167e+02,    8.400000000513305167e+02,    1.120000000068440841e+03,
    1.120000000068440841e+03,    1.120000000068440841e+03,    1.120000000068440841e+03,    1.120000000068440841e+03,
    1.120000000068440841e+03,    1.120000000068440841e+03,    1.120000000068440841e+03,    1.120000000068440841e+03,
    1.400000000085550937e+03,    1.400000000085550937e+03,    1.400000000085550937e+03,    1.400000000085550937e+03,
    1.400000000085550937e+03,    1.400000000085550937e+03,    1.400000000085550937e+03,    1.400000000085550937e+03,
    1.400000000085550937e+03,    1.680000000102661261e+03,    1.680000000102661261e+03,    1.680000000102661261e+03,
    1.680000000102661261e+03,    1.680000000102661261e+03,    1.680000000102661261e+03,    1.680000000102661261e+03,
    1.680000000102661261e+03,    1.680000000102661261e+03,    1.960000000119771357e+03,    1.960000000119771357e+03,
    1.960000000119771357e+03,    1.960000000119771357e+03,    1.960000000119771357e+03,    1.960000000119771357e+03,
    1.960000000119771357e+03,    1.960000000119771357e+03,    1.960000000119771357e+03,    2.240000000136881681e+03,
    2.240000000136881681e+03,    2.240000000136881681e+03,    2.240000000136881681e+03,    2.240000000136881681e+03,
    2.240000000136881681e+03,    2.240000000136881681e+03,    2.240000000136881681e+03,    2.240000000136881681e+03,
    2.520000000153992005e+03,    2.520000000153992005e+03,    2.520000000153992005e+03,    2.520000000153992005e+03,
    2.520000000153992005e+03,    2.520000000153992005e+03,    2.520000000153992005e+03,    2.520000000153992005e+03,
    2.520000000153992005e+03    ])

    d = np.zeros((len(shearExp),3))
    t = np.zeros((len(shearExp),3))
    z = np.zeros((len(shearExp),1))

    d[0] =  np.array([4.031666883235522469e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[1] =  np.array([4.309654241153704213e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[2] =  np.array([4.583093158081026353e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[3] =  np.array([5.253766697513531092e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[4] =  np.array([5.830032015809692325e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[5] =  np.array([6.290993670946596339e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[6] =  np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[7] =  np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[8] =  np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[9] =  np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[10] = np.array([6.299999999999999822e+00, 4.122115317985010741e+00, 3.870000000000000107e+00])
    d[11] = np.array([6.299999999999999822e+00, 4.374517943515363000e+00, 3.870000000000000107e+00])
    d[12] = np.array([6.299999999999999822e+00, 4.626920569045713805e+00, 3.870000000000000107e+00])
    d[13] = np.array([6.299999999999999822e+00, 4.899110491420564983e+00, 3.870000000000000107e+00])
    d[14] = np.array([6.299999999999999822e+00, 5.117518975827270999e+00, 3.870000000000000107e+00])
    d[15] = np.array([6.299999999999999822e+00, 5.381778682450956808e+00, 3.870000000000000107e+00])
    d[16] = np.array([6.299999999999999822e+00, 5.610118839174048411e+00, 3.870000000000000107e+00])
    d[17] = np.array([6.299999999999999822e+00, 5.768061947351490915e+00, 3.870000000000000107e+00])
    d[18] = np.array([6.299999999999999822e+00, 5.909761601471559267e+00, 3.870000000000000107e+00])
    d[19] = np.array([6.299999999999999822e+00, 5.953144557719228125e+00, 3.870000000000000107e+00])
    d[20] = np.array([6.299999999999999822e+00, 6.050227973242808588e+00, 3.870000000000000107e+00])
    d[21] = np.array([6.299999999999999822e+00, 6.174967590035166154e+00, 3.870000000000000107e+00])
    d[22] = np.array([6.299999999999999822e+00, 6.225927703983294315e+00, 3.870000000000000107e+00])

    t[0] =  np.array([1.373753248799710475e-02, 8.961265093295424713e-03, 6.069737078572242858e-03])
    t[1] =  np.array([1.344468186914220785e-02, 9.067963102977080053e-03, 6.075430496902587373e-03])
    t[2] =  np.array([1.334668687737095190e-02, 9.215013889967871297e-03, 6.081745000064524287e-03])
    t[3] =  np.array([1.326662639096528110e-02, 9.602737935008043291e-03, 6.094225695038010641e-03])
    t[4] =  np.array([1.330746471623020684e-02, 9.927892387527541967e-03, 6.108721620521439903e-03])
    t[5] =  np.array([1.340224523702634313e-02, 1.018364473682106981e-02, 6.119350514707877844e-03])
    t[6] =  np.array([1.339772447351569515e-02, 1.020186334868223438e-02, 6.125524919173359201e-03])
    t[7] =  np.array([1.339036772579017820e-02, 1.021237711870411416e-02, 6.131083172968316598e-03])
    t[8] =  np.array([1.339368033015011887e-02, 1.022318110814953926e-02, 6.137459738916579456e-03])
    t[9] =  np.array([1.342665364240907058e-02, 1.022530620767867979e-02, 6.145395814451119071e-03])
    t[10] = np.array([1.388025949071897656e-02, 1.017867362358580718e-02, 6.157166515668663527e-03])
    t[11] = np.array([1.410736739304117900e-02, 1.023818489290511200e-02, 6.169351756627831000e-03])
    t[12] = np.array([1.433447529536337889e-02, 1.029769616222441821e-02, 6.181536997586997879e-03])
    t[13] = np.array([1.476698136741198819e-02, 1.033410125843990494e-02, 6.195867071722434283e-03])
    t[14] = np.array([1.487979962512351109e-02, 1.045722822539733865e-02, 6.210450326314632605e-03])
    t[15] = np.array([1.519193104408997179e-02, 1.055198686168482350e-02, 6.233169935338438855e-03])
    t[16] = np.array([1.546501392982327014e-02, 1.064288916556537824e-02, 6.240643400921973283e-03])
    t[17] = np.array([1.566633693102119362e-02, 1.071510377887451883e-02, 6.260106479198360337e-03])
    t[18] = np.array([1.584954525774699349e-02, 1.078337812130044163e-02, 6.277187960594510230e-03])
    t[19] = np.array([1.591319712257295918e-02, 1.082001769234294461e-02, 6.282764581965635114e-03])
    t[20] = np.array([1.604745857984710145e-02, 1.087652028662552256e-02, 6.296122628779887015e-03])
    t[21] = np.array([1.625886986887571314e-02, 1.093016374877251069e-02, 6.311808464664987392e-03])
    t[22] = np.array([1.637077121391420156e-02, 1.097322305404818850e-02, 6.325968409792066122e-03])

    z[0] =  np.array([6.567854953555401210e+01])
    z[1] =  np.array([6.715112871354617141e+01])
    z[2] =  np.array([6.878223434514177370e+01])
    z[3] =  np.array([7.307061329783023496e+01])
    z[4] =  np.array([7.699355271493787711e+01])
    z[5] =  np.array([8.027800640017883893e+01])
    z[6] =  np.array([8.034585346650580107e+01])
    z[7] =  np.array([8.034733508204459440e+01])
    z[8] =  np.array([8.035591094778357046e+01])
    z[9] =  np.array([8.038924707776322975e+01])
    z[10] = np.array([8.211056645856244529e+01])
    z[11] = np.array([8.373496432334449000e+01])
    z[12] = np.array([8.535936218812651077e+01])
    z[13] = np.array([8.727348368080940588e+01])
    z[14] = np.array([8.871108492395632084e+01])
    z[15] = np.array([9.055421965985837573e+01])
    z[16] = np.array([9.217694301375664168e+01])
    z[17] = np.array([9.332317000525370077e+01])
    z[18] = np.array([9.435233385616278667e+01])
    z[19] = np.array([9.467922924159444165e+01])
    z[20] = np.array([9.539927493288458038e+01])
    z[21] = np.array([9.633409183091929151e+01])
    z[22] = np.array([9.674855318226644840e+01])

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

    # ideal AEP:  array([  3.80574795e+08,   3.86005653e+08,   3.92250091e+08,
    #          4.04004476e+08,   4.16395031e+08,   4.28751383e+08,
    #          4.35039233e+08,   4.41277549e+08,   4.47624785e+08,
    #          4.54137566e+08,   4.65953795e+08,   4.78250206e+08,
    #          4.90531161e+08,   5.04492693e+08,   5.17441323e+08,
    #          5.32291705e+08,   5.46587564e+08,   5.59488956e+08,
    #          5.72189947e+08,   5.81669887e+08,   5.92073064e+08,
    #          6.03740332e+08,   6.13852224e+08])
    # AEP:  array([  2.77690225e+08,   2.81652905e+08,   2.86209222e+08,
    #          2.94785928e+08,   3.03826820e+08,   3.12842757e+08,
    #          3.17430750e+08,   3.21982600e+08,   3.26613925e+08,
    #          3.31366041e+08,   3.39987871e+08,   3.48960071e+08,
    #          3.58340087e+08,   3.69052419e+08,   3.79085036e+08,
    #          3.90765954e+08,   4.02379438e+08,   4.13063917e+08,
    #          4.23787669e+08,   4.32357660e+08,   4.42283396e+08,
    #          4.53314434e+08,   4.62873857e+08])
    # COE:  array([ 66.90901188,  66.44461682,  65.95681046,  65.483476  ,
    #         64.90274176,  64.21877347,  63.40523604,  62.59951854,
    #         61.80742065,  61.02653362,  60.29441726,  59.53227825,
    #         58.76738631,  57.99937482,  57.20500505,  56.42366451,
    #         55.62466359,  54.80944713,  53.99315447,  53.18305476,
    #         52.41492636,  51.65720616,  50.89188268])
    # cost:  array([  1.85799786e+10,   1.87143193e+10,   1.88774474e+10,
    #          1.93036072e+10,   1.97191937e+10,   2.00903781e+10,
    #          2.01267716e+10,   2.01559557e+10,   2.01871642e+10,
    #          2.02221208e+10,   2.04993706e+10,   2.07743880e+10,
    #          2.10587103e+10,   2.14048096e+10,   2.16855614e+10,
    #          2.20484471e+10,   2.23822209e+10,   2.26398049e+10,
    #          2.28816331e+10,   2.29941011e+10,   2.31822516e+10,
    #          2.34169572e+10,   2.35565220e+10])
    # tower cost:  array([ 16145835.4016022 ,  16846006.64487366,  17718680.152159  ,
    #         20144924.86701702,  22499035.86698339,  24575119.38194149,
    #         24631373.64514998,  24641287.46261233,  24661934.81985193,
    #         24702811.20770163,  26220907.72343086,  27713911.80933661,
    #         29256121.39591962,  31152035.32093517,  32627544.16979089,
    #         34587545.43860674,  36360418.56779528,  37666726.10979441,
    #         38872426.13151057,  39286902.46413659,  40166542.14337573,
    #         41322031.5431823 ,  41885285.70189101])
    # wake loss:  array([ 27.03399468,  27.03399468,  27.03399468,  27.03399468,
    #         27.03399468,  27.03399468,  27.03399468,  27.03399468,
    #         27.03399468,  27.03399468,  27.03399468,  27.03399468,
    #         26.94855792,  26.84682568,  26.73854619,  26.58800623,
    #         26.38335278,  26.17121167,  25.93584151,  25.66958176,
    #         25.29918648,  24.91566162,  24.59523007])
