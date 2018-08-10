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

    d[0] =  np.array([6.299999999999999822e+00, 5.342039137435868668e+00, 3.870000000000000107e+00])
    d[1] =  np.array([6.299999999999999822e+00, 5.339601166461292792e+00, 3.870000000000000107e+00])
    d[2] =  np.array([6.299999999999999822e+00, 5.337599683052275168e+00, 3.870000000000000107e+00])
    d[3] =  np.array([6.299999999999999822e+00, 5.334788436138993895e+00, 3.870000000000000107e+00])
    d[4] =  np.array([6.299999999999999822e+00, 5.332308984883670888e+00, 3.870000000000000107e+00])
    d[5] =  np.array([6.299999999999999822e+00, 5.329625302389318087e+00, 3.870000000000000107e+00])
    d[6] =  np.array([6.299999999999999822e+00, 5.327027597517816737e+00, 3.870000000000000107e+00])
    d[7] =  np.array([6.299999999999999822e+00, 5.324496315784653433e+00, 3.870000000000000107e+00])
    d[8] =  np.array([6.299999999999999822e+00, 5.321578945625829782e+00, 3.870000000000000107e+00])
    d[9] =  np.array([6.299999999999999822e+00, 5.318915483285024948e+00, 3.870000000000000107e+00])
    d[10] = np.array([6.299999999999999822e+00, 5.316982117255053097e+00, 3.870000000000000107e+00])
    d[11] = np.array([6.299999999999999822e+00, 5.312443974251181977e+00, 3.870000000000000107e+00])
    d[12] = np.array([6.299999999999999822e+00, 5.310647123827376248e+00, 3.870000000000000107e+00])
    d[13] = np.array([6.299999999999999822e+00, 5.307415719598258974e+00, 3.870000000000000107e+00])
    d[14] = np.array([6.299999999999999822e+00, 5.304707981716425103e+00, 3.870000000000000107e+00])
    d[15] = np.array([6.299999999999999822e+00, 5.301352978092205070e+00, 3.870000000000000107e+00])
    d[16] = np.array([6.299999999999999822e+00, 5.298264497597791944e+00, 3.870000000000000107e+00])
    d[17] = np.array([6.299999999999999822e+00, 5.295116776750570686e+00, 3.870000000000000107e+00])
    d[18] = np.array([6.299999999999999822e+00, 5.292072365795948308e+00, 3.870000000000000107e+00])
    d[19] = np.array([6.299999999999999822e+00, 5.288677891503358985e+00, 3.870000000000000107e+00])
    d[20] = np.array([6.299999999999999822e+00, 5.285613699310034441e+00, 3.870000000000000107e+00])
    d[21] = np.array([6.299999999999999822e+00, 5.282310004073563547e+00, 3.870000000000000107e+00])
    d[22] = np.array([6.299999999999999822e+00, 5.278879355985431232e+00, 3.870000000000000107e+00])

    t[0] =  np.array([1.501402198332815220e-02, 1.031016511839632187e-02, 6.100323331251567352e-03])
    t[1] =  np.array([1.501998722419947176e-02, 1.032195882980214453e-02, 6.108033344395917726e-03])
    t[2] =  np.array([1.502512513111290034e-02, 1.033495732114542674e-02, 6.111065339731587207e-03])
    t[3] =  np.array([1.502553342691203414e-02, 1.035213903752588843e-02, 6.121029931695098729e-03])
    t[4] =  np.array([1.502993290010026101e-02, 1.036622792020373829e-02, 6.129838383455574018e-03])
    t[5] =  np.array([1.503476466259729902e-02, 1.038057226817683774e-02, 6.138449512394261384e-03])
    t[6] =  np.array([1.504061704606968197e-02, 1.039363644249715821e-02, 6.146507480328230154e-03])
    t[7] =  np.array([1.504381768796672972e-02, 1.040883126628233506e-02, 6.154800487711852008e-03])
    t[8] =  np.array([1.505358899481452212e-02, 1.042167725877435509e-02, 6.163223775385700751e-03])
    t[9] =  np.array([1.505713540947986769e-02, 1.043802436978295833e-02, 6.169607938789083493e-03])
    t[10] = np.array([1.505547216474509807e-02, 1.045149397212216842e-02, 6.178919841691111504e-03])
    t[11] = np.array([1.508641206468592198e-02, 1.045341594193171901e-02, 6.185008030024905824e-03])
    t[12] = np.array([1.507859703261876967e-02, 1.048001525946985930e-02, 6.195974758840733232e-03])
    t[13] = np.array([1.508437628017272186e-02, 1.049673262223921842e-02, 6.203103727930224520e-03])
    t[14] = np.array([1.508240182093223934e-02, 1.051538114843598125e-02, 6.214399791497109832e-03])
    t[15] = np.array([1.510308368510534724e-02, 1.052331234492653346e-02, 6.220168261000338407e-03])
    t[16] = np.array([1.510793473449882456e-02, 1.054162580835752393e-02, 6.231797814131296957e-03])
    t[17] = np.array([1.511807215325117019e-02, 1.055537112222191404e-02, 6.240901096543682858e-03])
    t[18] = np.array([1.512433791381001362e-02, 1.057195751166423170e-02, 6.250599207348902717e-03])
    t[19] = np.array([1.513722135453691824e-02, 1.058482183991603202e-02, 6.256756239038589740e-03])
    t[20] = np.array([1.514337519473068182e-02, 1.060249283454834518e-02, 6.266787301450274690e-03])
    t[21] = np.array([1.515309135492717146e-02, 1.061790327482309802e-02, 6.276448373487632758e-03])
    t[22] = np.array([1.516535536475858113e-02, 1.063134495885644307e-02, 6.288553951564683497e-03])

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
    # AEP:  array([  2.99500217e+08,   3.04828323e+08,   3.10251216e+08,
    #          3.15770582e+08,   3.21388137e+08,   3.27105629e+08,
    #          3.32924835e+08,   3.38847564e+08,   3.44875659e+08,
    #          3.50987319e+08,   3.57110592e+08,   3.63342799e+08,
    #          3.69685876e+08,   3.76125083e+08,   3.82589126e+08,
    #          3.89168163e+08,   3.95821721e+08,   4.02542963e+08,
    #          4.09356127e+08,   4.16222611e+08,   4.23170610e+08,
    #          4.30087315e+08,   4.36959544e+08])
    # COE:  array([ 71.2920079 ,  70.16223994,  69.05181088,  67.96250158,
    #         66.8916058 ,  65.83924963,  64.80461632,  63.78843574,
    #         62.79025696,  61.81286724,  60.86524924,  59.93369233,
    #         59.02355351,  58.12703981,  57.25754642,  56.40129775,
    #         55.56653879,  54.74974431,  53.94959379,  53.16850986,
    #         52.40549258,  51.67000043,  50.96246557])
    # cost:  array([  2.13519719e+10,   2.13874379e+10,   2.14234083e+10,
    #          2.14605587e+10,   2.14981686e+10,   2.15363891e+10,
    #          2.15750662e+10,   2.16145561e+10,   2.16548312e+10,
    #          2.16955325e+10,   2.17356252e+10,   2.17764755e+10,
    #          2.18201741e+10,   2.18630377e+10,   2.19061146e+10,
    #          2.19495895e+10,   2.19944430e+10,   2.20391243e+10,
    #          2.20845968e+10,   2.21299360e+10,   2.21764642e+10,
    #          2.22226117e+10,   2.22685357e+10])
    # tower cost:  array([ 33624668.30612545,  33645922.15917718,  33666629.21273773,
    #         33691647.79733855,  33715641.41564736,  33739635.34395269,
    #         33762424.50823487,  33786530.25571495,  33811668.96509925,
    #         33836191.11813421,  33855752.07431881,  33875985.11774821,
    #         33912066.06432243,  33937763.44486073,  33963910.51731502,
    #         33987833.0117734 ,  34018496.8180616 ,  34044883.43222763,
    #         34072946.21358919,  34097657.89728649,  34127413.8649003 ,
    #         34155785.94104531,  34184510.35642831])
    # wake loss:  array([ 27.03399468,  27.03399468,  27.03399468,  27.03399468,
    #         27.03399468,  27.03399468,  27.03399468,  27.03399468,
    #         27.03399468,  27.0218036 ,  26.96060579,  26.90033719,
    #         26.84098594,  26.77807539,  26.69248707,  26.60810024,
    #         26.50515799,  26.38036108,  26.25021745,  26.10488596,
    #         25.95119955,  25.74759155,  25.49279969])
