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
    rotor_diameter = 126.4

    nRows = 5
    nTurbs = nRows**2

    turbineX = np.array([
    4.199999999573020091e+02,    8.399999999146041318e+02,    1.259999999871906084e+03,    1.679999999829208264e+03,
    2.099999999786510216e+03,    4.199999999573020091e+02,    8.399999999146041318e+02,    1.259999999871906084e+03,
    1.679999999829208264e+03,    2.099999999786510216e+03,    4.199999999573020091e+02,    8.399999999146041318e+02,
    1.259999999871906084e+03,    1.679999999829208264e+03,    2.099999999786510216e+03,    4.199999999573020091e+02,
    8.399999999146041318e+02,    1.259999999871906084e+03,    1.679999999829208264e+03,    2.099999999786510216e+03,
    4.199999999573020091e+02,    8.399999999146041318e+02,    1.259999999871906084e+03,    1.679999999829208264e+03,
    2.099999999786510216e+03,    ])

    turbineY = np.array([
    4.199999999573020091e+02,    4.199999999573020091e+02,    4.199999999573020091e+02,    4.199999999573020091e+02,
    4.199999999573020091e+02,    8.399999999146041318e+02,    8.399999999146041318e+02,    8.399999999146041318e+02,
    8.399999999146041318e+02,    8.399999999146041318e+02,    1.259999999871906084e+03,    1.259999999871906084e+03,
    1.259999999871906084e+03,    1.259999999871906084e+03,    1.259999999871906084e+03,    1.679999999829208264e+03,
    1.679999999829208264e+03,    1.679999999829208264e+03,    1.679999999829208264e+03,    1.679999999829208264e+03,
    2.099999999786510216e+03,    2.099999999786510216e+03,    2.099999999786510216e+03,    2.099999999786510216e+03,
    2.099999999786510216e+03,    ])

    d = np.zeros((len(shearExp),3))
    t = np.zeros((len(shearExp),3))
    z = np.zeros((len(shearExp),1))

    d[0] =  np.array([4.566095893236161274e+00, 4.479393579588903940e+00, 3.870000000000000107e+00])
    d[1] =  np.array([4.562927446591896086e+00, 4.476582332520526286e+00, 3.870000000000000107e+00])
    d[2] =  np.array([4.569254850404981916e+00, 4.482605778175588362e+00, 3.870000000000000107e+00])
    d[3] =  np.array([4.650095596014178767e+00, 4.567085660445609818e+00, 3.870000000000000107e+00])
    d[4] =  np.array([4.777966743801109750e+00, 4.699255113185441424e+00, 3.870000000000000107e+00])
    d[5] =  np.array([4.900452924272197919e+00, 4.826028432252489786e+00, 3.870000000000000107e+00])
    d[6] =  np.array([6.299999999999999822e+00, 4.029504338536843200e+00, 3.870000000000000107e+00])
    d[7] =  np.array([5.400101243162804820e+00, 4.980575201893661763e+00, 3.870000000000000107e+00])
    d[8] =  np.array([5.911338163873766760e+00, 4.861848159741462894e+00, 3.870000000000000107e+00])
    d[9] =  np.array([6.299999999999999822e+00, 4.648982553913112881e+00, 3.870000000000000107e+00])
    d[10] = np.array([6.299999999999999822e+00, 5.024491612242774963e+00, 3.870000000000000107e+00])
    d[11] = np.array([6.299999999999999822e+00, 5.065642230309819638e+00, 3.870000000000000107e+00])
    d[12] = np.array([6.299999999999999822e+00, 5.079977880380464939e+00, 3.870000000000000107e+00])
    d[13] = np.array([6.299999999999999822e+00, 5.197769933196061842e+00, 3.870000000000000107e+00])
    d[14] = np.array([6.299999999999999822e+00, 5.187673920429269891e+00, 3.870000000000000107e+00])
    d[15] = np.array([6.299999999999999822e+00, 5.175270915662891547e+00, 3.870000000000000107e+00])
    d[16] = np.array([6.299999999999999822e+00, 5.165200823200771119e+00, 3.870000000000000107e+00])
    d[17] = np.array([6.299999999999999822e+00, 5.232491594210011421e+00, 3.870000000000000107e+00])
    d[18] = np.array([6.299999999999999822e+00, 5.444323877765031128e+00, 3.870000000000000107e+00])
    d[19] = np.array([6.299999999999999822e+00, 5.586253374066673771e+00, 3.870000000000000107e+00])
    d[20] = np.array([6.299999999999999822e+00, 5.610763237197915920e+00, 3.870000000000000107e+00])
    d[21] = np.array([6.299999999999999822e+00, 5.675589920919452247e+00, 3.870000000000000107e+00])
    d[22] = np.array([6.299999999999999822e+00, 5.781249248411302943e+00, 3.870000000000000107e+00])

    t[0] =  np.array([2.689041348770712658e-02, 1.729356363194372095e-02, 1.096066201490442267e-02])
    t[1] =  np.array([2.690205767882562077e-02, 1.729832054126238111e-02, 1.096067609051564612e-02])
    t[2] =  np.array([2.686608767224954800e-02, 1.728328761830777785e-02, 1.096069030180397280e-02])
    t[3] =  np.array([2.725924248369328978e-02, 1.749672269247128792e-02, 1.096072465267731652e-02])
    t[4] =  np.array([2.776020873863862398e-02, 1.778069596231334620e-02, 1.096078057429970137e-02])
    t[5] =  np.array([2.837457005149656916e-02, 1.811858827889388512e-02, 1.096083023395684475e-02])
    t[6] =  np.array([2.272449864230883729e-02, 2.246634351812831429e-02, 1.096088506667077661e-02])
    t[7] =  np.array([2.746716416575575512e-02, 1.851182487331047566e-02, 1.096092648179992912e-02])
    t[8] =  np.array([2.552540205296796594e-02, 1.940684410910630334e-02, 1.096097304401315829e-02])
    t[9] =  np.array([2.421386786866024446e-02, 2.035677576751766804e-02, 1.096100949157409639e-02])
    t[10] = np.array([2.502009354446689190e-02, 1.913720367672428965e-02, 1.096106405446520965e-02])
    t[11] = np.array([2.513057796615583706e-02, 1.900630965115744989e-02, 1.096108202812677915e-02])
    t[12] = np.array([2.519421390719095780e-02, 1.896805900638716402e-02, 1.096102224908114889e-02])
    t[13] = np.array([2.547632640484811042e-02, 1.867007789605136178e-02, 1.096117159984420428e-02])
    t[14] = np.array([2.549887793230067382e-02, 1.868859062960277762e-02, 1.096061343637310184e-02])
    t[15] = np.array([2.551782028620489320e-02, 1.871473257120858866e-02, 1.096002450665246528e-02])
    t[16] = np.array([2.554632204222305122e-02, 1.873379741945543908e-02, 1.096129301288417679e-02])
    t[17] = np.array([2.572508471573965380e-02, 1.862537000144567564e-02, 1.096133358224302308e-02])
    t[18] = np.array([2.618699891246580036e-02, 1.829052630428794346e-02, 1.096138975364544139e-02])
    t[19] = np.array([2.654113857100502336e-02, 1.809672312781554840e-02, 1.096144905625361099e-02])
    t[20] = np.array([2.664377285052128494e-02, 1.806679163248760267e-02, 1.096150060666299490e-02])
    t[21] = np.array([2.683574138565340828e-02, 1.800458309874804208e-02, 1.096092603528877121e-02])
    t[22] = np.array([2.712772007201041599e-02, 1.790343299126682247e-02, 1.096065669584306464e-02])

    z[0] =  np.array([7.320000000000000284e+01])
    z[1] =  np.array([7.320000000000000284e+01])
    z[2] =  np.array([7.320000000000000284e+01])
    z[3] =  np.array([7.678063020521960880e+01])
    z[4] =  np.array([8.221670293481257374e+01])
    z[5] =  np.array([8.817287333733413845e+01])
    z[6] =  np.array([9.262362931731894378e+01])
    z[7] =  np.array([9.522475919801485134e+01])
    z[8] =  np.array([9.751859161381347008e+01])
    z[9] =  np.array([9.855641781644303023e+01])
    z[10] = np.array([1.011262000374300953e+02])
    z[11] = np.array([1.014186195350277728e+02])
    z[12] = np.array([1.015385062301710661e+02])
    z[13] = np.array([1.023952029708525799e+02])
    z[14] = np.array([1.023434940855858741e+02])
    z[15] = np.array([1.022786929870232058e+02])
    z[16] = np.array([1.022299744599238807e+02])
    z[17] = np.array([1.027633161193467686e+02])
    z[18] = np.array([1.043903015221598594e+02])
    z[19] = np.array([1.055202458629894693e+02])
    z[20] = np.array([1.057389043477904664e+02])
    z[21] = np.array([1.062869667450312789e+02])
    z[22] = np.array([1.071689665476397693e+02])

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

    L_reinforced, Toweryaw, E, G, rho, sigma_y, kidx, kx, ky, kz, ktx, kty, ktz, nK, \
                midx, m, mIxx, mIyy, mIzz, mIxy, mIxz, mIyz, mrhox, mrhoy, mrhoz, nMass, \
                addGravityLoadForExtraMass, wind_zref, wind_z0, shearExp, wind_Uref1, \
                plidx1, Fx1, Fy1, Fz1, Mxx1, Myy1, Mzz1, nPL, wind_Uref2, \
                plidx2, Fx2, Fy2, Fz2, Mxx2, Myy2, Mzz2, gamma_f, gamma_m, gamma_n, \
                gamma_b, z_DEL, M_DEL, nDEL, gamma_fatigue, life, m_SN, min_d_to_t, min_taper \
                = setupTower(n)

    shearExp = np.linspace(0.08,0.3,23)

    nPoints = 3
    nFull = n
    rhoAir = air_density

    """OpenMDAO"""

    COE = np.zeros(23)
    AEP = np.zeros(23)
    idealAEP = np.zeros(23)
    cost = np.zeros(23)
    tower_cost = np.zeros(23)

    prob = Problem()
    root = prob.root = Group()

    for i in range(nGroups):
        root.add('d_param%s'%i, IndepVarComp('d_param%s'%i, d[0]), promotes=['*'])
        root.add('t_param%s'%i, IndepVarComp('t_param%s'%i, t[0]), promotes=['*'])
        root.add('get_z_param%s'%i, get_z(nPoints))
        root.add('get_z_full%s'%i, get_z(nFull))
        root.add('Tower%s_max_thrust'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
        root.add('Tower%s_max_speed'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])

        root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(z[0])), promotes=['*'])

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

    prob['ratedPower'] = np.ones(nTurbs)*5000. # in kw

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

    for k in range(23):
        prob['shearExp'] = shearExp[k]
        prob['d_param0'] = d[k]
        prob['t_param0'] = t[k]
        prob['turbineH0'] = z[k]
        prob.run()
        COE[k] = prob['COE']
        AEP[k] = prob['AEP']
        cost[k] = prob['farm_cost']
        tower_cost[k] = prob['tower_cost']



    nGroups = 1
    prob = Problem()
    root = prob.root = Group()

    for i in range(nGroups):
        root.add('d_param%s'%i, IndepVarComp('d_param%s'%i, d[0]), promotes=['*'])
        root.add('t_param%s'%i, IndepVarComp('t_param%s'%i, t[0]), promotes=['*'])
        root.add('get_z_param%s'%i, get_z(nPoints))
        root.add('get_z_full%s'%i, get_z(nFull))
        root.add('Tower%s_max_thrust'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
        root.add('Tower%s_max_speed'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])

        root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(z[0])), promotes=['*'])

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

    prob['ratedPower'] = np.ones(1)*5000. # in kw

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

    for k in range(23):
        prob['shearExp'] = shearExp[k]
        prob['d_param0'] = d[k]
        prob['t_param0'] = t[k]
        prob['turbineH0'] = z[k]
        prob.run()
        idealAEP[k] = prob['AEP']*25.

    print 'ideal AEP: ', repr(idealAEP)
    print 'AEP: ', repr(AEP)
    print 'COE: ', repr(COE)
    print 'cost: ', repr(cost)
    print 'tower cost: ', repr(tower_cost)

    print 'wake loss: ', repr((idealAEP-AEP)/idealAEP*100.)

    # ideal AEP:  array([  3.93091892e+08,   3.97612765e+08,   4.02185631e+08,
    #          4.13273138e+08,   4.29062587e+08,   4.47555165e+08,
    #          4.64750622e+08,   4.79363732e+08,   4.93325610e+08,
    #          5.05203032e+08,   5.21307420e+08,   5.32125254e+08,
    #          5.42365210e+08,   5.54812018e+08,   5.64669867e+08,
    #          5.74404282e+08,   5.83332604e+08,   5.93712602e+08,
    #          6.07991228e+08,   6.21406045e+08,   6.32213803e+08,
    #          6.44596787e+08,   6.58800398e+08])
    # AEP:  array([  2.87977299e+08,   2.91289270e+08,   2.94639330e+08,
    #          3.02761987e+08,   3.14329263e+08,   3.27876841e+08,
    #          3.40474153e+08,   3.51179650e+08,   3.61783823e+08,
    #          3.70805422e+08,   3.83127527e+08,   3.91453354e+08,
    #          3.99502107e+08,   4.09381258e+08,   4.17289979e+08,
    #          4.25210999e+08,   4.32886977e+08,   4.42105142e+08,
    #          4.54785563e+08,   4.66698861e+08,   4.76252792e+08,
    #          4.87015477e+08,   4.99231102e+08])
    # COE:  array([ 34.35457163,  34.03198297,  33.71577563,  33.42902993,
    #         33.10817267,  32.75504707,  32.39721136,  31.91836006,
    #         31.4603423 ,  30.97454743,  30.50560502,  30.02517595,
    #         29.56177632,  29.11218499,  28.67152135,  28.24517032,
    #         27.84913541,  27.47256737,  27.1008936 ,  26.72433951,
    #         26.3439428 ,  25.97626611,  25.61483334])
    # cost:  array([  9.89333675e+09,   9.91315147e+09,   9.93399356e+09,
    #          1.01210395e+10,   1.04068675e+10,   1.07396214e+10,
    #          1.10304131e+10,   1.12090785e+10,   1.13818429e+10,
    #          1.14855301e+10,   1.16875370e+10,   1.17534558e+10,
    #          1.18099919e+10,   1.19179829e+10,   1.19643385e+10,
    #          1.20101571e+10,   1.20555280e+10,   1.21457633e+10,
    #          1.23250952e+10,   1.24722188e+10,   1.25463763e+10,
    #          1.26508436e+10,   1.27877215e+10])
    # tower cost:  array([ 11934468.25109019,  11931396.67772063,  11934116.88809882,
    #         12857900.97971862,  14307956.85420625,  15967114.30597654,
    #         17341935.94767909,  18090877.15011676,  18828012.36772105,
    #         19159437.05945804,  20027207.90054901,  20130533.60216072,
    #         20181559.77200128,  20510097.78913343,  20496887.01028877,
    #         20479498.40713717,  20469410.46119757,  20704995.14833817,
    #         21412020.99131322,  21929255.97242977,  22039540.59727028,
    #         22309754.1202873 ,  22744234.77917372])
    # wake loss:  array([ 26.74046322,  26.74046322,  26.74046322,  26.74046322,
    #         26.74046322,  26.74046322,  26.74046322,  26.74046322,
    #         26.66429325,  26.60269271,  26.50641206,  26.43586234,
    #         26.34075717,  26.21261889,  26.10018646,  25.97356735,
    #         25.79071128,  25.53549636,  25.19866374,  24.89631143,
    #         24.6690298 ,  24.44649326,  24.22118996])
