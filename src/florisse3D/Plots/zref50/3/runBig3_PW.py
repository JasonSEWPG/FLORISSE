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

    d[0] =  np.array([4.907155727421000257e+00, 4.808790010943606141e+00, 3.870000000000000107e+00])
    d[1] =  np.array([4.919520213947410348e+00, 4.815678638149794466e+00, 3.870000000000000107e+00])
    d[2] =  np.array([4.919794822757180164e+00, 4.812692288310524980e+00, 3.870000000000000107e+00])
    d[3] =  np.array([6.299999999999999822e+00, 3.945636598018241070e+00, 3.870000000000000107e+00])
    d[4] =  np.array([4.920536736128623190e+00, 4.806306197968766902e+00, 3.870000000000000107e+00])
    d[5] =  np.array([4.925837234092897532e+00, 4.807974277783719330e+00, 3.870000000000000107e+00])
    d[6] =  np.array([4.928355080495827600e+00, 4.805091269886969663e+00, 3.870000000000000107e+00])
    d[7] =  np.array([4.932082783137645698e+00, 4.803629277580904855e+00, 3.870000000000000107e+00])
    d[8] =  np.array([4.936974886925444395e+00, 4.802950995728558325e+00, 3.870000000000000107e+00])
    d[9] =  np.array([4.937246551871896649e+00, 4.798917022336406824e+00, 3.870000000000000107e+00])
    d[10] = np.array([4.945224330173039640e+00, 4.800370250955873885e+00, 3.870000000000000107e+00])
    d[11] = np.array([4.943091473054466611e+00, 4.794213904010444338e+00, 3.870000000000000107e+00])
    d[12] = np.array([4.954679847666521475e+00, 4.798297429354060561e+00, 3.870000000000000107e+00])
    d[13] = np.array([4.968779035598542393e+00, 4.803820452934095542e+00, 3.870000000000000107e+00])
    d[14] = np.array([4.972301190029980056e+00, 4.801939748479716386e+00, 3.870000000000000107e+00])
    d[15] = np.array([4.966014744110774259e+00, 4.791700090832923919e+00, 3.870000000000000107e+00])
    d[16] = np.array([6.299999999999999822e+00, 3.954532190366839561e+00, 3.870000000000000107e+00])
    d[17] = np.array([4.976811821282029591e+00, 4.789636048662389456e+00, 3.870000000000000107e+00])
    d[18] = np.array([4.978456554439315340e+00, 4.784951513258335787e+00, 3.870000000000000107e+00])
    d[19] = np.array([4.985661700167629995e+00, 4.784315890051390241e+00, 3.870000000000000107e+00])
    d[20] = np.array([4.991717005044703726e+00, 4.782822860603353377e+00, 3.870000000000000107e+00])
    d[21] = np.array([4.993162989776277705e+00, 4.779033810225885581e+00, 3.870000000000000107e+00])
    d[22] = np.array([4.995587419948944508e+00, 4.773613009163921284e+00, 3.870000000000000107e+00])

    t[0] =  np.array([2.883272149436081136e-02, 1.838608908152278634e-02, 1.096072404119075201e-02])
    t[1] =  np.array([2.874873049542560344e-02, 1.836205931529477989e-02, 1.096074672747532515e-02])
    t[2] =  np.array([2.876542138405720206e-02, 1.837403504796128462e-02, 1.096076975185200642e-02])
    t[3] =  np.array([2.219729885275299741e-02, 2.248046623135627545e-02, 1.096078879526651839e-02])
    t[4] =  np.array([2.877917914329018723e-02, 1.839689940238823512e-02, 1.096081683510601902e-02])
    t[5] =  np.array([2.874126475597443137e-02, 1.839115856132596954e-02, 1.096084090428416959e-02])
    t[6] =  np.array([2.872855109852508193e-02, 1.839715686302722392e-02, 1.096086533191578498e-02])
    t[7] =  np.array([2.873355333011764351e-02, 1.840969714796401868e-02, 1.096089012407225859e-02])
    t[8] =  np.array([2.871139761548255342e-02, 1.841228147902606790e-02, 1.096091528545076882e-02])
    t[9] =  np.array([2.871320305048184751e-02, 1.842265675516217838e-02, 1.096094082179730964e-02])
    t[10] = np.array([2.868298409430448948e-02, 1.842422683935590957e-02, 1.096096673869869503e-02])
    t[11] = np.array([2.870779058087618435e-02, 1.844459824471885584e-02, 1.096099304182456338e-02])
    t[12] = np.array([2.864212560989724821e-02, 1.843266267823942539e-02, 1.096101973692899419e-02])
    t[13] = np.array([2.856416841971342810e-02, 1.841696587933711049e-02, 1.096104675932712928e-02])
    t[14] = np.array([2.855442690425570584e-02, 1.842561395096106724e-02, 1.096107432651693969e-02])
    t[15] = np.array([2.860776361895440295e-02, 1.845860114368741076e-02, 1.096110223294719140e-02])
    t[16] = np.array([2.223439889888991286e-02, 2.241970714257374847e-02, 1.096113055523936092e-02])
    t[17] = np.array([2.856831324538393335e-02, 1.846503304933220721e-02, 1.096115929959137932e-02])
    t[18] = np.array([2.856547760537711880e-02, 1.848481199626342622e-02, 1.096118847229088540e-02])
    t[19] = np.array([2.853448662967441110e-02, 1.848818364913614246e-02, 1.096121807971913754e-02])
    t[20] = np.array([2.851498118374965265e-02, 1.849638575881157668e-02, 1.096124812835171797e-02])
    t[21] = np.array([2.850984018353158386e-02, 1.850944598641909161e-02, 1.096127862476214444e-02])
    t[22] = np.array([2.851959699688913497e-02, 1.852946017667857909e-02, 1.096130957169600892e-02])

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
        print shearExp
        print shearExp[k]
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

    # ideal AEP:  array([  4.13075682e+08,   4.20424294e+08,   4.27903638e+08,
    #          4.35516040e+08,   4.43263866e+08,   4.51149525e+08,
    #          4.59175471e+08,   4.67344198e+08,   4.75658246e+08,
    #          4.83800955e+08,   4.91832254e+08,   5.00006429e+08,
    #          5.08326023e+08,   5.16615858e+08,   5.24886095e+08,
    #          5.33303459e+08,   5.41507934e+08,   5.49776909e+08,
    #          5.57992195e+08,   5.66244895e+08,   5.74368896e+08,
    #          5.81891933e+08,   5.89056286e+08])
    # AEP:  array([  3.02617331e+08,   3.08000891e+08,   3.13480223e+08,
    #          3.19057033e+08,   3.24733055e+08,   3.30510052e+08,
    #          3.36389823e+08,   3.42374194e+08,   3.48465028e+08,
    #          3.54549288e+08,   3.60649531e+08,   3.66858297e+08,
    #          3.73177517e+08,   3.79516718e+08,   3.85881815e+08,
    #          3.92360147e+08,   3.98823180e+08,   4.05371865e+08,
    #          4.11932639e+08,   4.18553587e+08,   4.25182113e+08,
    #          4.31607559e+08,   4.37970007e+08])
    # COE:  array([ 35.25214055,  34.74162782,  34.242918  ,  33.78076148,
    #         33.26913934,  32.79367012,  32.32630304,  31.87139558,
    #         31.42138226,  30.98629809,  30.56772225,  30.15418183,
    #         29.74734128,  29.35318535,  28.97062378,  28.59348968,
    #         28.24726821,  27.87278403,  27.52669229,  27.18848609,
    #         26.86102601,  26.55193868,  26.25626015])
    # cost:  array([  1.06679087e+10,   1.07004523e+10,   1.07344776e+10,
    #          1.07779895e+10,   1.08035892e+10,   1.08386376e+10,
    #          1.08742394e+10,   1.09119434e+10,   1.09492528e+10,
    #          1.09861699e+10,   1.10242347e+10,   1.10623118e+10,
    #          1.11010389e+10,   1.11400246e+10,   1.11792369e+10,
    #          1.12189458e+10,   1.12656653e+10,   1.12988425e+10,
    #          1.13391430e+10,   1.13798384e+10,   1.14208278e+10,
    #          1.14600174e+10,   1.14994545e+10])
    # tower cost:  array([ 16492859.33752908,  16490308.83580766,  16494292.34217262,
    #         16563078.61774057,  16496893.32803011,  16495077.98298375,
    #         16492718.8893107 ,  16501024.28209037,  16501712.79659994,
    #         16499834.30227974,  16505608.78475723,  16506640.37648823,
    #         16507491.11055629,  16509335.82618657,  16511679.58521711,
    #         16512600.23719032,  16565298.90901314,  16515479.55554846,
    #         16517040.92356223,  16518800.18715913,  16522365.26213906,
    #         16521856.30499885,  16525955.68387577])
    # wake loss:  array([ 26.74046322,  26.74046322,  26.74046322,  26.74046322,
    #         26.74046322,  26.74046322,  26.74046322,  26.74046322,
    #         26.74046322,  26.71587675,  26.67224888,  26.62928403,
    #         26.58697373,  26.53792715,  26.48275144,  26.42835138,
    #         26.34952233,  26.26611649,  26.17591372,  26.08258538,
    #         25.97403581,  25.82685291,  25.64886964])
