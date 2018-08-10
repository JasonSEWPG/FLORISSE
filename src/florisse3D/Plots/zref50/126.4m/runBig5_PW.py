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
7.000000003836817086e+02,1.400000000767363417e+03,2.100000001151045126e+03,2.800000001534726835e+03,
3.500000001918408543e+03,7.000000003836817086e+02,1.400000000767363417e+03,2.100000001151045126e+03,
2.800000001534726835e+03,3.500000001918408543e+03,7.000000003836817086e+02,1.400000000767363417e+03,
2.100000001151045126e+03,2.800000001534726835e+03,3.500000001918408543e+03,7.000000003836817086e+02,
1.400000000767363417e+03,2.100000001151045126e+03,2.800000001534726835e+03,3.500000001918408543e+03,
7.000000003836817086e+02,1.400000000767363417e+03,2.100000001151045126e+03,2.800000001534726835e+03,
3.500000001918408543e+03    ])

    turbineY = np.array([
7.000000003836817086e+02,7.000000003836817086e+02,7.000000003836817086e+02,7.000000003836817086e+02,
7.000000003836817086e+02,1.400000000767363417e+03,1.400000000767363417e+03,1.400000000767363417e+03,
1.400000000767363417e+03,1.400000000767363417e+03,2.100000001151045126e+03,2.100000001151045126e+03,
2.100000001151045126e+03,2.100000001151045126e+03,2.100000001151045126e+03,2.800000001534726835e+03,
2.800000001534726835e+03,2.800000001534726835e+03,2.800000001534726835e+03,2.800000001534726835e+03,
3.500000001918408543e+03,3.500000001918408543e+03,3.500000001918408543e+03,3.500000001918408543e+03,
3.500000001918408543e+03    ])

    d = np.zeros((len(shearExp),3))
    t = np.zeros((len(shearExp),3))
    z = np.zeros((len(shearExp),1))

    d[0] =  np.array([4.909432281333124592e+00, 4.810521701680144702e+00, 3.870000000000000107e+00])
    d[1] =  np.array([4.915334354258751759e+00, 4.812838498602501680e+00, 3.870000000000000107e+00])
    d[2] =  np.array([4.915713171702940087e+00, 4.809869131403178955e+00, 3.870000000000000107e+00])
    d[3] =  np.array([4.919210426633556565e+00, 4.809275694006574398e+00, 3.870000000000000107e+00])
    d[4] =  np.array([4.921981896309947757e+00, 4.807457743214307300e+00, 3.870000000000000107e+00])
    d[5] =  np.array([4.926415577899005527e+00, 4.806990232067035684e+00, 3.870000000000000107e+00])
    d[6] =  np.array([4.927469705196971361e+00, 4.804361577119533600e+00, 3.870000000000000107e+00])
    d[7] =  np.array([4.937150035631873379e+00, 4.807567836660142646e+00, 3.870000000000000107e+00])
    d[8] =  np.array([4.935503453893748116e+00, 4.802030768253668924e+00, 3.870000000000000107e+00])
    d[9] =  np.array([4.942944944330021073e+00, 4.803117266449885570e+00, 3.870000000000000107e+00])
    d[10] = np.array([4.948628855082681355e+00, 4.802972962392390244e+00, 3.870000000000000107e+00])
    d[11] = np.array([4.943385270085509653e+00, 4.794400166276450825e+00, 3.870000000000000107e+00])
    d[12] = np.array([4.954710319729698220e+00, 4.798220304688550542e+00, 3.870000000000000107e+00])
    d[13] = np.array([4.955792595499489472e+00, 4.794148434082376120e+00, 3.870000000000000107e+00])
    d[14] = np.array([4.956446707579179822e+00, 4.789773290813017503e+00, 3.870000000000000107e+00])
    d[15] = np.array([4.966905632178166385e+00, 4.792302806731107090e+00, 3.870000000000000107e+00])
    d[16] = np.array([4.971884746409698685e+00, 4.791489602053575325e+00, 3.870000000000000107e+00])
    d[17] = np.array([4.976171388986106336e+00, 4.788444571721384158e+00, 3.870000000000000107e+00])
    d[18] = np.array([4.979491357700029930e+00, 4.785527769120073494e+00, 3.870000000000000107e+00])
    d[19] = np.array([4.984398013827797236e+00, 4.783413009326286769e+00, 3.870000000000000107e+00])
    d[20] = np.array([4.995345419313667712e+00, 4.786417928828069712e+00, 3.870000000000000107e+00])
    d[21] = np.array([4.999412340622343187e+00, 4.782826487887919420e+00, 3.870000000000000107e+00])
    d[22] = np.array([5.004272335985882947e+00, 4.780074893461667251e+00, 3.870000000000000107e+00])

    t[0] =  np.array([2.880545796656985499e-02, 1.837710542543721792e-02, 1.096072404119075548e-02])
    t[1] =  np.array([2.876639011712384814e-02, 1.836793595290287656e-02, 1.096074672747532341e-02])
    t[2] =  np.array([2.878150169858460783e-02, 1.838017123431273658e-02, 1.096076975185200468e-02])
    t[3] =  np.array([2.875790015353450746e-02, 1.837960334658270828e-02, 1.096079311935812400e-02])
    t[4] =  np.array([2.876673574775499495e-02, 1.839237275068519167e-02, 1.096081683510601902e-02])
    t[5] =  np.array([2.875125645029083174e-02, 1.839636450953247709e-02, 1.096084090428416959e-02])
    t[6] =  np.array([2.875171244304519300e-02, 1.840541642051625112e-02, 1.096086533215831146e-02])
    t[7] =  np.array([2.869898120707435954e-02, 1.839718804141344394e-02, 1.096089012354349579e-02])
    t[8] =  np.array([2.871372376564476983e-02, 1.841284701274771562e-02, 1.096091528545076361e-02])
    t[9] =  np.array([2.868438815117866844e-02, 1.841346888210886393e-02, 1.096094082179730964e-02])
    t[10] = np.array([2.865734870344033638e-02, 1.841479022044360467e-02, 1.096096673869868809e-02])
    t[11] = np.array([2.870765079378160070e-02, 1.844463563422834781e-02, 1.096099303915801931e-02])
    t[12] = np.array([2.864397978928535926e-02, 1.843364015591441793e-02, 1.096101973692899939e-02])
    t[13] = np.array([2.864963890994766046e-02, 1.844798577463984887e-02, 1.096104682985173454e-02])
    t[14] = np.array([2.864953256204291407e-02, 1.845990730437812449e-02, 1.096107432651930585e-02])
    t[15] = np.array([2.860313498978163740e-02, 1.845696977186972454e-02, 1.096110223294719313e-02])
    t[16] = np.array([2.858261747140379561e-02, 1.846128115526648322e-02, 1.096113055523901744e-02])
    t[17] = np.array([2.857212536139784251e-02, 1.847314609746575007e-02, 1.096115929959136717e-02])
    t[18] = np.array([2.856491855987540462e-02, 1.848413412778898354e-02, 1.096118847229088540e-02])
    t[19] = np.array([2.854285778332420198e-02, 1.849126624487774581e-02, 1.096121769266239498e-02])
    t[20] = np.array([2.848839505341017875e-02, 1.848462320277265494e-02, 1.096124812835238410e-02])
    t[21] = np.array([2.847767791824475439e-02, 1.849796663630733431e-02, 1.096127862476329283e-02])
    t[22] = np.array([2.845927471631801153e-02, 1.850723571706221898e-02, 1.096130957562239591e-02])

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
    # AEP:  array([  3.63481089e+08,   3.69947414e+08,   3.76528775e+08,
    #          3.83227218e+08,   3.90044826e+08,   3.96983720e+08,
    #          4.04046056e+08,   4.11234031e+08,   4.18549880e+08,
    #          4.25880950e+08,   4.33250182e+08,   4.40750513e+08,
    #          4.48384274e+08,   4.56061403e+08,   4.63788229e+08,
    #          4.71652515e+08,   4.79461656e+08,   4.87267263e+08,
    #          4.95104956e+08,   5.02978377e+08,   5.10848598e+08,
    #          5.18519462e+08,   5.26129856e+08])
    # COE:  array([ 30.37078055,  29.94609657,  29.53088058,  29.12062693,
    #         28.72106954,  28.32675774,  27.93864302,  27.55763176,
    #         27.18223037,  26.82087953,  26.46888792,  26.123357  ,
    #         25.78291405,  25.45222062,  25.12944124,  24.81375681,
    #         24.50958814,  24.21551858,  23.92941644,  23.65050926,
    #         23.38105403,  23.12601439,  22.88005296])
    # cost:  array([  1.10392044e+10,   1.10784810e+10,   1.11192263e+10,
    #          1.11598168e+10,   1.12025046e+10,   1.12452617e+10,
    #          1.12884985e+10,   1.13326360e+10,   1.13771193e+10,
    #          1.14225017e+10,   1.14676505e+10,   1.15138830e+10,
    #          1.15606532e+10,   1.16077755e+10,   1.16547390e+10,
    #          1.17034708e+10,   1.17514077e+10,   1.17994295e+10,
    #          1.18475727e+10,   1.18956948e+10,   1.19441787e+10,
    #          1.19912885e+10,   1.20378790e+10])
    # tower cost:  array([ 16488618.45124384,  16486922.76713727,  16490808.84591367,
    #         16488353.12394327,  16495875.93464068,  16498502.98100495,
    #         16499129.76014973,  16500725.54187624,  16499147.08831158,
    #         16503444.01107254,  16504339.40309645,  16507294.63172741,
    #         16508227.0336556 ,  16509793.9630091 ,  16507991.44958882,
    #         16512954.73405659,  16514580.53155707,  16516981.94506074,
    #         16518839.8749893 ,  16518952.67934937,  16521845.23930573,
    #         16523600.92069299,  16524263.60238877])
    # wake loss:  array([ 12.00617588,  12.00617588,  12.00617588,  12.00617588,
    #         12.00617588,  12.00617588,  12.00617588,  12.00617588,
    #         12.00617588,  11.97186669,  11.91098618,  11.85103088,
    #         11.79198896,  11.72136976,  11.64021416,  11.56019943,
    #         11.45805506,  11.37000197,  11.27027206,  11.1729957 ,
    #         11.0591465 ,  10.89076294,  10.68258363])
