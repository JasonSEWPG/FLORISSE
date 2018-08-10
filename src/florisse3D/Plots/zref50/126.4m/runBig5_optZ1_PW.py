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

    d[0] =  np.array([4.565532656358957553e+00, 4.478871651800664822e+00, 3.870000000000000107e+00])
    d[1] =  np.array([4.571397463414825779e+00, 4.486098089671519240e+00, 3.870000000000000107e+00])
    d[2] =  np.array([4.567416384097509940e+00, 4.480859400202120035e+00, 3.870000000000000107e+00])
    d[3] =  np.array([4.651103051671702460e+00, 4.568078526326898370e+00, 3.870000000000000107e+00])
    d[4] =  np.array([4.776847503444017207e+00, 4.698505708782411361e+00, 3.870000000000000107e+00])
    d[5] =  np.array([4.900440517570047838e+00, 4.826098239734593065e+00, 3.870000000000000107e+00])
    d[6] =  np.array([5.013069944983060466e+00, 4.823052546923943140e+00, 3.870000000000000107e+00])
    d[7] =  np.array([5.529489556345979828e+00, 4.918070741632439180e+00, 3.870000000000000107e+00])
    d[8] =  np.array([5.866893252040956241e+00, 4.920285893383537612e+00, 3.870000000000000107e+00])
    d[9] =  np.array([6.299999999999999822e+00, 4.822208153927364904e+00, 3.870000000000000107e+00])
    d[10] = np.array([6.299999999999999822e+00, 5.077830111738254715e+00, 3.870000000000000107e+00])
    d[11] = np.array([6.299999999999999822e+00, 5.065991165437111476e+00, 3.870000000000000107e+00])
    d[12] = np.array([6.299999999999999822e+00, 5.084075855226185325e+00, 3.870000000000000107e+00])
    d[13] = np.array([6.299999999999999822e+00, 5.093302096274365809e+00, 3.870000000000000107e+00])
    d[14] = np.array([6.299999999999999822e+00, 5.185163774358684918e+00, 3.870000000000000107e+00])
    d[15] = np.array([6.299999999999999822e+00, 5.174796749374143801e+00, 3.870000000000000107e+00])
    d[16] = np.array([6.299999999999999822e+00, 5.165688672156716343e+00, 3.870000000000000107e+00])
    d[17] = np.array([6.299999999999999822e+00, 5.162272498682480837e+00, 3.870000000000000107e+00])
    d[18] = np.array([6.299999999999999822e+00, 5.206400339311103131e+00, 3.870000000000000107e+00])
    d[19] = np.array([6.299999999999999822e+00, 5.173724148425460356e+00, 3.870000000000000107e+00])
    d[20] = np.array([6.299999999999999822e+00, 5.367287251291319450e+00, 3.870000000000000107e+00])
    d[21] = np.array([6.299999999999999822e+00, 5.478440034593625000e+00, 3.870000000000000107e+00])
    d[22] = np.array([6.299999999999999822e+00, 5.589592817895930210e+00, 3.870000000000000107e+00])

    t[0] =  np.array([2.689269919480526821e-02, 1.729444496790772182e-02, 1.096066201490672291e-02])
    t[1] =  np.array([2.685010060984078031e-02, 1.727384612627792804e-02, 1.096053157856496062e-02])
    t[2] =  np.array([2.687867618282233084e-02, 1.728841254035005937e-02, 1.096069030180397974e-02])
    t[3] =  np.array([2.725371948499887356e-02, 1.749453178092916228e-02, 1.096072500688386082e-02])
    t[4] =  np.array([2.782269920159950033e-02, 1.780960161415296589e-02, 1.096077470641169081e-02])
    t[5] =  np.array([2.836663809916883591e-02, 1.811505887408564400e-02, 1.096082985431587505e-02])
    t[6] =  np.array([2.870734179290053423e-02, 1.855772235251332974e-02, 1.096087680615913905e-02])
    t[7] =  np.array([2.677235657051946216e-02, 1.882768536498809209e-02, 1.096092604909005425e-02])
    t[8] =  np.array([2.575836480347199614e-02, 1.917297166290135926e-02, 1.096098481570657174e-02])
    t[9] =  np.array([2.457387776148113948e-02, 1.977530382740160311e-02, 1.096101857443910164e-02])
    t[10] = np.array([2.513025342336440773e-02, 1.897759700511509814e-02, 1.096106747453592586e-02])
    t[11] = np.array([2.513543769130721292e-02, 1.900338427487447204e-02, 1.096105899155870936e-02])
    t[12] = np.array([2.520527550559059637e-02, 1.894447110879522375e-02, 1.096114224672913723e-02])
    t[13] = np.array([2.525669963919828856e-02, 1.893213580268161486e-02, 1.096116935723225223e-02])
    t[14] = np.array([2.549936075899423074e-02, 1.869243185408714716e-02, 1.095875782390028953e-02])
    t[15] = np.array([2.551736458189465470e-02, 1.871834758706439339e-02, 1.096123642464305256e-02])
    t[16] = np.array([2.553896433334383945e-02, 1.873743517532505676e-02, 1.096016133278157091e-02])
    t[17] = np.array([2.558362104454676295e-02, 1.875031083679949695e-02, 1.096119980408786360e-02])
    t[18] = np.array([2.571735404810517739e-02, 1.867491449375015267e-02, 1.096132721770763305e-02])
    t[19] = np.array([2.569299267083863941e-02, 1.873600599510397841e-02, 1.096129220615775848e-02])
    t[20] = np.array([2.613420559772831192e-02, 1.841560383506632426e-02, 1.096145244294295662e-02])
    t[21] = np.array([2.641380660313255600e-02, 1.825379218465764000e-02, 1.095866601579237500e-02])
    t[22] = np.array([2.669340760853679620e-02, 1.809198053424896160e-02, 1.095587958864179326e-02])

    z[0] =  np.array([7.320000000000000284e+01])
    z[1] =  np.array([7.320000000000000284e+01])
    z[2] =  np.array([7.320000000000000284e+01])
    z[3] =  np.array([7.678675660945374659e+01])
    z[4] =  np.array([8.242211511383247569e+01])
    z[5] =  np.array([8.815184596008758433e+01])
    z[6] =  np.array([9.167332913072286260e+01])
    z[7] =  np.array([9.521399175397696979e+01])
    z[8] =  np.array([9.763991265543805298e+01])
    z[9] =  np.array([9.973116573078067404e+01])
    z[10] = np.array([1.014926871505018511e+02])
    z[11] = np.array([1.014220536268491202e+02])
    z[12] = np.array([1.015634192124448703e+02])
    z[13] = np.array([1.016510441370007811e+02])
    z[14] = np.array([1.023261341942966851e+02])
    z[15] = np.array([1.022733248666106931e+02])
    z[16] = np.array([1.022318512721862760e+02])
    z[17] = np.array([1.022382315977573484e+02])
    z[18] = np.array([1.025948128245838831e+02])
    z[19] = np.array([1.023752774628266167e+02])
    z[20] = np.array([1.038635590930668542e+02])
    z[21] = np.array([1.047437447119433800e+02])
    z[22] = np.array([1.056239303308198885e+02])

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
    #          4.13284020e+08,   4.29448191e+08,   4.47513536e+08,
    #          4.62741966e+08,   4.79339340e+08,   4.93600797e+08,
    #          5.08069143e+08,   5.22225605e+08,   5.32134525e+08,
    #          5.42433759e+08,   5.52699770e+08,   5.64617184e+08,
    #          5.74387698e+08,   5.83337947e+08,   5.92122974e+08,
    #          6.02227329e+08,   6.10699270e+08,   6.25454804e+08,
    #          6.38712044e+08,   6.52572663e+08])
    # AEP:  array([  3.45896588e+08,   3.49874677e+08,   3.53898517e+08,
    #          3.63664414e+08,   3.77887886e+08,   3.93784274e+08,
    #          4.07184351e+08,   4.21789016e+08,   4.34872934e+08,
    #          4.48148570e+08,   4.61302552e+08,   4.70560389e+08,
    #          4.80335601e+08,   4.90026337e+08,   5.01425473e+08,
    #          5.10866939e+08,   5.20055502e+08,   5.29372878e+08,
    #          5.39908947e+08,   5.48532898e+08,   5.62853475e+08,
    #          5.75649621e+08,   5.88529024e+08])
    # COE:  array([ 29.6248982 ,  29.3579023 ,  29.09320907,  28.85461137,
    #         28.58954184,  28.29277703,  27.96212828,  27.60028615,
    #         27.21087228,  26.81037576,  26.41139905,  26.00513945,
    #         25.6164562 ,  25.24372353,  24.88396283,  24.53380316,
    #         24.20556371,  23.89112827,  23.58644911,  23.28980301,
    #         23.01761757,  22.74107134,  22.47474154])
    # cost:  array([  1.02471512e+10,   1.02715866e+10,   1.02960435e+10,
    #          1.04933953e+10,   1.08036415e+10,   1.11412507e+10,
    #          1.13857411e+10,   1.16414975e+10,   1.18332719e+10,
    #          1.20150316e+10,   1.21836458e+10,   1.22369885e+10,
    #          1.23044959e+10,   1.23700894e+10,   1.24774528e+10,
    #          1.25335089e+10,   1.25882366e+10,   1.26473153e+10,
    #          1.27345349e+10,   1.27752231e+10,   1.29555460e+10,
    #          1.30908891e+10,   1.32270377e+10])
    # tower cost:  array([ 11933940.55214095,  11934882.61624805,  11933944.57434152,
    #         12859531.54305269,  14364943.04281902,  15960191.0113309 ,
    #         16997855.23023696,  18098368.89996162,  18860758.99676038,
    #         19548889.18372348,  20153788.35863838,  20131834.46427092,
    #         20185474.18660865,  20230094.04421138,  20490313.92758453,
    #         20479571.59012201,  20470163.1894206 ,  20485691.77983882,
    #         20644446.20259076,  20561722.854106  ,  21205944.15791781,
    #         21603425.811908  ,  22003070.86381607])
    # wake loss:  array([ 12.00617588,  12.00617588,  12.00617588,  12.00617588,
    #         12.00617588,  12.00617588,  12.00617588,  12.00617588,
    #         11.89784607,  11.79378304,  11.66604102,  11.57115985,
    #         11.44806285,  11.33950761,  11.19195675,  11.0588647 ,
    #         10.84833328,  10.59747697,  10.34798301,  10.17953928,
    #         10.00892932,   9.87337296,   9.81402426])
