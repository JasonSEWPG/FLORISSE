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
5.600000001110121275e+02,1.120000000222024482e+03,1.680000000333036496e+03,2.240000000444048965e+03,
2.800000000555060979e+03,5.600000001110121275e+02,1.120000000222024482e+03,1.680000000333036496e+03,
2.240000000444048965e+03,2.800000000555060979e+03,5.600000001110121275e+02,1.120000000222024482e+03,
1.680000000333036496e+03,2.240000000444048965e+03,2.800000000555060979e+03,5.600000001110121275e+02,
1.120000000222024482e+03,1.680000000333036496e+03,2.240000000444048965e+03,2.800000000555060979e+03,
5.600000001110121275e+02,1.120000000222024482e+03,1.680000000333036496e+03,2.240000000444048965e+03,
2.800000000555060979e+03])

    turbineY = np.array([
5.600000001110121275e+02,5.600000001110121275e+02,5.600000001110121275e+02,5.600000001110121275e+02,
5.600000001110121275e+02,1.120000000222024482e+03,1.120000000222024482e+03,1.120000000222024482e+03,
1.120000000222024482e+03,1.120000000222024482e+03,1.680000000333036496e+03,1.680000000333036496e+03,
1.680000000333036496e+03,1.680000000333036496e+03,1.680000000333036496e+03,2.240000000444048965e+03,
2.240000000444048965e+03,2.240000000444048965e+03,2.240000000444048965e+03,2.240000000444048965e+03,
2.800000000555060979e+03,2.800000000555060979e+03,2.800000000555060979e+03,2.800000000555060979e+03,
2.800000000555060979e+03])

    d = np.zeros((len(shearExp),3))
    t = np.zeros((len(shearExp),3))
    z = np.zeros((len(shearExp),1))

    d[0] =  np.array([4.914078232748213537e+00, 4.813923065837903970e+00, 3.870000000000000107e+00])
    d[1] =  np.array([4.917970783888011788e+00, 4.814309316961518093e+00, 3.870000000000000107e+00])
    d[2] =  np.array([4.919149413860367304e+00, 4.812243389092666135e+00, 3.870000000000000107e+00])
    d[3] =  np.array([4.920750484486822174e+00, 4.810010309569398679e+00, 3.870000000000000107e+00])
    d[4] =  np.array([4.918288250686670970e+00, 4.805146996867177833e+00, 3.870000000000000107e+00])
    d[5] =  np.array([4.922203542911687713e+00, 4.803991128187392157e+00, 3.870000000000000107e+00])
    d[6] =  np.array([4.923224605013237998e+00, 4.801496179260922759e+00, 3.870000000000000107e+00])
    d[7] =  np.array([4.927360506043768851e+00, 4.800029775417891997e+00, 3.870000000000000107e+00])
    d[8] =  np.array([4.939322301034541063e+00, 4.804732873367984602e+00, 3.870000000000000107e+00])
    d[9] =  np.array([4.942130557752419939e+00, 4.802524802024263018e+00, 3.870000000000000107e+00])
    d[10] = np.array([4.944917125274443492e+00, 4.800138257162477551e+00, 3.870000000000000107e+00])
    d[11] = np.array([4.948558319376624048e+00, 4.798251665921834608e+00, 3.870000000000000107e+00])
    d[12] = np.array([4.948688284249142200e+00, 4.793765660891635250e+00, 3.870000000000000107e+00])
    d[13] = np.array([4.959926565200694526e+00, 4.797296618896099041e+00, 3.870000000000000107e+00])
    d[14] = np.array([4.964974961300643308e+00, 4.796012690031624004e+00, 3.870000000000000107e+00])
    d[15] = np.array([4.969675630780885811e+00, 4.794651693718778418e+00, 3.870000000000000107e+00])
    d[16] = np.array([4.962874980592830276e+00, 4.784086754354523130e+00, 3.870000000000000107e+00])
    d[17] = np.array([4.976906069007556610e+00, 4.789159821314703080e+00, 3.870000000000000107e+00])
    d[18] = np.array([4.981979695399501473e+00, 4.787560309046361873e+00, 3.870000000000000107e+00])
    d[19] = np.array([4.987480380300868710e+00, 4.785616740769283339e+00, 3.870000000000000107e+00])
    d[20] = np.array([4.993788434102738982e+00, 4.784484532386522382e+00, 3.870000000000000107e+00])
    d[21] = np.array([4.997250784318665318e+00, 4.780923663015722802e+00, 3.870000000000000107e+00])
    d[22] = np.array([5.000597700245165811e+00, 4.777436176364043163e+00, 3.870000000000000107e+00])

    t[0] =  np.array([2.878539686075986745e-02, 1.837006552748228147e-02, 1.096072404119075201e-02])
    t[1] =  np.array([2.876271361491002446e-02, 1.836724142261468573e-02, 1.096074643963684293e-02])
    t[2] =  np.array([2.877060100490144365e-02, 1.837641683198973364e-02, 1.096076975185200642e-02])
    t[3] =  np.array([2.876997811208572425e-02, 1.838475137340159216e-02, 1.096079311935803900e-02])
    t[4] =  np.array([2.878936015794551079e-02, 1.839950162378781867e-02, 1.096081683510599820e-02])
    t[5] =  np.array([2.877282966694981112e-02, 1.840387509945272046e-02, 1.096084089890169734e-02])
    t[6] =  np.array([2.876270624895711434e-02, 1.840903548951330210e-02, 1.096086533215831320e-02])
    t[7] =  np.array([2.875441269676658196e-02, 1.841658536842885954e-02, 1.096089012407256216e-02])
    t[8] =  np.array([2.869532219148945504e-02, 1.840654655646068935e-02, 1.096091528545076708e-02])
    t[9] =  np.array([2.868926947423473817e-02, 1.841519494170885604e-02, 1.096094082179726453e-02])
    t[10] = np.array([2.868413593038230541e-02, 1.842458502753514804e-02, 1.096096673869789706e-02])
    t[11] = np.array([2.867399116207079018e-02, 1.843256374128224989e-02, 1.096099304182456512e-02])
    t[12] = np.array([2.868166390976932545e-02, 1.844712234997067379e-02, 1.096101973692894908e-02])
    t[13] = np.array([2.861861758366149311e-02, 1.843649985449471659e-02, 1.096104682985099728e-02])
    t[14] = np.array([2.860191273184104280e-02, 1.844343824155631265e-02, 1.096106698226800152e-02])
    t[15] = np.array([2.857180486414255538e-02, 1.844442026575178820e-02, 1.096110223294719140e-02])
    t[16] = np.array([2.864278293545101384e-02, 1.848483517377691107e-02, 1.096113055523935051e-02])
    t[17] = np.array([2.856358898669345350e-02, 1.846991335868398282e-02, 1.096115929959137585e-02])
    t[18] = np.array([2.853543540204545731e-02, 1.847360961631841492e-02, 1.096118847229088887e-02])
    t[19] = np.array([2.852500135538126036e-02, 1.848485731525017703e-02, 1.096121807971882182e-02])
    t[20] = np.array([2.849737319001883692e-02, 1.848985432320373462e-02, 1.096124812835161909e-02])
    t[21] = np.array([2.849361109083398472e-02, 1.850405417422670493e-02, 1.096127862476329110e-02])
    t[22] = np.array([2.848733751857958699e-02, 1.851632607637201944e-02, 1.096130957561911555e-02])

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
    # AEP:  array([  3.40901236e+08,   3.46965865e+08,   3.53138385e+08,
    #          3.59420713e+08,   3.65814804e+08,   3.72322645e+08,
    #          3.78946261e+08,   3.85687711e+08,   3.92549092e+08,
    #          3.99417608e+08,   4.06316057e+08,   4.13337230e+08,
    #          4.20483310e+08,   4.27664080e+08,   4.34885718e+08,
    #          4.42235829e+08,   4.49586150e+08,   4.57037908e+08,
    #          4.64455286e+08,   4.71832974e+08,   4.79189000e+08,
    #          4.86348066e+08,   4.93457186e+08])
    # COE:  array([ 31.9792749 ,  31.5266913 ,  31.08378269,  30.64780468,
    #         30.21863633,  29.79751691,  29.38231843,  28.9768437 ,
    #         28.57879176,  28.19302264,  27.81875972,  27.45070214,
    #         27.08810636,  26.73606412,  26.39445689,  26.05631763,
    #         25.73230189,  25.41269236,  25.103945  ,  24.80854541,
    #         24.52187188,  24.2517057 ,  23.99079758])
    # cost:  array([  1.09017743e+10,   1.09386857e+10,   1.09768768e+10,
    #          1.10154558e+10,   1.10544245e+10,   1.10942903e+10,
    #          1.11343197e+10,   1.11760125e+10,   1.12185787e+10,
    #          1.12607897e+10,   1.13032088e+10,   1.13463972e+10,
    #          1.13900966e+10,   1.14340543e+10,   1.14785723e+10,
    #          1.15230372e+10,   1.15688865e+10,   1.16145638e+10,
    #          1.16596599e+10,   1.17054898e+10,   1.17506113e+10,
    #          1.17947702e+10,   1.18384315e+10])
    # tower cost:  array([ 16492502.05416659,  16491456.22132029,  16494932.69821703,
    #         16496346.48783814,  16495623.57637617,  16496373.49925517,
    #         16493160.01800605,  16496822.88908517,  16501510.73127213,
    #         16503291.12684137,  16505255.97582011,  16507362.76215431,
    #         16507631.56677322,  16508237.64597997,  16511108.27104714,
    #         16507870.06070897,  16514712.84533846,  16515784.1242598 ,
    #         16514151.54783003,  16519633.49664922,  16520917.61463436,
    #         16523957.05147829,  16525593.83326733])
    # wake loss:  array([ 17.47245106,  17.47245106,  17.47245106,  17.47245106,
    #         17.47245106,  17.47245106,  17.47245106,  17.47245106,
    #         17.47245106,  17.4417489 ,  17.38726895,  17.33361694,
    #         17.28078229,  17.21816631,  17.146649  ,  17.07613706,
    #         16.97514992,  16.8684786 ,  16.76312142,  16.67333726,
    #         16.57121354,  16.41952071,  16.22919614])
