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
    nRuns = 10
    nGroups = 2

    datasize = 0
    rotor_diameter = 126.4
    nRows = 5
    nTurbs = nRows**2

    use_rotor_components = False

    d1 = np.zeros((nRuns,3))
    t1 = np.zeros((nRuns,3))
    z1 = np.zeros((nRuns,1))
    d2 = np.zeros((nRuns,3))
    t2 = np.zeros((nRuns,3))
    z2 = np.zeros((nRuns,1))

    d1[0] =  np.array([4.564483666024780284e+00, 4.478379714646034238e+00, 3.870000000000000107e+00])
    d1[1] =  np.array([4.567575514865621855e+00, 4.480696427375935542e+00, 3.870000000000000107e+00])
    d1[2] =  np.array([4.570508953155612453e+00, 4.483528938005642672e+00, 3.870000000000000107e+00])
    d1[3] =  np.array([4.563181795971372523e+00, 4.476583249469865855e+00, 3.870000000000000107e+00])
    d1[4] =  np.array([4.564422151060805177e+00, 4.477970528711303189e+00, 3.870000000000000107e+00])
    d1[5] =  np.array([4.551473846448559790e+00, 4.467513960196434653e+00, 3.870000000000000107e+00])
    d1[6] =  np.array([4.559295042826699174e+00, 4.474576550399952168e+00, 3.870000000000000107e+00])
    d1[7] =  np.array([4.556225033010999681e+00, 4.472250999883224054e+00, 3.870000000000000107e+00])
    d1[8] =  np.array([4.563996022266804786e+00, 4.477408582121781500e+00, 3.870000000000000107e+00])
    d1[9] =  np.array([4.567322336878096856e+00, 4.480516643741571770e+00, 3.870000000000000107e+00])

    d2[0] =  np.array([4.564724963111074452e+00, 4.478610031658998736e+00, 3.870000000000000107e+00])
    d2[1] =  np.array([4.568161550620737366e+00, 4.481272541432367973e+00, 3.870000000000000107e+00])
    d2[2] =  np.array([4.571295523462908683e+00, 4.484308815396567560e+00, 3.870000000000000107e+00])
    d2[3] =  np.array([4.569197088422150443e+00, 4.482313997725630550e+00, 3.870000000000000107e+00])
    d2[4] =  np.array([4.566704888945458229e+00, 4.480140939751279916e+00, 3.870000000000000107e+00])
    d2[5] =  np.array([4.588888459648298834e+00, 4.500979018518873609e+00, 3.870000000000000107e+00])
    d2[6] =  np.array([4.564330483848824116e+00, 4.479266730031589816e+00, 3.870000000000000107e+00])
    d2[7] =  np.array([6.299999999999999822e+00, 5.631059943146104807e+00, 3.870000000000000107e+00])
    d2[8] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d2[9] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])

    t1[0] =  np.array([2.688385510441610241e-02, 1.729039725569793060e-02, 1.096066201490631872e-02])
    t1[1] =  np.array([2.687344937197049324e-02, 1.728672222359386307e-02, 1.096063826496435599e-02])
    t1[2] =  np.array([2.685399332894416319e-02, 1.727914007517440675e-02, 1.096066142249990660e-02])
    t1[3] =  np.array([2.690316953494948857e-02, 1.729856469216037307e-02, 1.096066201490672291e-02])
    t1[4] =  np.array([2.689466897812553894e-02, 1.729489585796686185e-02, 1.096066146084024619e-02])
    t1[5] =  np.array([2.698477180642631923e-02, 1.733100314009462101e-02, 1.096066201490672291e-02])
    t1[6] =  np.array([2.693837388534095836e-02, 1.730772314741736359e-02, 1.096066201144167522e-02])
    t1[7] =  np.array([2.690617888153568407e-02, 1.729589601252217931e-02, 1.096066201490582433e-02])
    t1[8] =  np.array([2.690710769494702309e-02, 1.730045901406272735e-02, 1.096066201490671944e-02])
    t1[9] =  np.array([2.687475160760910767e-02, 1.728693019247271728e-02, 1.096066201490672291e-02])

    t2[0] =  np.array([2.688323795751021081e-02, 1.729017980142117175e-02, 1.096066201490632219e-02])
    t2[1] =  np.array([2.686787030276140825e-02, 1.728390004417722572e-02, 1.096063760716016813e-02])
    t2[2] =  np.array([2.684751508571334849e-02, 1.727641516796384405e-02, 1.096066142349838568e-02])
    t2[3] =  np.array([2.686510126099774659e-02, 1.728306738882170318e-02, 1.096066201490672118e-02])
    t2[4] =  np.array([2.687912742397554483e-02, 1.728855409682816466e-02, 1.096066146078803448e-02])
    t2[5] =  np.array([2.672264495106894261e-02, 1.722456888446098622e-02, 1.096066201490672118e-02])
    t2[6] =  np.array([2.690300348674007011e-02, 1.729300946348705584e-02, 1.096066201144266575e-02])
    t2[7] =  np.array([2.597925339077638424e-02, 1.798800929096535176e-02, 1.096077268092988249e-02])
    t2[8] =  np.array([2.756464290447126209e-02, 1.744513999255186201e-02, 1.096078974621999581e-02])
    t2[9] =  np.array([2.756464290447126902e-02, 1.744513999255194875e-02, 1.096078974622007041e-02])

    z1[0] =  np.array([7.320000000000000284e+01])
    z1[1] =  np.array([7.320000000000000284e+01])
    z1[2] =  np.array([7.320000000000000284e+01])
    z1[3] =  np.array([7.320000000000000284e+01])
    z1[4] =  np.array([7.320000000000000284e+01])
    z1[5] =  np.array([7.320000000000000284e+01])
    z1[6] =  np.array([7.320000000000000284e+01])
    z1[7] =  np.array([7.320000000000000284e+01])
    z1[8] =  np.array([7.320000000000000284e+01])
    z1[9] =  np.array([7.320000000000000284e+01])

    z2[0] =  np.array([7.320000000000000284e+01])
    z2[1] =  np.array([7.320000000000000284e+01])
    z2[2] =  np.array([7.320000000000000284e+01])
    z2[3] =  np.array([7.320000000000000284e+01])
    z2[4] =  np.array([7.320000000000000284e+01])
    z2[5] =  np.array([7.320000000000000284e+01])
    z2[6] =  np.array([7.320000000000000284e+01])
    z2[7] =  np.array([1.054895238967273485e+02])
    z2[8] =  np.array([1.110178013579909901e+02])
    z2[9] =  np.array([1.110178013579910470e+02])

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

    shearExp = 0.08

    nPoints = 3
    nFull = n
    rhoAir = air_density

    """OpenMDAO"""

    COE = np.zeros(nRuns)
    AEP = np.zeros(nRuns)
    idealAEP = np.zeros(nRuns)
    cost = np.zeros(nRuns)
    tower_cost = np.zeros(nRuns)

    prob = Problem()
    root = prob.root = Group()

    root.add('d_param0', IndepVarComp('d_param0', d1[0]), promotes=['*'])
    root.add('t_param0', IndepVarComp('t_param0', t1[0]), promotes=['*'])
    root.add('turbineH0', IndepVarComp('turbineH0', float(z1[0])), promotes=['*'])
    root.add('d_param1', IndepVarComp('d_param1', d2[0]), promotes=['*'])
    root.add('t_param1', IndepVarComp('t_param1', t2[0]), promotes=['*'])
    root.add('turbineH1', IndepVarComp('turbineH1', float(z2[0])), promotes=['*'])

    for i in range(nGroups):
        root.add('get_z_param%s'%i, get_z(nPoints))
        root.add('get_z_full%s'%i, get_z(nFull))
        root.add('Tower%s_max_thrust'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
        root.add('Tower%s_max_speed'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])

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

    prob['shearExp'] = shearExp
    density = np.array([0.025,0.05,0.075,0.1,0.125,\
                0.15,0.175,0.2,0.225,0.25])
    for k in range(nRuns):

        spacing = nRows/(2.*nRows-2.)*np.sqrt(3.1415926535/density[k])
        points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
        xpoints, ypoints = np.meshgrid(points, points)
        turbineX = np.ndarray.flatten(xpoints)
        turbineY = np.ndarray.flatten(ypoints)

        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY

        prob['d_param0'] = d1[k]
        prob['t_param0'] = t1[k]
        prob['turbineH0'] = z1[k]
        prob['d_param1'] = d2[k]
        prob['t_param1'] = t2[k]
        prob['turbineH1'] = z2[k]

        prob.run()
        COE[k] = prob['COE']
        AEP[k] = prob['AEP']
        cost[k] = prob['farm_cost']
        tower_cost[k] = prob['tower_cost']



    nGroups = 1
    prob = Problem()
    root = prob.root = Group()

    for i in range(nGroups):
        root.add('d_param%s'%i, IndepVarComp('d_param%s'%i, d1[0]), promotes=['*'])
        root.add('t_param%s'%i, IndepVarComp('t_param%s'%i, t1[0]), promotes=['*'])
        root.add('get_z_param%s'%i, get_z(nPoints))
        root.add('get_z_full%s'%i, get_z(nFull))
        root.add('Tower%s_max_thrust'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
        root.add('Tower%s_max_speed'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])

        root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(z1[0][i])), promotes=['*'])

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


    prob['shearExp'] = shearExp

    for k in range(nRuns):

        prob['d_param0'] = d1[k]
        prob['t_param0'] = t1[k]
        prob['turbineH0'] = z1[k]
        prob.run()
        AEP1 = prob['AEP']

        prob['d_param0'] = d2[k]
        prob['t_param0'] = t2[k]
        prob['turbineH0'] = z2[k]
        prob.run()
        AEP2 = prob['AEP']

        idealAEP[k] = 13.*AEP1 + 12.*AEP2

    print 'ideal AEP: ', repr(idealAEP)
    print 'AEP: ', repr(AEP)
    print 'COE: ', repr(COE)
    print 'cost: ', repr(cost)
    print 'tower cost: ', repr(tower_cost)

    print 'wake loss: ', repr((idealAEP-AEP)/idealAEP*100.)

    # ideal AEP:  array([  3.93091892e+08,   3.93091892e+08,   3.93091892e+08,
    #          3.93091892e+08,   3.93091892e+08,   3.93091892e+08,
    #          3.93091892e+08,   4.10386764e+08,   4.12927389e+08,
    #          4.12927389e+08])
    # AEP:  array([  3.62559731e+08,   3.35904581e+08,   3.13961935e+08,
    #          2.95390289e+08,   2.79470081e+08,   2.65635572e+08,
    #          2.53539350e+08,   2.63934135e+08,   2.59843899e+08,
    #          2.51457719e+08])
    # COE:  array([ 28.54255226,  30.32365298,  32.01628205,  33.64490383,
    #         35.21330562,  36.72942951,  38.19110677,  39.5737698 ,
    #         40.85567157,  42.01366474])
    # cost:  array([  1.03483801e+10,   1.01858539e+10,   1.00518939e+10,
    #          9.93837787e+09,   9.84106538e+09,   9.75664303e+09,
    #          9.68294837e+09,   1.04448687e+10,   1.06160970e+10,
    #          1.05646603e+10])
    # tower cost:  array([ 11929691.82889178,  11932146.81105111,  11932944.92212378,
    #         11932629.69374506,  11932338.04184561,  11933114.26973423,
    #         11934668.62117205,  16644940.6809602 ,  17962879.79808424,
    #         17961442.82661904])
    # wake loss:  array([  7.76718151,  14.54807708,  20.13014215,  24.85464721,
    #         28.90464369,  32.42405202,  35.50125177,  35.68648938,
    #         37.07273806,  39.10364734])
