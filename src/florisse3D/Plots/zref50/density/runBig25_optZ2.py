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

    d1[0] =  np.array([6.299999999999999822e+00, 5.084815861437921747e+00, 3.870000000000000107e+00])
    d1[1] =  np.array([6.299999999999999822e+00, 5.154858242281681768e+00, 3.870000000000000107e+00])
    d1[2] =  np.array([6.299999999999999822e+00, 5.170759067801751208e+00, 3.870000000000000107e+00])
    d1[3] =  np.array([6.299999999999999822e+00, 5.222865538569140043e+00, 3.870000000000000107e+00])
    d1[4] =  np.array([6.299999999999999822e+00, 5.153802474040152681e+00, 3.870000000000000107e+00])
    d1[5] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 4.496007137903683670e+00])
    d1[6] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 4.496007133903503039e+00])
    d1[7] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])
    d1[8] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])
    d1[9] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])

    d2[0] =  np.array([6.299999999999999822e+00, 5.157422592406389228e+00, 3.870000000000000107e+00])
    d2[1] =  np.array([6.299999999999999822e+00, 5.225124763513716353e+00, 3.870000000000000107e+00])
    d2[2] =  np.array([6.299999999999999822e+00, 5.166972146017319467e+00, 3.870000000000000107e+00])
    d2[3] =  np.array([6.299999999999999822e+00, 5.154243025010059043e+00, 3.870000000000000107e+00])
    d2[4] =  np.array([6.299999999999999822e+00, 5.153715292051719743e+00, 3.870000000000000107e+00])
    d2[5] =  np.array([4.731230430799243436e+00, 4.650305007860702666e+00, 3.870000000000000107e+00])
    d2[6] =  np.array([4.565599810365715605e+00, 4.480577482438998338e+00, 3.870000000000000107e+00])
    d2[7] =  np.array([4.546197388979932974e+00, 4.465502793439888762e+00, 3.870000000000000107e+00])
    d2[8] =  np.array([4.570516074661894912e+00, 4.485643655808891239e+00, 3.870000000000000107e+00])
    d2[9] =  np.array([4.541322376502024305e+00, 4.461300564697111781e+00, 3.870000000000000107e+00])

    t1[0] =  np.array([2.535814058759496697e-02, 1.894759336142693221e-02, 1.096119581959219963e-02])
    t1[1] =  np.array([2.555812389733524634e-02, 1.875214454499497807e-02, 1.096132963285771297e-02])
    t1[2] =  np.array([2.561684297424114687e-02, 1.868477332749934536e-02, 1.096121694140844705e-02])
    t1[3] =  np.array([2.569888161081852751e-02, 1.863483242707566004e-02, 1.096065621750010044e-02])
    t1[4] =  np.array([2.556669689435171755e-02, 1.876547616248028585e-02, 1.096131801358722277e-02])
    t1[5] =  np.array([2.906887112855758779e-02, 1.742682238659606225e-02, 1.069107227794641006e-02])
    t1[6] =  np.array([2.906887112348545063e-02, 1.742682238639328002e-02, 1.069107227916736395e-02])
    t1[7] =  np.array([3.193291529368773307e-02, 1.854309885385644480e-02, 1.041989487405534792e-02])
    t1[8] =  np.array([3.193456039185577078e-02, 1.854321164098457486e-02, 1.041989709197578551e-02])
    t1[9] =  np.array([3.192690325250158861e-02, 1.853935365001175076e-02, 1.041860492719482584e-02])

    t2[0] =  np.array([2.557974257535221285e-02, 1.873989701204687233e-02, 1.096112399225628378e-02])
    t2[1] =  np.array([2.569840702575953637e-02, 1.862894038871205568e-02, 1.096133635294333983e-02])
    t2[2] =  np.array([2.560571779738565062e-02, 1.870325200045355046e-02, 1.096130368032675016e-02])
    t2[3] =  np.array([2.555369908354272052e-02, 1.876140826852152124e-02, 1.096129520768764076e-02])
    t2[4] =  np.array([2.556636314786515929e-02, 1.876658211428442954e-02, 1.096132207029508997e-02])
    t2[5] =  np.array([2.730108767936750488e-02, 1.754555776083309426e-02, 1.096101005993383934e-02])
    t2[6] =  np.array([2.689137537580223516e-02, 1.729232521603735889e-02, 1.096092067253597382e-02])
    t2[7] =  np.array([2.695497476557945707e-02, 1.732071393084983271e-02, 1.096092067250508360e-02])
    t2[8] =  np.array([2.684621395407947239e-02, 1.727223510453464866e-02, 1.096092067253597903e-02])
    t2[9] =  np.array([2.700241315079144125e-02, 1.732128936819019721e-02, 1.096092067253598076e-02])

    z1[0] =  np.array([1.016513921524487785e+02])
    z1[1] =  np.array([1.021706596797356781e+02])
    z1[2] =  np.array([1.022902059310700196e+02])
    z1[3] =  np.array([1.026648477254384062e+02])
    z1[4] =  np.array([1.021749294918352291e+02])
    z1[5] =  np.array([1.128836011152223904e+02])
    z1[6] =  np.array([1.128836011048823025e+02])
    z1[7] =  np.array([1.190681769546910544e+02])
    z1[8] =  np.array([1.190687709329140631e+02])
    z1[9] =  np.array([1.190513636558863340e+02])

    z2[0] =  np.array([1.021963418140195472e+02])
    z2[1] =  np.array([1.026959641245153563e+02])
    z2[2] =  np.array([1.022650390861818579e+02])
    z2[3] =  np.array([1.021401152157021386e+02])
    z2[4] =  np.array([1.021745978733250126e+02])
    z2[5] =  np.array([7.912236004379585097e+01])
    z2[6] =  np.array([7.320000000000000284e+01])
    z2[7] =  np.array([7.320000000000000284e+01])
    z2[8] =  np.array([7.320000000000000284e+01])
    z2[9] =  np.array([7.320000000000000284e+01])

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

    shearExp = 0.25

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

    # ideal AEP:  array([  5.91136979e+08,   5.92681727e+08,   5.92243820e+08,
    #          5.92651963e+08,   5.91930715e+08,   5.66539441e+08,
    #          5.53635551e+08,   5.63075843e+08,   5.63076744e+08,
    #          5.63050348e+08])
    # AEP:  array([  5.51176665e+08,   5.14562785e+08,   4.80856270e+08,
    #          4.52735072e+08,   4.27295740e+08,   4.02956231e+08,
    #          3.82794412e+08,   3.82620384e+08,   3.70114661e+08,
    #          3.58752488e+08])
    # COE:  array([ 23.15287779,  24.42280896,  25.68520972,  26.92457231,
    #         28.13003171,  29.23002798,  30.15848907,  31.06796337,
    #         31.91401255,  32.72602861])
    # cost:  array([  1.27613260e+10,   1.25670686e+10,   1.23508941e+10,
    #          1.21896982e+10,   1.20198427e+10,   1.17784219e+10,
    #          1.15445011e+10,   1.18872361e+10,   1.18118439e+10,
    #          1.17405442e+10])
    # tower cost:  array([ 20352741.15632575,  20556338.08213493,  20484936.65016896,
    #         20559444.53004859,  20459510.18775579,  19903212.28035983,
    #         19163157.7698234 ,  21593806.96503383,  21601218.71524407,
    #         21587735.76904293])
    # wake loss:  array([  6.7599078 ,  13.18058899,  18.80771831,  23.60861014,
    #         27.81321712,  28.87410798,  30.85805069,  32.04816217,
    #         34.26923329,  36.28411928])
