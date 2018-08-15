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

    shearExp = 0.15
    nGroups = 2

    datasize = 0
    rotor_diameter = 70.0
    nRows = 9
    nTurbs = nRows**2

    use_rotor_components = False

    d1 = np.zeros((nRuns,3))
    t1 = np.zeros((nRuns,3))
    z1 = np.zeros((nRuns,1))
    d2 = np.zeros((nRuns,3))
    t2 = np.zeros((nRuns,3))
    z2 = np.zeros((nRuns,1))

    d1[0] =  np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d1[1] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d1[2] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d1[3] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d1[4] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d1[5] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d1[6] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d1[7] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d1[8] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d1[9] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])

    d2[0] =  np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[1] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d2[2] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d2[3] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d2[4] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d2[5] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d2[6] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.876301159288642495e+00])
    d2[7] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 4.381420675069112036e+00])
    d2[8] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 4.893905043768631558e+00])
    d2[9] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 5.017298151780923199e+00])

    t1[0] =  np.array([1.339224301498054964e-02, 1.021224120314019948e-02, 6.128677597917596420e-03])
    t1[1] =  np.array([1.201090917984903340e-02, 8.217666319793646632e-03, 6.062630752921796366e-03])
    t1[2] =  np.array([1.090992471711688117e-02, 7.776438282758380179e-03, 6.041637029401298112e-03])
    t1[3] =  np.array([1.043117813448488258e-02, 7.567623512361193287e-03, 6.025384437208491581e-03])
    t1[4] =  np.array([1.019839761907610255e-02, 7.463220143159868952e-03, 6.025905837565847210e-03])
    t1[5] =  np.array([1.019243055838351363e-02, 7.462802815539445155e-03, 6.025905837565598278e-03])
    t1[6] =  np.array([1.019867366414730497e-02, 7.463251463968971489e-03, 6.025905355058460734e-03])
    t1[7] =  np.array([1.019923522131543740e-02, 7.463271941314614417e-03, 6.025900893064654179e-03])
    t1[8] =  np.array([1.019923522901797740e-02, 7.463271949946125722e-03, 6.025903452268641797e-03])
    t1[9] =  np.array([1.019923474001788351e-02, 7.463271406142704158e-03, 6.025742220624985893e-03])

    t2[0] =  np.array([1.339614748925378220e-02, 1.021298351584201507e-02, 6.131963647384377700e-03])
    t2[1] =  np.array([1.614779703640868355e-02, 1.071817861046964020e-02, 6.170430457528397299e-03])
    t2[2] =  np.array([1.614779704681652828e-02, 1.071817864632576729e-02, 6.170430515396003839e-03])
    t2[3] =  np.array([1.614778467461714176e-02, 1.071816440089747371e-02, 6.170108356617495619e-03])
    t2[4] =  np.array([1.614779704683644637e-02, 1.071817864639507817e-02, 6.170430515500152299e-03])
    t2[5] =  np.array([1.614779746985417139e-02, 1.071818018174781201e-02, 6.170432794011957746e-03])
    t2[6] =  np.array([1.615215499310098529e-02, 1.072050844004615068e-02, 6.171743674986423026e-03])
    t2[7] =  np.array([1.651736912062292534e-02, 1.091144448880458466e-02, 6.295817497576860350e-03])
    t2[8] =  np.array([1.691253162576764751e-02, 1.111199568917785319e-02, 6.445177327283302740e-03])
    t2[9] =  np.array([1.701206157796635379e-02, 1.116144360091623697e-02, 6.490965452740825314e-03])

    z1[0] =  np.array([8.034235103922337373e+01])
    z1[1] =  np.array([5.553099370060170514e+01])
    z1[2] =  np.array([4.938828332127174292e+01])
    z1[3] =  np.array([4.646507388610031342e+01])
    z1[4] =  np.array([4.500000000000000000e+01])
    z1[5] =  np.array([4.500000000000000000e+01])
    z1[6] =  np.array([4.500000000000000000e+01])
    z1[7] =  np.array([4.500000000000000000e+01])
    z1[8] =  np.array([4.500000000000000000e+01])
    z1[9] =  np.array([4.500000000000000000e+01])

    z2[0] =  np.array([8.035219581154737511e+01])
    z2[1] =  np.array([9.684400305790336461e+01])
    z2[2] =  np.array([9.684400310207068685e+01])
    z2[3] =  np.array([9.684393455596156741e+01])
    z2[4] =  np.array([9.684400310215485774e+01])
    z2[5] =  np.array([9.684400479420555996e+01])
    z2[6] =  np.array([9.685737293634537082e+01])
    z2[7] =  np.array([9.798720746532920600e+01])
    z2[8] =  np.array([9.924478984628031242e+01])
    z2[9] =  np.array([9.956636725056176829e+01])

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

    prob['shearExp'] = shearExp
    density = np.array([0.0248420225,0.049684045,0.0745260675,0.09936809,0.1242101126,\
                0.1490521351,0.1738941576,0.1987361801,0.2235782026,0.2484202251])
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

        idealAEP[k] = 41.*AEP1 + 40.*AEP2

    print 'ideal AEP: ', repr(idealAEP)
    print 'AEP: ', repr(AEP)
    print 'COE: ', repr(COE)
    print 'cost: ', repr(cost)
    print 'tower cost: ', repr(tower_cost)

    print 'wake loss: ', repr((idealAEP-AEP)/idealAEP*100.)

    # ideal AEP:  array([  4.41277247e+08,   4.26153000e+08,   4.16433276e+08,
    #          4.11573770e+08,   4.09075006e+08,   4.09075007e+08,
    #          4.09088736e+08,   4.10245311e+08,   4.11524065e+08,
    #          4.11849623e+08])
    # AEP:  array([  3.85096935e+08,   3.54097850e+08,   3.27819833e+08,
    #          3.07863127e+08,   2.91514087e+08,   2.77639310e+08,
    #          2.65278546e+08,   2.56052470e+08,   2.47980901e+08,
    #          2.39281001e+08])
    # COE:  array([ 53.34225716,  58.13869533,  61.79511561,  65.15942352,
    #         68.34551393,  71.45518933,  74.5067039 ,  77.51085061,
    #         80.44889602,  83.31513546])
    # cost:  array([  2.05419397e+10,   2.05867870e+10,   2.02576645e+10,
    #          2.00601839e+10,   1.99236801e+10,   1.98387694e+10,
    #          1.97650301e+10,   1.98468447e+10,   1.99497897e+10,
    #          1.99357290e+10])
    # tower cost:  array([ 24643890.58216769,  26665200.56756362,  25653807.18009622,
    #         25206608.52885534,  24991607.97067021,  24990616.92602873,
    #         25003181.8939446 ,  25972652.84760883,  27040438.38448942,
    #         27314654.70255591])
    # wake loss:  array([ 12.73129586,  16.90828161,  21.27914556,  25.198555  ,
    #         28.73823071,  32.12997501,  35.15378879,  37.58552199,
    #         39.74085061,  41.90088147])
