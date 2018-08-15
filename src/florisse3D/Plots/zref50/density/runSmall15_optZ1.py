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
    nGroups = 1

    datasize = 0
    rotor_diameter = 70.0
    nRows = 9
    nTurbs = nRows**2

    use_rotor_components = False

    d = np.zeros((nRuns,3))
    t = np.zeros((nRuns,3))
    z = np.zeros((nRuns,1))

    d[0] =  np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[1] =  np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[2] =  np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[3] =  np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[4] =  np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[5] =  np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[6] =  np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[7] =  np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[8] =  np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d[9] =  np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])

    t[0] =  np.array([1.339658487732565961e-02, 1.021153171046277827e-02, 6.130087542186924039e-03])
    t[1] =  np.array([1.339675719515922431e-02, 1.021134228873393779e-02, 6.130525609115401731e-03])
    t[2] =  np.array([1.339893509967605978e-02, 1.021095565251082735e-02, 6.132016902309527125e-03])
    t[3] =  np.array([1.339589519978089810e-02, 1.021197199826080933e-02, 6.130676763819551713e-03])
    t[4] =  np.array([1.339585972753252856e-02, 1.021266557132632230e-02, 6.130825989298015373e-03])
    t[5] =  np.array([1.339584784656926673e-02, 1.021103631019252937e-02, 6.130309715387175298e-03])
    t[6] =  np.array([1.338336527254928125e-02, 1.021682873964033754e-02, 6.132784102115408642e-03])
    t[7] =  np.array([1.339912803926418539e-02, 1.021067038268205253e-02, 6.131863695215781195e-03])
    t[8] =  np.array([1.339611023253591861e-02, 1.021294774247323046e-02, 6.131059352636605148e-03])
    t[9] =  np.array([1.339702616135756211e-02, 1.021173437790728021e-02, 6.131488689123398145e-03])

    z[0] =  np.array([8.035000050965689411e+01])
    z[1] =  np.array([8.034988882787777698e+01])
    z[2] =  np.array([8.035352334673335406e+01])
    z[3] =  np.array([8.034869809880632374e+01])
    z[4] =  np.array([8.035096694181322619e+01])
    z[5] =  np.array([8.034652874502580744e+01])
    z[6] =  np.array([8.034555981433931038e+01])
    z[7] =  np.array([8.035350688901061744e+01])
    z[8] =  np.array([8.035195001397158876e+01])
    z[9] =  np.array([8.035118246347673221e+01])

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

    for i in range(nGroups):
        root.add('d_param%s'%i, IndepVarComp('d_param%s'%i, d[0]), promotes=['*'])
        root.add('t_param%s'%i, IndepVarComp('t_param%s'%i, t[0]), promotes=['*'])
        root.add('get_z_param%s'%i, get_z(nPoints))
        root.add('get_z_full%s'%i, get_z(nFull))
        root.add('Tower%s_max_thrust'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
        root.add('Tower%s_max_speed'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])

        root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(z[0][i])), promotes=['*'])

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

        prob['d_param0'] = d[k]
        prob['t_param0'] = t[k]
        prob['turbineH0'] = z[k]

        prob.run()
        COE[k] = prob['COE']
        AEP[k] = prob['AEP']
        cost[k] = prob['farm_cost']
        tower_cost[k] = prob['tower_cost']



    prob = Problem()
    root = prob.root = Group()

    for i in range(nGroups):
        root.add('d_param%s'%i, IndepVarComp('d_param%s'%i, d[0]), promotes=['*'])
        root.add('t_param%s'%i, IndepVarComp('t_param%s'%i, t[0]), promotes=['*'])
        root.add('get_z_param%s'%i, get_z(nPoints))
        root.add('get_z_full%s'%i, get_z(nFull))
        root.add('Tower%s_max_thrust'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
        root.add('Tower%s_max_speed'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])

        root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(z[0][i])), promotes=['*'])

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

        prob['d_param0'] = d[k]
        prob['t_param0'] = t[k]
        prob['turbineH0'] = z[k]

        prob.run()
        idealAEP[k] = prob['AEP']*81.

    print 'ideal AEP: ', repr(idealAEP)
    print 'AEP: ', repr(AEP)
    print 'COE: ', repr(COE)
    print 'cost: ', repr(cost)
    print 'tower cost: ', repr(tower_cost)

    print 'wake loss: ', repr((idealAEP-AEP)/idealAEP*100.)

    # ideal AEP:  array([  4.41284137e+08,   4.41283861e+08,   4.41292843e+08,
    #          4.41280918e+08,   4.41286525e+08,   4.41275557e+08,
    #          4.41273162e+08,   4.41292802e+08,   4.41288955e+08,
    #          4.41287058e+08])
    # AEP:  array([  3.85102965e+08,   3.40448317e+08,   3.05536495e+08,
    #          2.76946537e+08,   2.53396877e+08,   2.33671821e+08,
    #          2.16834123e+08,   2.02364806e+08,   1.89774010e+08,
    #          1.78741065e+08])
    # COE:  array([  53.34204455,   59.53723809,   65.64352437,   71.78756555,
    #          77.89228454,   83.94944808,   89.99284638,   95.99533063,
    #         101.95844161,  107.87417131])
    # cost:  array([  2.05421795e+10,   2.02693525e+10,   2.00564924e+10,
    #          1.98813177e+10,   1.97376617e+10,   1.96166204e+10,
    #          1.95135199e+10,   1.94260765e+10,   1.93490624e+10,
    #          1.92815442e+10])
    # tower cost:  array([ 24645175.66464813,  24645277.72087136,  24648314.531715  ,
    #         24645077.13285568,  24646613.93779092,  24643122.23898767,
    #         24641568.45437333,  24648046.68210733,  24647540.43943525,
    #         24646814.59946063])
    # wake loss:  array([ 12.73129192,  22.8504943 ,  30.76332417,  37.24030983,
    #         42.57769887,  47.0462803 ,  50.86170166,  54.14273576,
    #         56.99552224,  59.49551172])
