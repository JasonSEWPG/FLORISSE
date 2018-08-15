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

    d1[0] =  np.array([5.339797985581240525e+00, 4.952794884930523445e+00, 3.870000000000000107e+00])
    d1[1] =  np.array([6.299999999999999822e+00, 4.679885529313594361e+00, 3.870000000000000107e+00])
    d1[2] =  np.array([6.299999999999999822e+00, 5.248201809565558484e+00, 3.870000000000000107e+00])
    d1[3] =  np.array([4.569237326844922897e+00, 4.483057016935810601e+00, 3.870000000000000107e+00])
    d1[4] =  np.array([4.588967757750674892e+00, 4.501711210492655546e+00, 3.870000000000000107e+00])
    d1[5] =  np.array([4.556076957859620968e+00, 4.470623764190750471e+00, 3.870000000000000107e+00])
    d1[6] =  np.array([4.562479827708060931e+00, 4.476748037731302254e+00, 3.870000000000000107e+00])
    d1[7] =  np.array([4.567533985280156195e+00, 4.481506590536929124e+00, 3.870000000000000107e+00])
    d1[8] =  np.array([4.568744587662975754e+00, 4.482620649999291551e+00, 3.870000000000000107e+00])
    d1[9] =  np.array([4.571110695796898327e+00, 4.484754497303202747e+00, 3.870000000000000107e+00])

    d2[0] =  np.array([5.368973707693648123e+00, 4.968763994320246624e+00, 3.870000000000000107e+00])
    d2[1] =  np.array([4.916250683302947344e+00, 4.788647059650107174e+00, 3.870000000000000107e+00])
    d2[2] =  np.array([4.571921614166591930e+00, 4.485651422623104345e+00, 3.870000000000000107e+00])
    d2[3] =  np.array([6.299999999999999822e+00, 5.621133270823942318e+00, 3.870000000000000107e+00])
    d2[4] =  np.array([6.299999999999999822e+00, 6.285649259016022228e+00, 3.870000000000000107e+00])
    d2[5] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d2[6] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d2[7] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d2[8] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 5.062112878453073783e+00])
    d2[9] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 5.062112256932712384e+00])

    t1[0] =  np.array([2.763247064141686332e-02, 1.853532835975170434e-02, 1.096092120253541226e-02])
    t1[1] =  np.array([2.423425533818582278e-02, 2.026156693461212202e-02, 1.096094987431922747e-02])
    t1[2] =  np.array([2.538553407440627754e-02, 1.856766482112647346e-02, 1.096097612865345843e-02])
    t1[3] =  np.array([2.685965285460575891e-02, 1.727993023910811068e-02, 1.096076343237806054e-02])
    t1[4] =  np.array([2.672007037176698033e-02, 1.722212753911910033e-02, 1.096076343964454310e-02])
    t1[5] =  np.array([2.695235702966368094e-02, 1.731807929734392737e-02, 1.096076343964452228e-02])
    t1[6] =  np.array([2.690723092087002680e-02, 1.729985278510387872e-02, 1.096076343964454310e-02])
    t1[7] =  np.array([2.687780053438408936e-02, 1.728755790886598795e-02, 1.096076343964454483e-02])
    t1[8] =  np.array([2.686199282304579938e-02, 1.728092742349933778e-02, 1.096076343964454483e-02])
    t1[9] =  np.array([2.685140336561829197e-02, 1.727434781526180144e-02, 1.096076343964454310e-02])

    t2[0] =  np.array([2.756104370895232775e-02, 1.851483419706349484e-02, 1.096092632179944230e-02])
    t2[1] =  np.array([2.887378602529947630e-02, 1.847043848490211027e-02, 1.096084349546458739e-02])
    t2[2] =  np.array([2.684600717202545103e-02, 1.727457346729613374e-02, 1.096076343964454483e-02])
    t2[3] =  np.array([2.615938637322575908e-02, 1.801604440346711924e-02, 1.096099468585335843e-02])
    t2[4] =  np.array([2.781200290783480966e-02, 1.747311562264712917e-02, 1.096102938091757587e-02])
    t2[5] =  np.array([2.784964693180687345e-02, 1.746431258102115400e-02, 1.096103021406423343e-02])
    t2[6] =  np.array([2.784966026178429047e-02, 1.746433042825069198e-02, 1.096103029829969405e-02])
    t2[7] =  np.array([2.784966026177791015e-02, 1.746433042820626225e-02, 1.096103029835734238e-02])
    t2[8] =  np.array([2.919026871412325197e-02, 1.742406582012695840e-02, 1.057421264616147973e-02])
    t2[9] =  np.array([2.919026793489416793e-02, 1.742406582166544995e-02, 1.057421253019524832e-02])

    z1[0] =  np.array([9.467094061069705901e+01])
    z1[1] =  np.array([9.874607563253839260e+01])
    z1[2] =  np.array([1.026600002720029323e+02])
    z1[3] =  np.array([7.320000000000000284e+01])
    z1[4] =  np.array([7.320000000000000284e+01])
    z1[5] =  np.array([7.320000000000000284e+01])
    z1[6] =  np.array([7.320000000000000284e+01])
    z1[7] =  np.array([7.320000000000000284e+01])
    z1[8] =  np.array([7.320000000000000284e+01])
    z1[9] =  np.array([7.320000000000000284e+01])

    z2[0] =  np.array([9.495664177044199050e+01])
    z2[1] =  np.array([9.010677955243804149e+01])
    z2[2] =  np.array([7.320000000000000284e+01])
    z2[3] =  np.array([1.055286918805441871e+02])
    z2[4] =  np.array([1.110504650281124128e+02])
    z2[5] =  np.array([1.111724882888512695e+02])
    z2[6] =  np.array([1.111725017770347250e+02])
    z2[7] =  np.array([1.111725017770101829e+02])
    z2[8] =  np.array([1.141315259023103152e+02])
    z2[9] =  np.array([1.141315241175528570e+02])

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

    shearExp = 0.15

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

    # ideal AEP:  array([  4.78418545e+08,   4.77544443e+08,   4.61677296e+08,
    #          4.61691234e+08,   4.66908396e+08,   4.67022058e+08,
    #          4.67022070e+08,   4.67022070e+08,   4.69662010e+08,
    #          4.69662008e+08])
    # AEP:  array([  4.41258109e+08,   4.08599492e+08,   3.74621551e+08,
    #          3.54897726e+08,   3.44499103e+08,   3.29567419e+08,
    #          3.16390777e+08,   3.04630632e+08,   2.98304205e+08,
    #          2.88832768e+08])
    # COE:  array([ 26.60602138,  28.23233037,  29.60456042,  31.01147565,
    #         32.33335586,  33.53527025,  34.67752913,  35.78081048,
    #         36.85729771,  37.86581806])
    # cost:  array([  1.17401227e+10,   1.15357158e+10,   1.10905064e+10,
    #          1.10059022e+10,   1.11388121e+10,   1.10521325e+10,
    #          1.09716504e+10,   1.08999309e+10,   1.09946869e+10,
    #          1.09368890e+10])
    # tower cost:  array([ 17959156.51984414,  17928756.97442041,  16428854.65012237,
    #         16682397.9270203 ,  18002114.60921539,  18032889.4912218 ,
    #         18033087.59745981,  18034071.44617735,  18951335.87714211,
    #         18951865.22630659])
    # wake loss:  array([  7.76734867,  14.43738939,  18.85640588,  23.13093678,
    #         26.21698263,  29.43215135,  32.25357058,  34.77168409,
    #         36.48534512,  38.50199454])
